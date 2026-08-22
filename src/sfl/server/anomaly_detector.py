from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from sfl.server.trust_state import ACCEPT, FLAG, QUARANTINE

logger = logging.getLogger(__name__)

# Warm-up rounds before gate decisions are made.
# During warm-up the reference distributions are built but no client is
# flagged or quarantined — early rounds naturally look "anomalous" because
# the reference has not converged yet.
WARMUP_ROUNDS = 1


def _activation_divergence(
    current: np.ndarray,
    ref_mean: np.ndarray,
    ref_std: np.ndarray,
) -> float:
    """
    Gaussian-approximation symmetric KL divergence between the current
    activation batch distribution and the client's rolling reference.

    current: shape (batch_size, activation_dim)
    Returns a scalar ≥ 0; higher means more drift from the reference.
    """
    curr_mean = current.mean(axis=0)
    curr_std = current.std(axis=0) + 1e-8
    ref_std = ref_std + 1e-8

    # KL(current || reference) under Gaussian assumption, per dimension
    kl = (
        np.log(ref_std / curr_std)
        + (curr_std ** 2 + (curr_mean - ref_mean) ** 2) / (2.0 * ref_std ** 2)
        - 0.5
    )
    return float(kl.mean())


def _weight_cosine_anomaly(
    submitted: dict[str, torch.Tensor],
    global_weights: dict[str, torch.Tensor],
) -> float:
    """
    1 - cosine_similarity between submitted encoder weights and the last
    global encoder, averaged across all parameter tensors.

    Detects direction changes (label-flip proxy, backdoor, free rider).
    Returns 0 for identical directions, approaching 1 for opposite directions.
    NOTE: scale-invariant — does NOT detect gradient scaling attacks alone.
    """
    scores: list[float] = []
    for key in submitted:
        if key not in global_weights:
            continue
        w_sub = submitted[key].flatten().float()
        w_glb = global_weights[key].flatten().float()
        sim = F.cosine_similarity(w_sub.unsqueeze(0), w_glb.unsqueeze(0)).item()
        scores.append(1.0 - sim)
    return float(np.mean(scores)) if scores else 0.0


def _weight_norm_anomaly(
    submitted: dict[str, torch.Tensor],
    global_weights: dict[str, torch.Tensor],
) -> float:
    """
    Normalised log-ratio of L2 norms between submitted and global encoder weights,
    averaged across parameter tensors and capped at 1.0.

    Detects magnitude-based attacks (gradient scaling) that cosine anomaly misses.

    For clean clients: norm_ratio ≈ 1.0 → log(1) = 0.0 → anomaly ≈ 0.0
    For ×10 scaling:   norm_ratio ≈ 10.0 → |log(10)| / log(100) ≈ 0.77 → anomaly ≈ 0.77
    For free rider:    norm_ratio unpredictable, but cosine_anomaly catches those anyway.
    """
    scores: list[float] = []
    LOG_CAP = np.log(100.0)  # normalisation constant: ×100 attack → anomaly = 1.0
    for key in submitted:
        if key not in global_weights:
            continue
        norm_sub = submitted[key].float().norm().item()
        norm_glb = global_weights[key].float().norm().item()
        if norm_glb < 1e-8:
            continue
        ratio = norm_sub / norm_glb
        log_ratio = abs(np.log(max(ratio, 1e-8)))
        scores.append(min(log_ratio / LOG_CAP, 1.0))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# P1-5: update-space ("delta") anomaly signals
# ---------------------------------------------------------------------------
#
# The absolute-weight signals above compare theta_i against the global encoder
# theta^g. Because every client starts the round *from* theta^g and takes a
# small step, theta_i = theta^g + delta_i with ||delta_i|| << ||theta^g||, so
# cos(theta_i, theta^g) is close to 1 for honest and adversarial clients alike
# -- the shared base dominates the inner product and the discriminative part is
# a small perturbation on top. That is the review's deepest technical
# objection, and it is correct as stated about the cosine term.
#
# The functions below recompute both weight signals on the *update*
# delta_i = theta_i - theta^g instead, where the shared base cancels exactly.
#
# The reference direction is the mean of the *other* clients' updates
# (leave-one-out). A pooled reference including the client under test would let
# an attacker drag the reference toward itself, which at n=3 is a third of the
# mass -- enough to blunt its own score. Leave-one-out removes that channel.
#
# Selected by config `snas_anomaly_basis`; the default is "absolute", which is
# the behaviour every published result was produced with.

# Saturating a non-finite update at maximum anomaly is the correct behaviour and
# mirrors the absolute basis, whose min(log_ratio / LOG_CAP, 1.0) cap already
# absorbs an infinite norm ratio. Without it the score is NaN, which compares
# False against every threshold and therefore reads as "clean".
#
# It is DISABLED while the P1-5 `deltadet` arm is being collected. The first 11
# of that arm's 30 runs were produced before the guard existed, and switching it
# on midway would make the same degenerate rounds score 1.0 in some seeds and
# NaN in others -- which would inflate the delta basis's apparent performance on
# the straggler experiment, where those rounds are concentrated. One arm, one
# implementation.
#
# TO RE-ENABLE: set True, restart the server, and rerun the whole deltadet arm
# (delete its entries from results/p08_rerun_progress.json first). Until then
# analyze_anomaly_basis.py excludes non-finite scores and reports the count.
SATURATE_NONFINITE_DELTA = False


def _flat_delta(
    submitted: dict[str, torch.Tensor],
    global_weights: dict[str, torch.Tensor],
) -> np.ndarray:
    """theta_i - theta^g, flattened over the float tensors the two share."""
    parts: list[np.ndarray] = []
    for key in submitted:
        if key not in global_weights:
            continue
        sub = submitted[key].detach().cpu()
        glb = global_weights[key].detach().cpu()
        # Test dtype BEFORE casting. BatchNorm's num_batches_tracked is an
        # int64 counter in the tens of thousands; cast first and it passes the
        # float test, then dominates the delta vector and swamps every real
        # weight difference. Same class of layout bug as the omitted BatchNorm
        # buffers that made FLTrust degenerate (SESSION_2026-08-18_part2 §4).
        if not sub.is_floating_point() or sub.shape != glb.shape:
            continue
        parts.append((sub.float() - glb.float()).numpy().ravel())
    return np.concatenate(parts) if parts else np.array([])


def compute_deltas(
    all_submissions: dict[int, dict[str, torch.Tensor]],
    global_weights: dict[str, torch.Tensor],
) -> dict[int, np.ndarray]:
    """Flattened update vector for every client submitted this round."""
    return {
        cid: _flat_delta(weights, global_weights)
        for cid, weights in all_submissions.items()
    }


def _delta_cosine_anomaly(client_id: int, deltas: dict[int, np.ndarray]) -> float:
    """1 - cos(delta_i, mean *direction* of peers' updates).

    Peers are unit-normalised before averaging. Using the raw mean instead is
    actively broken at this federation size: a 10x gradient-scaling attacker
    produces an update ~900x the honest norm, so for each honest client the
    two-peer mean is essentially the attacker's vector, and the honest clients
    score *higher* than the attacker (measured: honest 1.06 and 1.10 against
    attacker 1.08 -- the ranking inverts). Normalising first removes magnitude
    from what is supposed to be a direction measure; magnitude is already
    handled separately by _delta_norm_anomaly.

    Residual limitation at n=3: one attacker is still half of every honest
    client's two-peer reference, which lifts honest scores to ~0.34-0.38
    against the attacker's ~1.08. Separation survives, but the absolute
    basis's thresholds (flag 0.22, quarantine 0.35) do not transfer to this
    basis and must be recalibrated before the decisions mean anything.

    **At n=2 this signal carries no information at all**, and provably so. With
    a single peer j, the reference for i is delta_j / ||delta_j||, so

        anomaly_i = 1 - cos(delta_i, delta_j) = anomaly_j

    because cosine is symmetric. _delta_norm_anomaly degenerates the same way:
    |log(||d_i|| / ||d_j||)| = |log(||d_j|| / ||d_i||)|. Attacker and honest
    client therefore receive *identical* scores, differing only through the
    staleness denominator. Confirmed in the logs -- in the straggler
    experiment, where one client misses the window and n drops to 2, the two
    remaining clients record byte-identical cosine and norm anomalies in every
    such round. Any peer-referenced detector inherits this; it is the detection
    analogue of the coordinate-wise median losing its breakdown point at n=2.
    """
    own = deltas.get(client_id)
    peers = [d for cid, d in deltas.items() if cid != client_id and d.size]
    if own is None or not own.size or not peers:
        return 0.0
    # With the gate off a compounding attacker overflows float32 within ten
    # rounds, so the update stops being finite. The absolute basis absorbs this
    # in its min(..., 1.0) cap; without an equivalent here the score becomes
    # NaN, which compares False against every threshold and reads as "clean".
    # Saturate instead: a non-finite update is maximally anomalous.
    if SATURATE_NONFINITE_DELTA:
        if not np.all(np.isfinite(own)):
            return 1.0
        peers = [d for d in peers if np.all(np.isfinite(d))]
        if not peers:
            return 1.0
    directions = []
    for d in peers:
        norm = float(np.linalg.norm(d))
        if norm > 1e-12:
            directions.append(d / norm)
    if not directions:
        return 0.0
    reference = np.mean(np.stack(directions, axis=0), axis=0)
    denom = float(np.linalg.norm(own) * np.linalg.norm(reference))
    if denom < 1e-12:
        return 0.0
    return float(1.0 - float(np.dot(own, reference)) / denom)


def _delta_norm_anomaly(client_id: int, deltas: dict[int, np.ndarray]) -> float:
    """|log(||delta_i|| / median peer ||delta_j||)| / log(100), capped at 1.

    Same normalisation as the absolute-weight norm term, so the two bases stay
    on one scale and the configured thresholds keep their meaning. The peer
    median rather than mean keeps the reference itself robust to the attacker.
    """
    own = deltas.get(client_id)
    peers = [d for cid, d in deltas.items() if cid != client_id and d.size]
    if own is None or not own.size or not peers:
        return 0.0
    if SATURATE_NONFINITE_DELTA:
        if not np.all(np.isfinite(own)):
            return 1.0
        peers = [d for d in peers if np.all(np.isfinite(d))]
        if not peers:
            return 1.0
    own_norm = float(np.linalg.norm(own))
    ref_norm = float(np.median([float(np.linalg.norm(d)) for d in peers]))
    if ref_norm < 1e-12:
        return 0.0
    ratio = max(own_norm / ref_norm, 1e-8)
    return float(min(abs(np.log(ratio)) / np.log(100.0), 1.0))


class MaliciousClientGate:
    """
    Computes the Staleness-Normalized Anomaly Score (SNAS) for each client
    and issues gate decisions: 'accept', 'flag', or 'quarantine'.

    SNAS_i = (alpha * activation_divergence_i
              + beta  * cosine_anomaly_i
              + gamma * norm_anomaly_i)
             / (1 + tau_i)

    Three signals:
    - activation_divergence: detects behavioural drift during training
    - cosine_anomaly:        detects direction changes in submitted weights
                             (label-flip, backdoor, free rider)
    - norm_anomaly:          detects magnitude attacks that cosine misses
                             (gradient scaling × large factor)

    The denominator dampens the score for legitimately stale clients.
    alpha + beta + gamma should sum to 1.0.
    """

    def __init__(
        self,
        threshold_quarantine: float = 0.8,
        threshold_flag: float = 0.4,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.3,
        ema_alpha: float = 0.1,
        anomaly_basis: str = "absolute",
    ) -> None:
        self.threshold_q = threshold_quarantine
        self.threshold_f = threshold_flag
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ema_alpha = ema_alpha
        if anomaly_basis not in ("absolute", "delta"):
            raise ValueError(
                f"snas_anomaly_basis must be 'absolute' or 'delta', got {anomaly_basis!r}"
            )
        # P1-5. 'absolute' compares theta_i against theta^g and is what every
        # published number used. 'delta' compares the update theta_i - theta^g
        # against the other clients' updates, cancelling the shared base.
        self.anomaly_basis = anomaly_basis

        # Per-client rolling reference distributions (built from clean rounds)
        self._ref_mean: dict[int, np.ndarray] = {}
        self._ref_std: dict[int, np.ndarray] = {}
        self._initialized: set[int] = set()

    def has_reference(self, client_id: int) -> bool:
        return client_id in self._initialized

    def _weight_signals(
        self,
        client_id: int,
        submitted_weights: dict[str, torch.Tensor],
        global_weights: dict[str, torch.Tensor],
        all_submissions: dict[int, dict[str, torch.Tensor]] | None,
    ) -> tuple[float, float]:
        """(cosine_anomaly, norm_anomaly) under the configured basis.

        Falls back to the absolute basis when delta mode is selected but the
        round has no peer submissions to reference -- a single-client round
        carries no update-space signal at all, and silently returning 0.0 there
        would read as "clean" rather than "not measurable".
        """
        if self.anomaly_basis == "delta" and all_submissions and len(all_submissions) > 1:
            deltas = compute_deltas(all_submissions, global_weights)
            return (
                _delta_cosine_anomaly(client_id, deltas),
                _delta_norm_anomaly(client_id, deltas),
            )
        return (
            _weight_cosine_anomaly(submitted_weights, global_weights),
            _weight_norm_anomaly(submitted_weights, global_weights),
        )

    def compute_snas(
        self,
        client_id: int,
        activations: np.ndarray,
        submitted_weights: dict[str, torch.Tensor],
        global_weights: dict[str, torch.Tensor] | None,
        staleness: int,
        all_submissions: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> float:
        """
        Compute SNAS for one client. Returns 0.0 if reference or global
        weights are not yet available (warm-up phase).

        all_submissions is every encoder submitted this round, required only by
        the 'delta' basis for its leave-one-out peer reference.
        """
        if not self.has_reference(client_id) or global_weights is None:
            return 0.0

        div  = _activation_divergence(activations, self._ref_mean[client_id], self._ref_std[client_id])
        cos, norm = self._weight_signals(
            client_id, submitted_weights, global_weights, all_submissions
        )
        snas = (self.alpha * div + self.beta * cos + self.gamma * norm) / (1.0 + staleness)
        logger.debug(
            "Client %d SNAS=%.4f (div=%.4f, cos=%.4f, norm=%.4f, tau=%d)",
            client_id, snas, div, cos, norm, staleness,
        )
        return snas

    def gate(self, client_id: int, snas: float, round_idx: int) -> str:
        """
        Map SNAS to a gate decision.
        During warm-up (round_idx <= WARMUP_ROUNDS) always returns 'accept'.
        """
        if round_idx <= WARMUP_ROUNDS:
            return ACCEPT
        if snas > self.threshold_q:
            logger.warning("Client %d QUARANTINED (SNAS=%.4f > %.2f).", client_id, snas, self.threshold_q)
            return QUARANTINE
        if snas > self.threshold_f:
            logger.warning("Client %d FLAGGED (SNAS=%.4f > %.2f).", client_id, snas, self.threshold_f)
            return FLAG
        return ACCEPT

    def update_reference(self, client_id: int, activations: np.ndarray) -> None:
        """
        Update the rolling reference distribution for a client using EMA.
        Should be called after a clean (accepted) round.
        """
        curr_mean = activations.mean(axis=0)
        curr_std = activations.std(axis=0) + 1e-8

        if client_id not in self._initialized:
            self._ref_mean[client_id] = curr_mean.copy()
            self._ref_std[client_id] = curr_std.copy()
            self._initialized.add(client_id)
        else:
            a = self.ema_alpha
            self._ref_mean[client_id] = (1.0 - a) * self._ref_mean[client_id] + a * curr_mean
            self._ref_std[client_id] = (1.0 - a) * self._ref_std[client_id] + a * curr_std

    def snas_metrics(
        self,
        client_id: int,
        activations: np.ndarray,
        submitted_weights: dict[str, torch.Tensor],
        global_weights: dict[str, torch.Tensor] | None,
        staleness: int,
        all_submissions: dict[int, dict[str, torch.Tensor]] | None = None,
    ) -> dict:
        """Return detailed component scores for logging."""
        if not self.has_reference(client_id) or global_weights is None:
            return {
                "activation_divergence": None,
                "cosine_anomaly": None,
                "norm_anomaly": None,
                "snas": None,
                "staleness_tau": staleness,
                "anomaly_basis": self.anomaly_basis,
            }
        div  = _activation_divergence(activations, self._ref_mean[client_id], self._ref_std[client_id])
        cos, norm = self._weight_signals(
            client_id, submitted_weights, global_weights, all_submissions
        )
        snas = (self.alpha * div + self.beta * cos + self.gamma * norm) / (1.0 + staleness)
        return {
            "activation_divergence": round(div, 6),
            "cosine_anomaly": round(cos, 6),
            "norm_anomaly": round(norm, 6),
            "snas": round(snas, 6),
            "staleness_tau": staleness,
            # Recorded per row so a mixed log can never be misread as one basis.
            "anomaly_basis": self.anomaly_basis,
        }
