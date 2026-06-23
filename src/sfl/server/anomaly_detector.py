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
    ) -> None:
        self.threshold_q = threshold_quarantine
        self.threshold_f = threshold_flag
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ema_alpha = ema_alpha

        # Per-client rolling reference distributions (built from clean rounds)
        self._ref_mean: dict[int, np.ndarray] = {}
        self._ref_std: dict[int, np.ndarray] = {}
        self._initialized: set[int] = set()

    def has_reference(self, client_id: int) -> bool:
        return client_id in self._initialized

    def compute_snas(
        self,
        client_id: int,
        activations: np.ndarray,
        submitted_weights: dict[str, torch.Tensor],
        global_weights: dict[str, torch.Tensor] | None,
        staleness: int,
    ) -> float:
        """
        Compute SNAS for one client. Returns 0.0 if reference or global
        weights are not yet available (warm-up phase).
        """
        if not self.has_reference(client_id) or global_weights is None:
            return 0.0

        div  = _activation_divergence(activations, self._ref_mean[client_id], self._ref_std[client_id])
        cos  = _weight_cosine_anomaly(submitted_weights, global_weights)
        norm = _weight_norm_anomaly(submitted_weights, global_weights)
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
    ) -> dict:
        """Return detailed component scores for logging."""
        if not self.has_reference(client_id) or global_weights is None:
            return {
                "activation_divergence": None,
                "cosine_anomaly": None,
                "norm_anomaly": None,
                "snas": None,
                "staleness_tau": staleness,
            }
        div  = _activation_divergence(activations, self._ref_mean[client_id], self._ref_std[client_id])
        cos  = _weight_cosine_anomaly(submitted_weights, global_weights)
        norm = _weight_norm_anomaly(submitted_weights, global_weights)
        snas = (self.alpha * div + self.beta * cos + self.gamma * norm) / (1.0 + staleness)
        return {
            "activation_divergence": round(div, 6),
            "cosine_anomaly": round(cos, 6),
            "norm_anomaly": round(norm, 6),
            "snas": round(snas, 6),
            "staleness_tau": staleness,
        }
