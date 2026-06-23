from __future__ import annotations

import logging
from typing import Callable

import torch

from sfl.server.staleness import StalenessDecay
from sfl.server.trust_state import QUARANTINE, FLAG

logger = logging.getLogger(__name__)


class RobustAsyncFedAvg:
    """
    Weighted FedAvg aggregator for the async setting.

    Replaces the synchronous fedavg.py with an aggregator that accounts for:
      1. Staleness: each client's weight is multiplied by decay_fn(tau_i)
      2. Anomaly gating: quarantined clients are excluded entirely;
         flagged clients receive a 0.5 penalty on their weight
      3. Sample-proportional base weight (same as original FedAvg)

    Final weights are renormalised to sum to 1.0 across accepted clients.

    If every submitted client is quarantined, aggregation is skipped and
    None is returned — the global encoder from the previous round persists.
    If only one client passes the gate, its encoder becomes the global encoder
    (weight = 1.0), which is correct behaviour.
    """

    def __init__(
        self,
        decay_fn: Callable[[int], float] = StalenessDecay.exponential,
        clip_ratio: float = 2.0,
    ) -> None:
        self.decay_fn = decay_fn
        # Per-layer norm clipping: clip submitted weight norms to
        # clip_ratio × global encoder norm before aggregation.
        # Prevents gradient-scaling attacks from corrupting the global
        # encoder during warm-up rounds before SNAS detection activates.
        self.clip_ratio = clip_ratio

    def _clip_submission(
        self,
        weights: dict[str, torch.Tensor],
        reference: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Clip each tensor so its L2 norm ≤ clip_ratio × reference norm."""
        clipped = {}
        for key, w in weights.items():
            if key not in reference:
                clipped[key] = w
                continue
            ref_norm = reference[key].float().norm().item()
            sub_norm = w.float().norm().item()
            if ref_norm > 1e-8 and sub_norm > self.clip_ratio * ref_norm:
                scale = (self.clip_ratio * ref_norm) / sub_norm
                clipped[key] = (w.float() * scale).to(w.dtype)
                logger.debug("Clipped key=%s sub_norm=%.3f ref_norm=%.3f scale=%.3f", key, sub_norm, ref_norm, scale)
            else:
                clipped[key] = w
        return clipped

    def aggregate(
        self,
        submissions: dict[int, dict[str, torch.Tensor]],
        staleness_counts: dict[int, int],
        gate_decisions: dict[int, str],
        sample_counts: dict[int, int],
        device: torch.device | None = None,
        prev_global_encoder: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """
        Parameters
        ----------
        submissions        : client_id → encoder state_dict (submitted clients only)
        staleness_counts   : client_id → τ_i (all registered clients)
        gate_decisions     : client_id → 'accept' | 'flag' | 'quarantine'
        sample_counts      : client_id → dataset size (for base weight)
        device             : target device for tensors (optional)
        prev_global_encoder: previous round's global encoder for norm clipping

        Returns
        -------
        Aggregated global encoder state_dict, or None if all clients quarantined.
        """
        # Step 0 — norm clipping (applied before gate filtering)
        if prev_global_encoder is not None and self.clip_ratio is not None:
            submissions = {
                cid: self._clip_submission(w, prev_global_encoder)
                for cid, w in submissions.items()
            }

        # Step 1 — filter quarantined clients
        accepted = {
            cid: w for cid, w in submissions.items()
            if gate_decisions.get(cid, "accept") != QUARANTINE
        }

        if not accepted:
            logger.warning("All submitted clients quarantined — skipping aggregation this round.")
            return None

        # Step 2 — compute raw weights
        raw_weights: dict[int, float] = {}
        for cid in accepted:
            tau = staleness_counts.get(cid, 0)
            staleness_w = self.decay_fn(tau)
            flag_penalty = 0.5 if gate_decisions.get(cid) == FLAG else 1.0
            raw_weights[cid] = sample_counts[cid] * staleness_w * flag_penalty

        # Step 3 — normalise
        total = sum(raw_weights.values())
        norm_weights = {cid: raw_weights[cid] / total for cid in accepted}

        logger.info(
            "RobustAsyncFedAvg: aggregating %d clients | normalised weights: %s",
            len(accepted),
            {cid: round(w, 4) for cid, w in norm_weights.items()},
        )

        # Step 4 — weighted average
        keys = list(next(iter(accepted.values())).keys())
        global_state: dict[str, torch.Tensor] = {}
        for key in keys:
            tensors = [
                norm_weights[cid] * accepted[cid][key].float()
                for cid in accepted
            ]
            agg = torch.stack(tensors).sum(dim=0)
            global_state[key] = agg.to(device) if device is not None else agg

        return global_state

    def log_weights(
        self,
        submissions: dict[int, dict],
        staleness_counts: dict[int, int],
        gate_decisions: dict[int, str],
        sample_counts: dict[int, int],
    ) -> dict[int, dict]:
        """
        Return per-client weight breakdown for JSONL logging — does not
        perform aggregation.
        """
        accepted = {
            cid: w for cid, w in submissions.items()
            if gate_decisions.get(cid, "accept") != QUARANTINE
        }
        if not accepted:
            return {}

        raw: dict[int, float] = {}
        for cid in accepted:
            tau = staleness_counts.get(cid, 0)
            raw[cid] = (
                sample_counts[cid]
                * self.decay_fn(tau)
                * (0.5 if gate_decisions.get(cid) == FLAG else 1.0)
            )
        total = sum(raw.values())
        return {
            cid: {
                "staleness_tau": staleness_counts.get(cid, 0),
                "staleness_weight": self.decay_fn(staleness_counts.get(cid, 0)),
                "flag_penalty": 0.5 if gate_decisions.get(cid) == FLAG else 1.0,
                "aggregation_weight_normalized": raw[cid] / total,
                "included_in_aggregation": True,
            }
            for cid in accepted
        }
