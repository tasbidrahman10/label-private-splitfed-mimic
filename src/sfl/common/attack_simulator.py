from __future__ import annotations

import copy

import torch


class AttackSimulator:
    """
    Wraps a client encoder submission with an adversarial transformation.

    All methods take a clean state_dict and return a poisoned copy.
    The original dict is never mutated.

    Attack types (from the guideline):
        gradient_scaling  — multiply all weights by a large scalar
        label_flip_proxy  — add large structured Gaussian noise
        free_rider        — submit random weights of the same shape
        backdoor          — perturb a small fraction of weights by a large amount
        slow_poisoner     — gradient_scaling applied after deliberate delay
                            (caller must also register τ=delay_rounds in StalenessRegistry)
    """

    @staticmethod
    def gradient_scaling(
        weights: dict[str, torch.Tensor],
        scale: float = 10.0,
    ) -> dict[str, torch.Tensor]:
        """Multiply all floating-point weight tensors by a large scalar."""
        return {
            k: v.clone().float() * scale if v.is_floating_point() else v.clone()
            for k, v in weights.items()
        }

    @staticmethod
    def label_flip_proxy(
        weights: dict[str, torch.Tensor],
        noise_std: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        """Add large structured Gaussian noise to floating-point tensors."""
        return {
            k: v.clone() + torch.randn_like(v.float()) * noise_std if v.is_floating_point() else v.clone()
            for k, v in weights.items()
        }

    @staticmethod
    def free_rider(
        weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Submit random weights — float tensors randomised, integer tensors kept."""
        return {
            k: torch.randn_like(v.float()) if v.is_floating_point() else v.clone()
            for k, v in weights.items()
        }

    @staticmethod
    def backdoor(
        weights: dict[str, torch.Tensor],
        trigger_scale: float = 5.0,
        trigger_fraction: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Perturb a random fraction of floating-point weights by a large trigger value."""
        poisoned = copy.deepcopy(weights)
        for k, v in poisoned.items():
            if not v.is_floating_point():
                continue
            mask = (torch.rand_like(v.float()) < trigger_fraction).float()
            poisoned[k] = v + mask * trigger_scale
        return poisoned

    @staticmethod
    def slow_poisoner(
        weights: dict[str, torch.Tensor],
        scale: float = 5.0,
    ) -> dict[str, torch.Tensor]:
        """
        Gradient scaling applied after a deliberate delay.

        The delay itself must be configured in multi_runner.py via delay_config
        so that the staleness registry records τ = delay_rounds for this client.
        This simulates the slow poisoner pattern: arrive late AND submit poisoned
        weights, exploiting the staleness dampening in SNAS.
        """
        return AttackSimulator.gradient_scaling(weights, scale=scale)

    # ------------------------------------------------------------------
    # Adaptive attacks (Task 3) — attacker knows SNAS structure
    # ------------------------------------------------------------------

    @staticmethod
    def threshold_aware_scaling(
        weights: dict[str, torch.Tensor],
        beta: float = 0.35,
        gamma: float = 0.65,
        threshold_quarantine: float = 0.35,
        safety_margin: float = 0.9,
        norm_cap_log_base: float = 100.0,
    ) -> tuple[dict[str, torch.Tensor], float]:
        """
        Attack A — Threshold-Aware Gradient Scaling ('Just-Under').

        Binary-searches a scale factor s such that the resulting SNAS
        (assuming tau=0, on-time submission) lands at
        safety_margin * threshold_quarantine. This maximises damage while
        staying below the quarantine threshold.

        cosine_anomaly = 0 because scaling preserves weight direction exactly.
        norm_anomaly = |log(s)| / log(norm_cap_log_base), capped at 1.

        Returns (poisoned_weights, chosen_scale_factor).
        Caller should log the scale factor per round.
        """
        import numpy as np

        def snas_at_scale(s: float) -> float:
            cosine_anomaly = 0.0  # scaling preserves direction exactly
            single_ratio = min(abs(np.log(max(s, 1e-9))) / np.log(norm_cap_log_base), 1.0)
            norm_anomaly = single_ratio  # identical for all layers at uniform scale
            return beta * cosine_anomaly + gamma * norm_anomaly

        lo, hi = 1.0, 50.0
        target = safety_margin * threshold_quarantine
        for _ in range(40):  # binary search
            mid = (lo + hi) / 2.0
            if snas_at_scale(mid) < target:
                lo = mid
            else:
                hi = mid
        scale = lo

        poisoned = {
            k: v.clone().float() * scale if v.is_floating_point() else v.clone()
            for k, v in weights.items()
        }
        return poisoned, scale

    @staticmethod
    def slow_ramp(
        weights: dict[str, torch.Tensor],
        round_num: int,
        ramp_rate: float = 1.15,
        start_scale: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], float]:
        """
        Attack C — Slow-Ramp Mimicry.

        Gradually increases scaling factor across rounds:
        scale(r) = start_scale * ramp_rate ** (r - 1)

        Tests whether SNAS's per-round memoryless scoring allows a slow
        drift to accumulate damage before crossing the quarantine threshold.

        Returns (poisoned_weights, scale_factor_this_round).
        """
        scale = start_scale * (ramp_rate ** (round_num - 1))
        poisoned = {
            k: v.clone().float() * scale if v.is_floating_point() else v.clone()
            for k, v in weights.items()
        }
        return poisoned, scale

    @classmethod
    def apply(
        cls,
        weights: dict[str, torch.Tensor],
        attack_type: str,
        params: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Apply a named attack. Used by multi_runner.py attack_config dispatch.

        Example attack_config entry:
            {'type': 'gradient_scaling', 'params': {'scale': 10.0}}
        """
        params = params or {}
        dispatch = {
            "gradient_scaling":        cls.gradient_scaling,
            "label_flip_proxy":        cls.label_flip_proxy,
            "free_rider":              cls.free_rider,
            "backdoor":                cls.backdoor,
            "slow_poisoner":           cls.slow_poisoner,
            # Adaptive attacks (Task 3)
            "threshold_aware_scaling": lambda w, **kw: cls.threshold_aware_scaling(w, **kw)[0],
            "slow_ramp":               lambda w, **kw: cls.slow_ramp(w, **kw)[0],
        }
        if attack_type not in dispatch:
            raise ValueError(
                f"Unknown attack type '{attack_type}'. "
                f"Choose from: {list(dispatch)}"
            )
        return dispatch[attack_type](weights, **params)
