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
            "gradient_scaling": cls.gradient_scaling,
            "label_flip_proxy": cls.label_flip_proxy,
            "free_rider": cls.free_rider,
            "backdoor": cls.backdoor,
            "slow_poisoner": cls.slow_poisoner,
        }
        if attack_type not in dispatch:
            raise ValueError(
                f"Unknown attack type '{attack_type}'. "
                f"Choose from: {list(dispatch)}"
            )
        return dispatch[attack_type](weights, **params)
