from __future__ import annotations

import torch


def fedavg_state_dicts(
    states: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No client states supplied for FedAvg")
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("FedAvg sample counts must sum to a positive value")

    averaged: dict[str, torch.Tensor] = {}
    for key in states[0]:
        averaged[key] = sum(
            state[key].float().to(device) * (count / total)
            for state, count in zip(states, sample_counts)
        )
    return averaged

