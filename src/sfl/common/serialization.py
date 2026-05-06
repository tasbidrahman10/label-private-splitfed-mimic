from __future__ import annotations

import base64
import io
from typing import Any

import torch


def tensor_to_payload(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def payload_to_tensor(payload: list[Any], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(payload, dtype=torch.float32, device=device)


def state_dict_to_base64(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    cpu_state = {key: value.detach().cpu() for key, value in state_dict.items()}
    torch.save(cpu_state, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def base64_to_state_dict(payload: str, device: torch.device | str) -> dict[str, torch.Tensor]:
    raw = base64.b64decode(payload.encode("ascii"))
    buffer = io.BytesIO(raw)
    state = torch.load(buffer, map_location=device)
    return {key: value.to(device) for key, value in state.items()}


def estimate_tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()

