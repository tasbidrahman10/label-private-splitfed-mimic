from __future__ import annotations

import time
from typing import Any

import requests
import torch

from sfl.common.serialization import (
    base64_to_state_dict,
    state_dict_to_base64,
    tensor_to_payload,
)


class SFLServerClient:
    def __init__(self, server_url: str, auth_token: str = "", timeout: float = 120.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self.server_url}{path}",
            headers=self.headers,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def register_client(
        self,
        client_id: int,
        client_name: str,
        train_size: int,
        pos_weight: float,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/register_client",
            json={
                "client_id": client_id,
                "client_name": client_name,
                "train_size": train_size,
                "pos_weight": pos_weight,
            },
        )

    def predict(self, activation: torch.Tensor) -> dict[str, Any]:
        return self._request(
            "POST",
            "/predict",
            json={"activation": tensor_to_payload(activation)},
        )

    def option_b_forward(
        self,
        client_id: int,
        activation: torch.Tensor,
        round_idx: int,
        batch_idx: int,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/option-b/forward",
            json={
                "client_id": client_id,
                "activation": tensor_to_payload(activation),
                "round": round_idx,
                "batch": batch_idx,
            },
        )

    def option_b_backward(
        self,
        client_id: int,
        context_id: str,
        logit_gradient: torch.Tensor,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/option-b/backward",
            json={
                "client_id": client_id,
                "context_id": context_id,
                "logit_gradient": tensor_to_payload(logit_gradient),
            },
        )

    def submit_encoder(
        self,
        client_id: int,
        encoder_state: dict[str, torch.Tensor],
        sample_count: int,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/fedavg/submit_encoder",
            json={
                "client_id": client_id,
                "sample_count": sample_count,
                "state_dict": state_dict_to_base64(encoder_state),
            },
        )

    def get_global_encoder(self, device: torch.device) -> dict[str, torch.Tensor]:
        response = self._request("GET", "/fedavg/global_encoder")
        return base64_to_state_dict(response["state_dict"], device)

    def wait_for_global_encoder(
        self,
        device: torch.device,
        poll_seconds: float = 2.0,
        max_wait_seconds: float = 300.0,
    ) -> dict[str, torch.Tensor]:
        deadline = time.time() + max_wait_seconds
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                return self.get_global_encoder(device)
            except requests.HTTPError as exc:
                last_error = exc
                if exc.response is not None and exc.response.status_code != 404:
                    raise
            time.sleep(poll_seconds)
        raise TimeoutError("Timed out waiting for global encoder") from last_error

    def save_server_checkpoint(self, filename: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/checkpoint/server",
            json={"filename": filename},
        )
