from __future__ import annotations

import time
from typing import Any

import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sfl.common.serialization import (
    base64_to_state_dict,
    state_dict_to_base64,
    tensor_to_payload,
)


class SFLServerClient:
    """HTTP client for the split-learning server.

    Requests go through a persistent ``requests.Session`` rather than
    module-level ``requests.request`` calls. A training run issues roughly
    ``rounds x batches x 2`` requests (forward + backward per batch), which is
    ~12k for a 10-round/200-batch run. Without keep-alive each of those opens
    and closes a TCP connection, and every closed socket occupies an ephemeral
    port in TIME_WAIT for ~120s on Windows — enough to exhaust the ~16k dynamic
    port range within a run or two and fail with WinError 10048. Pooling keeps
    the whole run on a handful of reused connections.
    """

    def __init__(self, server_url: str, auth_token: str = "", timeout: float = 120.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        self._session = requests.Session()
        self._session.headers.update(self.headers)
        # A pooled connection can still go stale if the server closes it while
        # idle (long async windows / delay sleeps). Retrying connection-level
        # failures re-establishes the socket transparently instead of failing the
        # whole run. allowed_methods=False so POSTs retry too: these errors occur
        # before the request is delivered, so replaying it is safe. read=0 keeps
        # us from replaying a request the server may already have processed.
        retry = Retry(
            total=4,
            connect=4,
            read=0,
            status=0,
            backoff_factor=0.3,
            allowed_methods=False,
        )
        # pool_maxsize above the number of concurrent callers keeps urllib3 from
        # discarding (and thus re-opening) connections when the pool is full.
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=32, max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "SFLServerClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = self._session.request(
            method,
            f"{self.server_url}{path}",
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

    def open_fedavg_round(self) -> dict[str, Any]:
        """Signal the server to open the async submission window for this round."""
        return self._request("POST", "/fedavg/open_round")

    def reset_server(self, config_overrides: dict | None = None) -> dict[str, Any]:
        """Reinitialise server state (for automated multi-seed / sweep runs)."""
        return self._request("POST", "/admin/reset",
                             json={"config_overrides": config_overrides or {}})

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
