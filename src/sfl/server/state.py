from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sfl.common.config import ensure_dir
from sfl.common.logging_utils import choose_device, set_seed
from sfl.common.models import ServerTopModel
from sfl.server.anomaly_detector import MaliciousClientGate
from sfl.server.async_controller import AsyncRoundController
from sfl.server.robust_fedavg import RobustAsyncFedAvg
from sfl.server.staleness import StalenessDecay, StalenessRegistry
from sfl.server.trust_state import TrustState


@dataclass
class ClientRegistration:
    client_id: int
    client_name: str
    train_size: int
    pos_weight: float


@dataclass
class OptionBContext:
    client_id: int
    activation: torch.Tensor
    logits: torch.Tensor
    round_idx: int | None
    batch_idx: int | None


@dataclass
class ServerState:
    config: dict
    device: torch.device = field(init=False)
    model: ServerTopModel = field(init=False)
    optimizer: optim.Optimizer = field(init=False)
    clients: dict[int, ClientRegistration] = field(default_factory=dict)
    option_b_contexts: dict[str, OptionBContext] = field(default_factory=dict)
    global_encoder: dict[str, torch.Tensor] | None = None
    # Snapshot of global encoder from the previous round, used for SNAS cosine
    # anomaly comparison. Kept separate from global_encoder which is cleared to
    # None at round open so clients receive 404 while the window is in progress.
    prev_global_encoder: dict[str, torch.Tensor] | None = None
    metrics: list[dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    results_dir: Path = field(init=False)

    # Async extension fields
    activation_buffer: dict[int, np.ndarray] = field(default_factory=dict)
    current_round: int = 0
    staleness_registry: StalenessRegistry = field(init=False)
    async_controller: AsyncRoundController = field(init=False)
    gate: MaliciousClientGate = field(init=False)
    trust: TrustState = field(init=False)
    robust_fedavg: RobustAsyncFedAvg = field(init=False)

    def __post_init__(self) -> None:
        set_seed(int(self.config.get("seed", 42)))
        self.device = choose_device(str(self.config.get("device", "auto")))
        self.results_dir = ensure_dir(
            self.config.get("results_dir", "results/label_private_splitfed/server")
        )
        self.model = ServerTopModel(
            input_dim=int(self.config.get("activation_dim", 32)),
            hidden_dim=int(self.config.get("server_hidden_dim", 64)),
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("lr_server", 1e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-3)),
        )

        n = int(self.config.get("expected_num_clients", 3))
        client_ids = list(range(n))

        self.staleness_registry = StalenessRegistry(client_ids)
        self.async_controller = AsyncRoundController(
            client_ids=client_ids,
            base_window_seconds=float(self.config.get("async_window_seconds", 25.0)),
            max_consecutive_misses=int(self.config.get("max_consecutive_misses", 3)),
        )
        self.gate = MaliciousClientGate(
            threshold_quarantine=float(self.config.get("snas_threshold_quarantine", 0.8)),
            threshold_flag=float(self.config.get("snas_threshold_flag", 0.4)),
            alpha=float(self.config.get("snas_alpha", 0.5)),
            beta=float(self.config.get("snas_beta", 0.2)),
            gamma=float(self.config.get("snas_gamma", 0.3)),
        )
        self.trust = TrustState(
            client_ids=client_ids,
            rehab_rounds=int(self.config.get("rehab_rounds", 3)),
        )
        decay_name = str(self.config.get("staleness_decay", "exponential"))
        self.robust_fedavg = RobustAsyncFedAvg(
            decay_fn=StalenessDecay.get(decay_name),
            clip_ratio=float(self.config.get("fedavg_clip_ratio", 2.0)),
        )

    def expected_token(self) -> str:
        return str(self.config.get("auth_token", ""))

    def criterion_for(self, client_id: int) -> nn.Module:
        registration = self.clients.get(client_id)
        pos_weight = registration.pos_weight if registration else 1.0
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
        )

    def append_metric(self, row: dict) -> None:
        row = {"timestamp": time.time(), **row}
        self.metrics.append(row)
        metrics_path = self.results_dir / "server_metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def save_checkpoint(self, filename: str = "server_model_latest.pth") -> Path:
        path = self.results_dir / filename
        torch.save(self.model.state_dict(), path)
        return path

    def register_client(self, registration: ClientRegistration) -> None:
        self.clients[registration.client_id] = registration
