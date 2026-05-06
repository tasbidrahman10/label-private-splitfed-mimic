from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sfl.common.config import ensure_dir
from sfl.common.logging_utils import choose_device, set_seed
from sfl.common.models import ServerTopModel
from sfl.server.fedavg import fedavg_state_dicts


@dataclass
class ClientRegistration:
    client_id: int
    client_name: str
    train_size: int
    pos_weight: float


@dataclass
class EncoderSubmission:
    state_dict: dict[str, torch.Tensor]
    sample_count: int


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
    encoder_submissions: dict[int, EncoderSubmission] = field(default_factory=dict)
    option_b_contexts: dict[str, OptionBContext] = field(default_factory=dict)
    global_encoder: dict[str, torch.Tensor] | None = None
    metrics: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    results_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        set_seed(int(self.config.get("seed", 42)))
        self.device = choose_device(str(self.config.get("device", "auto")))
        self.results_dir = ensure_dir(self.config.get("results_dir", "results/label_private_splitfed/server"))
        self.model = ServerTopModel(
            input_dim=int(self.config.get("activation_dim", 32)),
            hidden_dim=int(self.config.get("server_hidden_dim", 64)),
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("lr_server", 1e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-3)),
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

    def submit_encoder(
        self,
        client_id: int,
        state_dict: dict[str, torch.Tensor],
        sample_count: int,
    ) -> dict:
        self.encoder_submissions[client_id] = EncoderSubmission(state_dict, sample_count)
        expected = int(self.config.get("expected_num_clients") or max(len(self.clients), 1))
        submitted = len(self.encoder_submissions)
        ready = submitted >= expected
        if ready:
            ordered_ids = sorted(self.encoder_submissions)
            states = [self.encoder_submissions[cid].state_dict for cid in ordered_ids]
            counts = [self.encoder_submissions[cid].sample_count for cid in ordered_ids]
            self.global_encoder = fedavg_state_dicts(states, counts, self.device)
            self.encoder_submissions.clear()
        return {
            "ready": ready,
            "submitted_clients": submitted,
            "expected_clients": expected,
        }
