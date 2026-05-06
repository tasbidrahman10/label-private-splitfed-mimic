from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sfl.client.api import SFLServerClient
from sfl.common.config import ensure_dir
from sfl.common.metrics import binary_metrics
from sfl.common.models import ClientEncoder
from sfl.common.serialization import estimate_tensor_bytes, payload_to_tensor


class BaseClientTrainer:
    def __init__(
        self,
        config: dict,
        encoder: ClientEncoder,
        train_loader,
        val_loader,
        api: SFLServerClient,
        device: torch.device,
        metrics_filename: str,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.api = api
        self.device = device
        self.client_id = int(config["client_id"])
        self.optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=float(config.get("lr_encoder", 1e-4)),
            weight_decay=float(config.get("weight_decay", 1e-3)),
        )
        self.results_dir = ensure_dir(config.get("results_dir", f"results/label_private_splitfed/client{self.client_id}"))
        self.metrics_path = self.results_dir / metrics_filename

    @torch.no_grad()
    def validate(self, round_idx: int, max_batches: int | None = None) -> dict:
        self.encoder.eval()
        labels: list[np.ndarray] = []
        logits: list[np.ndarray] = []

        for batch_idx, (x_batch, y_batch) in enumerate(self.val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x_batch = x_batch.to(self.device)
            activation = self.encoder(x_batch)
            response = self.api.predict(activation.detach())
            batch_logits = payload_to_tensor(response["logits"], self.device)
            labels.append(y_batch.numpy())
            logits.append(batch_logits.detach().cpu().numpy())

        if not labels:
            row = {"round": round_idx, "client_id": self.client_id, "val_batches": 0}
        else:
            metric_values = binary_metrics(np.vstack(labels), np.vstack(logits))
            row = {
                "round": round_idx,
                "client_id": self.client_id,
                "val_batches": len(labels),
                **metric_values,
            }
        self._append_metrics(row)
        return row

    def submit_encoder_for_fedavg(self) -> dict:
        return self.api.submit_encoder(
            self.client_id,
            self.encoder.state_dict(),
            sample_count=len(self.train_loader.dataset),
        )

    def load_global_encoder(self, max_wait_seconds: float = 300.0) -> None:
        global_state = self.api.wait_for_global_encoder(
            self.device,
            max_wait_seconds=max_wait_seconds,
        )
        self.encoder.load_state_dict(global_state)

    def save_encoder(self, filename: str) -> Path:
        path = self.results_dir / filename
        torch.save(self.encoder.state_dict(), path)
        return path

    def _append_metrics(self, row: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")


class OptionBClientTrainer(BaseClientTrainer):
    def __init__(
        self,
        config: dict,
        encoder: ClientEncoder,
        train_loader,
        val_loader,
        api: SFLServerClient,
        device: torch.device,
        pos_weight: float,
    ) -> None:
        super().__init__(
            config,
            encoder,
            train_loader,
            val_loader,
            api,
            device,
            metrics_filename="client_metrics_option_b.jsonl",
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
        )

    def train_round(self, round_idx: int, max_batches: int | None = None) -> dict:
        self.encoder.train()
        losses: list[float] = []
        activation_bytes = 0
        gradient_bytes = 0

        for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            activation = self.encoder(x_batch)
            forward_response = self.api.option_b_forward(
                self.client_id,
                activation.detach(),
                round_idx,
                batch_idx,
            )

            logits = payload_to_tensor(forward_response["logits"], self.device)
            logits.requires_grad_(True)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            backward_response = self.api.option_b_backward(
                self.client_id,
                forward_response["context_id"],
                logits.grad.detach(),
            )
            grad = payload_to_tensor(backward_response["activation_gradient"], self.device)
            activation.backward(grad)
            self.optimizer.step()

            losses.append(float(loss.detach().cpu().item()))
            activation_bytes += estimate_tensor_bytes(activation)
            gradient_bytes += estimate_tensor_bytes(grad)

        avg_loss = sum(losses) / max(len(losses), 1)
        row = {
            "round": round_idx,
            "client_id": self.client_id,
            "batches": len(losses),
            "avg_local_loss": avg_loss,
            "activation_bytes": activation_bytes,
            "gradient_bytes": gradient_bytes,
        }
        self._append_metrics(row)
        return row
