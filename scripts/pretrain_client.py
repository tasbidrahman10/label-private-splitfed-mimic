"""
Config-driven client pretraining.

Port of the three duplicated notebooks/client{0,1,2}_model.ipynb notebooks,
parameterized over any configs/clients/*.yaml so it runs unchanged for n=3,
n=6, or any other client count. Fits a StandardScaler on this client's local
training split, saves it to model_dir/scaler.pkl, then trains a temporary
encoder+classifier head (Adam @ lr_encoder, ReduceLROnPlateau, up to 80
epochs, early stop at patience 8) so ClientEncoder starts from a warm,
discriminative representation before federated rounds begin.

The scaler is fit first and the training/val loaders are then built through
sfl.client.dataset.load_client_dataloaders — the same function used during
real federated training — so pretraining sees exactly the tensors federated
training will.

Usage:
    python scripts/pretrain_client.py --config configs/clients/client0_micu.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sfl.client.dataset import VASO_COLS, load_client_dataloaders
from sfl.common.config import client_config_paths, ensure_dir, load_yaml, resolve_path
from sfl.common.logging_utils import choose_device, set_seed
from sfl.common.models import ClientEncoder


class TemporaryClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LocalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = ClientEncoder(input_dim, hidden_dim)
        self.classifier = TemporaryClassifier(hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def fit_and_save_scaler(config: dict) -> None:
    """Fit on continuous features from this client's training split (same
    seed/test_size/stratify as load_client_dataloaders, so it fits on exactly
    the rows that end up as "train"). Vasopressor columns are binary and left
    unscaled."""
    data_path = resolve_path(config["data_path"])
    model_dir = ensure_dir(config["model_dir"])
    target_col = config.get("target_col", "mortality")
    seed = int(config.get("seed", 42))

    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != target_col]
    input_dim = int(config.get("input_dim", 266))
    if len(feature_cols) != input_dim:
        raise ValueError(f"{data_path}: expected {input_dim} features, found {len(feature_cols)}")

    x_all = df[feature_cols].values.astype(np.float32)
    y_all = df[target_col].values.astype(np.float32)

    x_train_raw, _, _, _ = train_test_split(
        x_all, y_all, test_size=0.2, random_state=seed, stratify=y_all,
    )

    vaso_present = [c for c in VASO_COLS if c in feature_cols]
    scale_cols = [c for c in feature_cols if c not in vaso_present]
    scale_idx = [feature_cols.index(c) for c in scale_cols]

    scaler = StandardScaler()
    scaler.fit(x_train_raw[:, scale_idx])

    with (model_dir / "scaler.pkl").open("wb") as fh:
        pickle.dump(scaler, fh)


def pretrain(config: dict, resume: bool = True, force: bool = False) -> None:
    set_seed(int(config.get("seed", 42)))
    device = choose_device("auto")
    model_dir = ensure_dir(config["model_dir"])
    client_name = config.get("client_name", "client")

    done_marker = model_dir / "pretrain_done.json"
    if done_marker.exists() and not force:
        prev = json.loads(done_marker.read_text())
        print(f"[{client_name}] already pretrained "
              f"(best epoch {prev['best_epoch']}, val_loss={prev['best_val_loss']:.4f}, "
              f"input_dim={prev.get('input_dim')}, at {prev.get('completed_at')}). "
              f"Use --force to retrain.")
        return

    fit_and_save_scaler(config)
    train_loader, val_loader, pos_weight_value = load_client_dataloaders(config, device)

    input_dim = int(config.get("input_dim", 266))
    hidden_dim = int(config.get("encoder_hidden_dim", 64))
    model = LocalModel(input_dim, hidden_dim).to(device)

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.get("lr_encoder", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-3)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    num_epochs = 80
    patience = 8
    best_val_loss = float("inf")
    patience_ctr = 0
    best_epoch = 0
    start_epoch = 0
    best_path = model_dir / "best.pth"
    ckpt_path = model_dir / "pretrain_checkpoint.pth"

    # Resume from the last completed epoch if a checkpoint is present. The
    # checkpoint carries optimizer, scheduler, early-stopping bookkeeping and
    # RNG state, so a resumed run continues the same trajectory rather than
    # restarting the schedule with fresh randomness.
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_loss = ckpt["best_val_loss"]
        best_epoch = ckpt["best_epoch"]
        patience_ctr = ckpt["patience_ctr"]
        start_epoch = ckpt["epoch"]
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        print(f"[{client_name}] resuming from epoch {start_epoch + 1} "
              f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})")

    print(f"[{client_name}] pos_weight={pos_weight_value:.2f} "
          f"train_batches={len(train_loader)} val_batches={len(val_loader)} device={device}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                val_loss_sum += criterion(logits, labels).item()
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        val_loss = val_loss_sum / len(val_loader)
        probs = np.concatenate(all_probs).ravel()
        targets = np.concatenate(all_targets).ravel()
        auroc = roc_auc_score(targets, probs)

        scheduler.step(val_loss)

        print(f"[{client_name}] epoch {epoch + 1:>2}/{num_epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auroc={auroc:.4f}")

        stop_early = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[{client_name}] early stop at epoch {epoch + 1} "
                      f"(best epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                stop_early = True

        # Checkpoint after the epoch's bookkeeping is settled, so a resume
        # restarts at the next epoch with identical state. Written to a temp
        # file and moved into place so an interrupt mid-write cannot leave a
        # truncated checkpoint behind.
        tmp_path = ckpt_path.with_suffix(".tmp")
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "patience_ctr": patience_ctr,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }, tmp_path)
        tmp_path.replace(ckpt_path)

        if stop_early:
            break

    model.load_state_dict(torch.load(best_path, weights_only=True))
    torch.save(model.state_dict(), model_dir / "local_model.pth")
    torch.save(model.encoder.state_dict(), model_dir / "encoder.pth")

    # Completion marker — lets a batch driver skip finished clients on restart.
    (model_dir / "pretrain_done.json").write_text(json.dumps({
        "client_name": client_name,
        "client_id": int(config["client_id"]),
        "input_dim": input_dim,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "data_path": str(config["data_path"]),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    ckpt_path.unlink(missing_ok=True)
    print(f"[{client_name}] done. best epoch {best_epoch}, val_loss={best_val_loss:.4f} -> "
          f"saved encoder.pth / local_model.pth / best.pth / scaler.pkl in {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain one client's encoder before federated rounds. "
                    "Interrupt-safe: resumes from the last completed epoch and "
                    "skips clients that already finished."
    )
    parser.add_argument("--config", help="Path to a client YAML. Omit to run every client in the active profile.")
    parser.add_argument("--all", action="store_true",
                        help="Pretrain every client in the active profile (see SFL_CLIENT_CONFIG_DIR).")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if a completion marker exists.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any epoch checkpoint and start from scratch.")
    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("pass --config <yaml> or --all")

    paths = [Path(args.config)] if args.config else client_config_paths()
    for i, path in enumerate(paths, 1):
        if len(paths) > 1:
            print(f"\n=== [{i}/{len(paths)}] {path.name} ===")
        pretrain(load_yaml(path), resume=not args.no_resume, force=args.force)


if __name__ == "__main__":
    main()
