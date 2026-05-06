from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from sfl.common.config import resolve_path


VASO_COLS = [
    "vaso_dopamine",
    "vaso_epinephrine",
    "vaso_norepinephrine",
    "vaso_phenylephrine",
    "vaso_vasopressin",
]


def load_client_dataloaders(config: dict, device: torch.device) -> tuple[DataLoader, DataLoader, float]:
    data_path = resolve_path(config["data_path"])
    model_dir = resolve_path(config["model_dir"])
    target_col = config.get("target_col", "mortality")
    seed = int(config.get("seed", 42))
    batch_size = int(config.get("batch_size", 64))

    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col != target_col]
    input_dim = int(config.get("input_dim", 266))
    if len(feature_cols) != input_dim:
        raise ValueError(f"{data_path}: expected {input_dim} features, found {len(feature_cols)}")

    x_all = df[feature_cols].values.astype(np.float32)
    y_all = df[target_col].values.astype(np.float32)

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=seed,
        stratify=y_all,
    )

    vaso_present = [col for col in VASO_COLS if col in feature_cols]
    scale_cols = [col for col in feature_cols if col not in vaso_present]
    scale_idx = [feature_cols.index(col) for col in scale_cols]
    vaso_idx = [feature_cols.index(col) for col in vaso_present]

    scaler_path = model_dir / "scaler.pkl"
    with scaler_path.open("rb") as fh:
        scaler = pickle.load(fh)

    def preprocess(raw: np.ndarray) -> np.ndarray:
        x_cont = scaler.transform(raw[:, scale_idx])
        x_cont = np.clip(x_cont, -5.0, 5.0)
        if not vaso_idx:
            return x_cont.astype(np.float32)
        return np.hstack([x_cont, raw[:, vaso_idx]]).astype(np.float32)

    x_train = preprocess(x_train_raw)
    x_val = preprocess(x_val_raw)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    max_pos_weight = float(config.get("max_pos_weight", 5.0))
    pos_weight = float(min(n_neg / max(n_pos, 1), max_pos_weight))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train).reshape(-1, 1)),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val).reshape(-1, 1)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, pos_weight


def load_encoder_weights_path(config: dict) -> Path:
    return resolve_path(config["model_dir"]) / "encoder.pth"

