"""
Create the FLTrust root dataset indices.

Samples a stratified 5% slice from each client's training partition,
saving only the row indices (not the data) to configs/fltrust_root_indices.json.
The server loads this file at startup to hold its reference dataset.

Run once before any baseline comparison experiments:
    python scripts/create_fltrust_root.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfl.common.config import load_yaml, client_config_paths, fltrust_root_indices_path

ROOT_FRACTION = 0.05   # 5% of each client's training set
SEED = 42
TARGET_COL = "mortality"

# Active federation profile — see SFL_CLIENT_CONFIG_DIR in sfl.common.config.
CLIENT_CONFIGS = [str(p) for p in client_config_paths()]

output_path = fltrust_root_indices_path()
root_indices: dict = {}
total_root_samples = 0

for cfg_path in CLIENT_CONFIGS:
    cfg = load_yaml(cfg_path)
    client_id = int(cfg["client_id"])
    data_path = ROOT / cfg["data_path"]

    df = pd.read_csv(data_path)
    labels = df[TARGET_COL].values

    # 80/20 train/val split (matching dataset.py)
    train_idx, _ = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=labels, random_state=SEED
    )

    # Stratified 5% sample from training indices
    train_labels = labels[train_idx]
    root_idx, _ = train_test_split(
        train_idx,
        test_size=1.0 - ROOT_FRACTION,
        stratify=train_labels,
        random_state=SEED,
    )

    mortality_rate = train_labels[
        np.isin(train_idx, root_idx)
    ].mean() if len(root_idx) else 0.0

    print(
        f"Client {client_id}: {len(train_idx):,} train samples -> "
        f"{len(root_idx)} root samples ({ROOT_FRACTION*100:.0f}%), "
        f"mortality={mortality_rate:.3f}"
    )

    root_indices[str(client_id)] = root_idx.tolist()
    total_root_samples += len(root_idx)

meta = {
    "description": (
        "FLTrust root dataset indices. 5% stratified sample from each client's "
        "training partition (80/20 split, seed=42). Indices index into the "
        "original CSV files in data/processed/. NOT the data itself — no "
        "patient records are stored here."
    ),
    "root_fraction": ROOT_FRACTION,
    "seed": SEED,
    "total_root_samples": total_root_samples,
    "clients": root_indices,
}

output_path.write_text(json.dumps(meta, indent=2))
print(f"\nSaved to {output_path}")
print(f"Total root samples: {total_root_samples:,}")
