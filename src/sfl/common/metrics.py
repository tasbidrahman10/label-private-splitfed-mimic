from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    labels = y_true.reshape(-1).astype(int)
    preds = (probs >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "auprc": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

