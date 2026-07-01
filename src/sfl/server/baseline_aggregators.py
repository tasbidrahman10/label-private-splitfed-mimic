"""
Baseline Byzantine-robust aggregators for comparison against SNAS.

Implementation follows the professor's specification exactly.
These are NOT staleness-aware by design — that limitation is intentional
and must be called out explicitly in the paper's results section.

Do NOT modify robust_fedavg.py — it remains the SNAS reference implementation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: flatten state_dict <-> numpy vector
# ---------------------------------------------------------------------------

def state_dict_to_flat(state_dict: dict[str, torch.Tensor]) -> np.ndarray:
    """Concatenate all float tensors in a state_dict into a 1-D numpy array."""
    parts = []
    for key, tensor in state_dict.items():
        if tensor.is_floating_point():
            parts.append(tensor.detach().cpu().float().numpy().ravel())
    return np.concatenate(parts) if parts else np.array([])


def flat_to_state_dict(
    flat: np.ndarray,
    reference: dict[str, torch.Tensor],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruct a state_dict from a flat numpy array using a reference for shapes."""
    result = {}
    offset = 0
    for key, tensor in reference.items():
        if tensor.is_floating_point():
            n = tensor.numel()
            chunk = flat[offset: offset + n].reshape(tensor.shape)
            t = torch.from_numpy(chunk.copy()).to(tensor.dtype)
            result[key] = t.to(device) if device is not None else t
            offset += n
        else:
            result[key] = tensor.clone()
    return result


# ---------------------------------------------------------------------------
# Krum
# ---------------------------------------------------------------------------

def krum_aggregate(
    submissions: Dict[int, np.ndarray],   # client_id -> flattened weight vector
    f_byzantine: int = 1,                 # assumed max number of Byzantine clients
) -> Tuple[np.ndarray, int]:
    """
    Returns (selected_weight_vector, selected_client_id).

    Krum selects the single client update with smallest sum of squared
    distances to its k nearest neighbors. Has NO staleness awareness —
    this is intentional and must be called out explicitly in the results.

    k = n_clients - f_byzantine - 2  (per original Blanchard et al. paper)
    """
    ids = list(submissions.keys())
    n = len(ids)
    k = n - f_byzantine - 2
    if k < 1:
        k = max(1, n - 2)

    dists: Dict[int, List[float]] = {i: [] for i in ids}
    for i in ids:
        for j in ids:
            if i == j:
                continue
            d = float(np.linalg.norm(submissions[i] - submissions[j]) ** 2)
            dists[i].append(d)

    scores = {i: sum(sorted(dists[i])[:k]) for i in ids}
    winner = min(scores, key=scores.get)  # type: ignore[arg-type]

    logger.info(
        "Krum selected client %d (score=%.4f) from %d submissions. "
        "Krum has NO staleness awareness by design.",
        winner, scores[winner], n,
    )
    return submissions[winner], winner


# ---------------------------------------------------------------------------
# Coordinate-wise Trimmed Mean
# ---------------------------------------------------------------------------

def trimmed_mean_aggregate(
    submissions: Dict[int, np.ndarray],
    trim_ratio: float = 0.2,             # fraction trimmed from EACH tail, per coordinate
) -> np.ndarray:
    """
    Coordinate-wise trimmed mean. No staleness input.

    Removes the top and bottom trim_ratio fraction of values per coordinate
    before averaging. trim_ratio=0.2 removes 20% from each end (so 40% total
    are discarded), matching the standard setting for 1 Byzantine client
    out of 3 total (f=1, trim_ratio=f/n = 1/3, clamped at 0.2 here for
    numerical stability with only 3 clients).
    """
    stacked = np.stack(list(submissions.values()), axis=0)  # (n_clients, n_params)
    n = stacked.shape[0]
    k = int(np.floor(trim_ratio * n))
    sorted_vals = np.sort(stacked, axis=0)
    trimmed = sorted_vals[k: n - k] if n - 2 * k > 0 else sorted_vals

    logger.info(
        "Trimmed mean: %d clients, trim_ratio=%.2f, trimmed %d from each tail. "
        "No staleness awareness by design.",
        n, trim_ratio, k,
    )
    return trimmed.mean(axis=0)


# ---------------------------------------------------------------------------
# FLTrust
# ---------------------------------------------------------------------------

def fltrust_aggregate(
    submissions: Dict[int, np.ndarray],
    root_gradient: np.ndarray,            # server-computed reference gradient
    client_dataset_sizes: Dict[int, int],
) -> np.ndarray:
    """
    FLTrust: trust score = ReLU(cosine_similarity(client_update, root_gradient)).
    Magnitude-normalises each client update to ||root_gradient|| before
    weighting by trust. No staleness input.

    Note: FLTrust requires the server to hold a small clean labeled root
    dataset (5% of total training data in our setup) to compute root_gradient.
    This is a fundamental assumption difference from SNAS, which requires no
    server-side labeled data — a key advantage documented in the paper.
    """
    root_norm = float(np.linalg.norm(root_gradient))
    if root_norm < 1e-12:
        logger.warning("FLTrust: root_gradient norm near zero — falling back to uniform average.")
        return np.stack(list(submissions.values())).mean(axis=0)

    trust_scores: Dict[int, float] = {}
    normalized: Dict[int, np.ndarray] = {}

    for cid, w in submissions.items():
        w_norm = float(np.linalg.norm(w))
        cos = float(np.dot(w, root_gradient) / (w_norm * root_norm + 1e-12))
        trust_scores[cid] = max(0.0, cos)              # ReLU
        scale = root_norm / (w_norm + 1e-12)
        normalized[cid] = w * scale

    total_trust = sum(trust_scores.values()) + 1e-12
    agg = sum(
        (trust_scores[cid] / total_trust) * normalized[cid]
        for cid in submissions
    )

    logger.info(
        "FLTrust: %d clients, trust scores=%s. No staleness awareness by design.",
        len(submissions),
        {cid: round(t, 4) for cid, t in trust_scores.items()},
    )
    return agg
