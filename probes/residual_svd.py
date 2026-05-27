"""Residual SVD probe.

Project the embedding matrix onto its top-k right singular vectors and report
the residual energy fraction (1 - sum of top-k squared singular values / total
squared singular values). Low residual after small k indicates the embeddings
live in a low-dim subspace ("anisotropy" / dimensional collapse).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def residual_energy(x: np.ndarray, k: int) -> Dict[str, float]:
    if x.ndim != 2:
        raise ValueError(f"expected 2D, got shape {x.shape}")
    if k < 1:
        raise ValueError("k must be >= 1")
    s = np.linalg.svd(x.astype(np.float64, copy=False), compute_uv=False)
    total = float((s ** 2).sum())
    if total == 0.0:
        return {"k": k, "explained_fraction": 0.0, "residual_fraction": 0.0}
    k_eff = min(k, s.size)
    explained = float((s[:k_eff] ** 2).sum())
    return {
        "k": k,
        "explained_fraction": explained / total,
        "residual_fraction": 1.0 - explained / total,
    }
