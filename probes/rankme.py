"""RankMe effective rank.

Garrido et al., "RankMe: Assessing the downstream performance of pretrained
self-supervised representations by their rank" (ICML 2023).

Effective rank is exp(H(p)) where p is the normalized singular-value
distribution. For an N x D embedding matrix (rows = samples), this is the
entropy-based effective rank of the cross-product / equivalently of the SVD.
"""
from __future__ import annotations

import numpy as np


def effective_rank(x: np.ndarray, eps: float = 1e-12) -> float:
    """Return RankMe effective rank of a 2D embedding matrix ``x``.

    Parameters
    ----------
    x : ndarray, shape (N, D)
    eps : float
        Numerical floor for zero singular values.
    """
    if x.ndim != 2:
        raise ValueError(f"expected 2D, got shape {x.shape}")
    # Use SVD on the smaller side for speed.
    s = np.linalg.svd(x.astype(np.float64, copy=False), compute_uv=False)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * np.log(p + eps)).sum()
    return float(np.exp(entropy))
