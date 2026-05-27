"""Two-NN intrinsic-dimension estimator.

Facco et al., "Estimating the intrinsic dimension of datasets by a minimal
neighborhood information" (Scientific Reports 2017).

For each point, let r1 < r2 be distances to its 1st and 2nd nearest
neighbours and mu = r2 / r1. The CDF of mu is F(mu) = 1 - mu^(-d), so a
linear fit of -log(1 - F(mu)) vs log(mu) through the origin gives slope d.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def intrinsic_dim(x: np.ndarray, discard_fraction: float = 0.1) -> float:
    """Estimate intrinsic dimension of point cloud ``x`` (shape (N, D)).

    ``discard_fraction`` drops the largest mu values (heavy tail) before the
    linear fit, as recommended by Facco et al.
    """
    if x.ndim != 2:
        raise ValueError(f"expected 2D, got shape {x.shape}")
    n = x.shape[0]
    if n < 10:
        raise ValueError("Two-NN requires at least 10 samples")

    nn = NearestNeighbors(n_neighbors=3).fit(x)
    dists, _ = nn.kneighbors(x)  # (N, 3): self, 1st, 2nd
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    # Drop degenerate (r1 == 0) points.
    mask = r1 > 0
    mu = (r2[mask] / r1[mask])
    mu = mu[mu > 1.0]
    mu = np.sort(mu)
    m = mu.size
    # Empirical CDF.
    f_emp = np.arange(1, m + 1) / (m + 1)
    # Discard the heavy tail.
    keep = int(m * (1.0 - discard_fraction))
    mu = mu[:keep]
    f_emp = f_emp[:keep]
    # Linear fit through origin: -log(1 - F) = d * log(mu).
    x_fit = np.log(mu)
    y_fit = -np.log(1.0 - f_emp)
    d_hat = float((x_fit @ y_fit) / (x_fit @ x_fit))
    return d_hat
