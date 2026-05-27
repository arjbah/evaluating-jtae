"""CompA F/G/H pairwise scores + standard retrieval metrics.

The F/G/H definitions follow ``data/CompA/evaluation/benchmark_eval.py`` and
the CompA paper (ICLR 2024). Inputs are pre-computed embeddings; this module
is pure NumPy and stays GPU-free so it can be unit-tested cheaply.

Convention
----------
Cosine similarity is the dot product of L2-normalized vectors:

    sim(t, a) = (t / ||t||) . (a / ||a||)

Let ``t0``, ``a0`` be the pair text and audio for row *i*, and ``t1``, ``a1``
be the reversed/perturbed counterparts. For each row we compute the 2x2 grid

    s[i, j] = sim(t_i, a_j)        # i, j in {0, 1}

and define

    F-row = 1  iff  s[0,0] > s[1,0]  AND  s[1,1] > s[0,1]
    G-row = 1  iff  s[0,0] > s[0,1]  AND  s[1,1] > s[1,0]
    H-row = F-row AND G-row

F, G, H are means over rows. Higher = better compositional discrimination.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairwiseScore:
    F: float
    G: float
    H: float
    n_pairs: int


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two ``[N, D]`` arrays."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a_n * b_n, axis=1)


def fgh_scores(
    text_pair: np.ndarray,
    audio_pair: np.ndarray,
    text_rev: np.ndarray,
    audio_rev: np.ndarray,
) -> PairwiseScore:
    """Compute CompA F/G/H scores from pre-computed embeddings.

    All inputs are ``[N, D]`` numpy arrays.
    """
    n = text_pair.shape[0]
    if not (audio_pair.shape[0] == text_rev.shape[0] == audio_rev.shape[0] == n):
        raise ValueError("all four embedding arrays must have the same number of rows")
    if n == 0:
        return PairwiseScore(F=0.0, G=0.0, H=0.0, n_pairs=0)

    s00 = _row_cosine(text_pair, audio_pair)
    s01 = _row_cosine(text_pair, audio_rev)
    s10 = _row_cosine(text_rev, audio_pair)
    s11 = _row_cosine(text_rev, audio_rev)

    f = ((s00 > s10) & (s11 > s01)).astype(np.float64)
    g = ((s00 > s01) & (s11 > s10)).astype(np.float64)
    h = f * g

    return PairwiseScore(
        F=float(f.mean()),
        G=float(g.mean()),
        H=float(h.mean()),
        n_pairs=n,
    )


def recall_at_k(
    queries: np.ndarray,
    candidates: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Recall@k for query->candidate retrieval, treating each query as having
    exactly one correct candidate (index ``ground_truth[i]`` in ``candidates``).
    """
    sims = queries @ candidates.T               # [N_q, N_c]
    # top-k indices per query (unordered top-k is sufficient for recall)
    if k >= candidates.shape[0]:
        return 1.0
    topk_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    hits = np.array([
        gt in row for gt, row in zip(ground_truth, topk_idx)
    ])
    return float(hits.mean())


def mrr(
    queries: np.ndarray,
    candidates: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Mean reciprocal rank of the ground-truth candidate per query."""
    sims = queries @ candidates.T
    # rank: position (1-indexed) of ground-truth in descending-sim ordering
    order = np.argsort(-sims, axis=1)
    ranks = np.empty(queries.shape[0], dtype=np.int64)
    for i, gt in enumerate(ground_truth):
        ranks[i] = int(np.where(order[i] == gt)[0][0]) + 1
    return float(np.mean(1.0 / ranks))
