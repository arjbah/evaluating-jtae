"""Base class for CLAP model adapters.

Each concrete adapter wraps one CLAP variant (LAION, MGA, M2D) behind a
uniform ``embed_audio`` / ``embed_text`` interface so downstream retrieval
metrics and geometric probes can stay model-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization. Zero rows are preserved as zero."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe = np.where(norms > eps, norms, 1.0)
    out = x / safe
    # zero rows stay zero
    out[norms.squeeze(-1) <= eps] = 0.0
    return out.astype(np.float32, copy=False)


class CLAPAdapter(ABC):
    """Abstract base for a CLAP audio-text embedder.

    Subclasses set class attrs ``name`` and ``dim`` and implement
    ``_embed_audio_raw`` / ``_embed_text_raw`` returning ``[N, D]`` float32
    arrays. The public ``embed_audio`` / ``embed_text`` apply L2 normalization
    when ``normalize=True`` (default).
    """

    name: str = "unset"
    dim: int = 0

    def __init__(self, normalize: bool = True) -> None:
        if type(self) is CLAPAdapter:  # safety net for direct instantiation
            raise TypeError("CLAPAdapter is abstract; use a concrete subclass.")
        self.normalize = normalize

    @abstractmethod
    def _embed_audio_raw(self, paths: list[str]) -> np.ndarray: ...

    @abstractmethod
    def _embed_text_raw(self, texts: list[str]) -> np.ndarray: ...

    def embed_audio(self, paths: list[str]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._embed_audio_raw(list(paths)).astype(np.float32, copy=False)
        self._check_shape(emb, len(paths))
        return l2_normalize(emb) if self.normalize else emb

    def embed_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._embed_text_raw(list(texts)).astype(np.float32, copy=False)
        self._check_shape(emb, len(texts))
        return l2_normalize(emb) if self.normalize else emb

    def _check_shape(self, emb: np.ndarray, n_expected: int) -> None:
        if emb.ndim != 2:
            raise ValueError(f"{self.name}: expected 2D embeddings, got shape {emb.shape}")
        if emb.shape[0] != n_expected:
            raise ValueError(
                f"{self.name}: expected {n_expected} rows, got {emb.shape[0]}"
            )
        if emb.shape[1] != self.dim:
            raise ValueError(
                f"{self.name}: expected dim={self.dim}, got {emb.shape[1]}"
            )
