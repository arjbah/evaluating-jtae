"""LAION-CLAP adapter (upstream ``laion-clap`` package).

Wraps the LAION-CLAP ``630k-audioset-fusion-best.pt`` checkpoint behind the
common :class:`eval.adapters.base.CLAPAdapter` interface so retrieval and
geometric probes can stay model-agnostic.

The checkpoint name encodes its training recipe: the ``-fusion-`` suffix
means feature-fusion is enabled on the HTSAT audio branch, so
``CLAP_Module`` must be constructed with ``enable_fusion=True``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from eval.adapters.base import CLAPAdapter


class LaionCompaAdapter(CLAPAdapter):
    """LAION-CLAP fusion variant (HTSAT-tiny audio + RoBERTa text, dim=512)."""

    name = "laion"
    dim = 512

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        normalize: bool = True,
        text_batch_size: int = 32,
        audio_batch_size: int = 16,
    ) -> None:
        super().__init__(normalize=normalize)

        # Import lazily so the rest of the test suite does not require the
        # heavy laion-clap install when running in the base env.
        import torch
        from laion_clap import CLAP_Module

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.text_batch_size = text_batch_size
        self.audio_batch_size = audio_batch_size

        self._model = CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny", device=device)
        self._model.load_ckpt(ckpt_path)

    def _embed_audio_raw(self, paths: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for i in range(0, len(paths), self.audio_batch_size):
            batch = paths[i : i + self.audio_batch_size]
            emb = self._model.get_audio_embedding_from_filelist(x=batch, use_tensor=False)
            chunks.append(np.asarray(emb, dtype=np.float32))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.dim), dtype=np.float32)

    def _embed_text_raw(self, texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.text_batch_size):
            batch = texts[i : i + self.text_batch_size]
            emb = self._model.get_text_embedding(batch, use_tensor=False)
            chunks.append(np.asarray(emb, dtype=np.float32))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.dim), dtype=np.float32)
