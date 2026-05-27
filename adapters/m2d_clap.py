"""M2D-CLAP adapter.

Wraps `PortableM2D` from `models/m2d/examples/portable_m2d.py`. Audio is
loaded with librosa at the M2D sample rate (16kHz) and the model handles
log-mel spec internally. Text uses the encoder dictated by the checkpoint
naming convention (here: BERT base, dim 768).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

from eval.adapters.base import CLAPAdapter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_M2D_EXAMPLES = _REPO_ROOT / "models/m2d/examples"


def _ensure_m2d_on_path() -> None:
    p = str(_M2D_EXAMPLES)
    if p not in sys.path:
        sys.path.insert(0, p)


class M2dClapAdapter(CLAPAdapter):
    """Adapter for M2D-CLAP (ViT-B + BERT-base, 768-dim)."""

    name = "m2d"
    dim = 768

    def __init__(
        self,
        ckpt_path: str,
        device: str | None = None,
        normalize: bool = True,
        text_batch_size: int = 32,
        audio_batch_size: int = 8,
    ) -> None:
        super().__init__(normalize=normalize)
        self.ckpt_path = ckpt_path
        self.text_batch_size = text_batch_size
        self.audio_batch_size = audio_batch_size

        _ensure_m2d_on_path()
        import torch  # noqa: WPS433
        import librosa  # noqa: WPS433
        from portable_m2d import PortableM2D  # type: ignore  # noqa: WPS433

        self._torch = torch
        self._librosa = librosa

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # flat_features=True keeps per-patch dim at 768 (matches sem_token).
        self.model = PortableM2D(weight_file=ckpt_path, flat_features=True).to(device).eval()
        # Ensure text encoder is materialised once (also placed on device).
        self.model.get_clap_text_encoder()

        self.sample_rate = 16000
        self.max_samples = self.sample_rate * 10  # 10-second window

    # ----- internals -----

    def _load_audio(self, path: str):
        wav, _ = self._librosa.load(path, sr=self.sample_rate, mono=True)
        wav = self._torch.from_numpy(wav).float()
        if wav.shape[-1] > self.max_samples:
            wav = wav[..., : self.max_samples]
        elif wav.shape[-1] < self.max_samples:
            pad = self.max_samples - wav.shape[-1]
            wav = self._torch.nn.functional.pad(wav, (0, pad))
        return wav

    def _embed_audio_raw(self, paths: Iterable[str]) -> np.ndarray:
        torch = self._torch
        paths = list(paths)
        out: List[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(paths), self.audio_batch_size):
                chunk = paths[i : i + self.audio_batch_size]
                wavs = torch.stack([self._load_audio(p) for p in chunk]).to(self.device)
                emb = self.model.encode_clap_audio(wavs)
                out.append(emb.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0)

    def _embed_text_raw(self, texts: Iterable[str]) -> np.ndarray:
        torch = self._torch
        texts = list(texts)
        out: List[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.text_batch_size):
                chunk = texts[i : i + self.text_batch_size]
                emb = self.model.encode_clap_text(chunk, truncate=True)
                out.append(emb.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0)
