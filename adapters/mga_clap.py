"""MGA-CLAP adapter (Yiming Li et al., ICT-CAS).

Wraps ``models/MGA-CLAP/models/ase_model.ASE`` behind the common
:class:`eval.adapters.base.CLAPAdapter` interface. The MGA repo uses
local-relative imports (``from models.audio_encoder import ...``,
``from tools.utils import *``), so we prepend the repo path to
``sys.path`` and stub out optional training-only dependencies
(``sed_scores_eval``) that are not pip-installable on paths with
spaces.

Embedding dim = 1024 (sparse-codebook aggregated MSC output).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np

from eval.adapters.base import CLAPAdapter

MGA_REPO = Path(__file__).resolve().parents[2] / "models/MGA-CLAP"
MGA_CONFIG = MGA_REPO / "settings/inference_example.yaml"


def _ensure_sed_scores_stub() -> None:
    """Stub the optional ``sed_scores_eval`` package (only used in training utils)."""
    if "sed_scores_eval" in sys.modules:
        return
    root = types.ModuleType("sed_scores_eval")
    utils = types.ModuleType("sed_scores_eval.utils")
    scores = types.ModuleType("sed_scores_eval.utils.scores")
    scores.create_score_dataframe = lambda *_a, **_k: None  # noqa: E731
    utils.scores = scores
    root.utils = utils
    sys.modules["sed_scores_eval"] = root
    sys.modules["sed_scores_eval.utils"] = utils
    sys.modules["sed_scores_eval.utils.scores"] = scores


class MgaClapAdapter(CLAPAdapter):
    """MGA-CLAP wrapper (HTSAT audio + bert-base text + 4096-entry codebook MSC, dim=1024)."""

    name = "mga"
    dim = 1024

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        normalize: bool = True,
        text_batch_size: int = 32,
        audio_batch_size: int = 8,
        sr: int = 32000,
        max_seconds: float = 10.0,
    ) -> None:
        super().__init__(normalize=normalize)

        _ensure_sed_scores_stub()
        repo = str(MGA_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # Heavy imports done lazily so the base env doesn't need them.
        import librosa
        import soundfile  # noqa: F401  (ensure libsndfile is available)
        import torch
        import torch.nn.functional as F
        import yaml
        from models.ase_model import ASE  # type: ignore  # noqa: PLC0415

        self._torch = torch
        self._F = F
        self._librosa = librosa

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.text_batch_size = text_batch_size
        self.audio_batch_size = audio_batch_size
        self.sr = sr
        self.max_samples = int(sr * max_seconds)

        with open(MGA_CONFIG) as f:
            config = yaml.safe_load(f)
        config["device"] = device

        self._model = ASE(config).to(device)
        cp = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(cp, dict) and "model" in cp:
            state = cp["model"]
        else:
            state = cp
        # Some MGA ckpts may carry "module." prefix from DDP. Strip if present.
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    def _load_audio(self, path: str):
        # librosa returns mono float32 at the requested sample rate.
        wav, _ = self._librosa.load(path, sr=self.sr, mono=True)
        wav_t = self._torch.from_numpy(wav)
        if wav_t.shape[0] >= self.max_samples:
            wav_t = wav_t[: self.max_samples]
        else:
            wav_t = self._F.pad(wav_t, (0, self.max_samples - wav_t.shape[0]))
        return wav_t

    def _embed_audio_raw(self, paths: list[str]) -> np.ndarray:
        torch = self._torch
        F = self._F
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(paths), self.audio_batch_size):
                batch_paths = paths[i : i + self.audio_batch_size]
                wavs = torch.stack([self._load_audio(p) for p in batch_paths]).to(self.device)
                _, frame_embeds = self._model.encode_audio(wavs)
                audio_embeds = self._model.msc(frame_embeds, self._model.codebook)
                audio_embeds = F.normalize(audio_embeds, dim=-1)
                chunks.append(audio_embeds.detach().cpu().float().numpy())
        return np.concatenate(chunks, axis=0)

    def _embed_text_raw(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        F = self._F
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.text_batch_size):
                batch = list(texts[i : i + self.text_batch_size])
                _, word_embeds, attn_mask = self._model.encode_text(batch)
                text_embeds = self._model.msc(word_embeds, self._model.codebook, attn_mask)
                text_embeds = F.normalize(text_embeds, dim=-1)
                chunks.append(text_embeds.detach().cpu().float().numpy())
        return np.concatenate(chunks, axis=0)
