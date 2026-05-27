"""Modality gap between two embedding clouds.

Liang et al., "Mind the Gap: Understanding the Modality Gap in Multi-modal
Contrastive Representation Learning" (NeurIPS 2022).

Given (already L2-normalized) audio and text embeddings, report:
    - l2_gap:  || mean(audio) - mean(text) ||_2
    - cosine:  cos(mean(audio), mean(text))
    - angle_deg: arccos(cosine) in degrees
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def gap(audio_emb: np.ndarray, text_emb: np.ndarray) -> Dict[str, float]:
    if audio_emb.ndim != 2 or text_emb.ndim != 2:
        raise ValueError("expected 2D embedding matrices")
    if audio_emb.shape[1] != text_emb.shape[1]:
        raise ValueError(
            f"dim mismatch: audio {audio_emb.shape[1]} vs text {text_emb.shape[1]}"
        )
    a = audio_emb.mean(axis=0).astype(np.float64)
    t = text_emb.mean(axis=0).astype(np.float64)
    l2 = float(np.linalg.norm(a - t))
    na = float(np.linalg.norm(a))
    nt = float(np.linalg.norm(t))
    if na == 0.0 or nt == 0.0:
        cos = 0.0
    else:
        cos = float(np.clip((a @ t) / (na * nt), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos))) if -1.0 <= cos <= 1.0 else float("nan")
    return {"l2_gap": l2, "cosine": cos, "angle_deg": angle}
