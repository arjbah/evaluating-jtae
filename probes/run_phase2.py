"""Phase-2 driver: compute geometric probes on cached Phase-1 embeddings.

For each model × benchmark pair, load the ``{tag}_embeddings.npz`` produced
by :mod:`eval.retrieval.run_phase1`, then compute:

- RankMe effective rank on the concatenated audio embeddings and on text.
- Two-NN intrinsic dimension on audio and text clouds.
- Residual SVD energy at k in {1, 5, 10} on audio.
- Modality gap between audio and text means.

Writes ``results/probes/{model}_{bench}_probes.json`` and aggregates into
``results/probes/phase2_summary.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.probes import modality_gap, rankme, residual_svd, two_nn

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stack_audio(npz) -> np.ndarray:
    return np.concatenate([npz["audio_pair"], npz["audio_rev"]], axis=0)


def _stack_text(npz) -> np.ndarray:
    return np.concatenate([npz["text_pair"], npz["text_rev"]], axis=0)


def compute_probes(npz_path: Path) -> Dict:
    npz = np.load(npz_path)
    a = _stack_audio(npz)
    t = _stack_text(npz)

    out: Dict[str, object] = {
        "embeddings_path": str(npz_path),
        "n_audio": int(a.shape[0]),
        "n_text": int(t.shape[0]),
        "dim": int(a.shape[1]),
        "rankme_audio": rankme.effective_rank(a),
        "rankme_text": rankme.effective_rank(t),
        "two_nn_audio": two_nn.intrinsic_dim(a),
        "two_nn_text": two_nn.intrinsic_dim(t),
        "residual_svd_audio": {
            f"k={k}": residual_svd.residual_energy(a, k=k) for k in (1, 5, 10)
        },
        "residual_svd_text": {
            f"k={k}": residual_svd.residual_energy(t, k=k) for k in (1, 5, 10)
        },
        "modality_gap": modality_gap.gap(a, t),
    }
    return out


SUMMARY_COLS = [
    "model",
    "bench",
    "n_audio",
    "n_text",
    "dim",
    "rankme_audio",
    "rankme_text",
    "two_nn_audio",
    "two_nn_text",
    "residual_audio_k1",
    "residual_audio_k5",
    "residual_audio_k10",
    "modality_l2_gap",
    "modality_cosine",
    "modality_angle_deg",
]


def _flatten_for_summary(model: str, bench: str, p: Dict) -> Dict:
    return {
        "model": model,
        "bench": bench,
        "n_audio": p["n_audio"],
        "n_text": p["n_text"],
        "dim": p["dim"],
        "rankme_audio": p["rankme_audio"],
        "rankme_text": p["rankme_text"],
        "two_nn_audio": p["two_nn_audio"],
        "two_nn_text": p["two_nn_text"],
        "residual_audio_k1": p["residual_svd_audio"]["k=1"]["residual_fraction"],
        "residual_audio_k5": p["residual_svd_audio"]["k=5"]["residual_fraction"],
        "residual_audio_k10": p["residual_svd_audio"]["k=10"]["residual_fraction"],
        "modality_l2_gap": p["modality_gap"]["l2_gap"],
        "modality_cosine": p["modality_gap"]["cosine"],
        "modality_angle_deg": p["modality_gap"]["angle_deg"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-dir", default=str(REPO_ROOT / "results/retrieval"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results/probes"))
    ns = p.parse_args(argv)

    ret_dir = Path(ns.retrieval_dir)
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for npz_path in sorted(ret_dir.glob("*_embeddings.npz")):
        tag = npz_path.stem.removesuffix("_embeddings")
        model, _, bench = tag.partition("_")
        print(f"[probes] {model:>6} {bench:>10}", flush=True)
        probes = compute_probes(npz_path)
        probes["model"] = model
        probes["bench"] = bench
        with (out_dir / f"{tag}_probes.json").open("w") as f:
            json.dump(probes, f, indent=2)
        rows.append(_flatten_for_summary(model, bench, probes))

    summary_csv = out_dir / "phase2_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} probe rows → {summary_csv}")
    for r in rows:
        print(
            f"  {r['model']:>6} {r['bench']:>10} "
            f"rmA={r['rankme_audio']:.1f} rmT={r['rankme_text']:.1f} "
            f"idA={r['two_nn_audio']:.2f} idT={r['two_nn_text']:.2f} "
            f"gap={r['modality_l2_gap']:.3f} ang={r['modality_angle_deg']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
