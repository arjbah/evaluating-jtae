"""Phase-3 correlation analysis.

Merges the Phase-1 retrieval summary (F/G/H per model x bench) with the
Phase-2 probe summary, then computes Spearman and Pearson correlations
between each probe and each retrieval score across the 6 (model x bench)
cells.

Writes ``results/analysis/phase3_merged.csv`` and
``results/analysis/phase3_correlations.csv``.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE_COLS = [
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
SCORE_COLS = ["F", "G", "H"]


def merge(retrieval_csv: Path, probes_csv: Path) -> pd.DataFrame:
    r = pd.read_csv(retrieval_csv)
    p = pd.read_csv(probes_csv)
    df = r.merge(p, on=["model", "bench"], how="inner")
    return df


def correlations(df: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    for probe, score in product(PROBE_COLS, SCORE_COLS):
        x = df[probe].to_numpy(dtype=float)
        y = df[score].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            rows.append({
                "probe": probe, "score": score,
                "spearman_r": float("nan"), "spearman_p": float("nan"),
                "pearson_r": float("nan"), "pearson_p": float("nan"),
                "n": int(len(x)),
            })
            continue
        sp = stats.spearmanr(x, y)
        pe = stats.pearsonr(x, y)
        rows.append({
            "probe": probe, "score": score,
            "spearman_r": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "pearson_r": float(pe.statistic), "pearson_p": float(pe.pvalue),
            "n": int(len(x)),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default=str(REPO_ROOT / "results/retrieval/phase1_summary.csv"))
    ap.add_argument("--probes", default=str(REPO_ROOT / "results/probes/phase2_summary.csv"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results/analysis"))
    ns = ap.parse_args(argv)

    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merge(Path(ns.retrieval), Path(ns.probes))
    df.to_csv(out_dir / "phase3_merged.csv", index=False)

    corr_rows = correlations(df)
    with (out_dir / "phase3_correlations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        w.writeheader()
        w.writerows(corr_rows)

    print(f"merged shape: {df.shape}")
    print("\nTop |Spearman| correlations with H:")
    h_rows = sorted(
        [r for r in corr_rows if r["score"] == "H" and not np.isnan(r["spearman_r"])],
        key=lambda r: -abs(r["spearman_r"]),
    )
    for r in h_rows[:6]:
        print(f"  {r['probe']:>22}  rho={r['spearman_r']:+.3f}  p={r['spearman_p']:.3f}")
    print(f"\nwrote {out_dir/'phase3_merged.csv'}")
    print(f"wrote {out_dir/'phase3_correlations.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
