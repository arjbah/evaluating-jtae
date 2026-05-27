"""Phase-3 figures: bar chart of H and scatter plots of probes vs H.

Outputs PNG and PDF into ``figures/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS = ["laion", "mga", "m2d"]
BENCHES = ["order", "attribute"]
PROBE_KEYS = [
    ("rankme_text", "RankMe (text)"),
    ("rankme_audio", "RankMe (audio)"),
    ("two_nn_audio", "Two-NN ID (audio)"),
    ("modality_angle_deg", "Modality angle (deg)"),
]


def fig_h_bars(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    width = 0.35
    x = np.arange(len(MODELS))
    for i, bench in enumerate(BENCHES):
        ys = [
            float(df[(df.model == m) & (df.bench == bench)]["H"].iloc[0])
            for m in MODELS
        ]
        ax.bar(x + (i - 0.5) * width, ys, width, label=bench)
    ax.axhline(0.025, color="grey", linestyle="--", linewidth=0.8, label="random (1/40)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in MODELS])
    ax.set_ylabel("Hard compositional score H")
    ax.set_ylim(0, max(0.18, df["H"].max() * 1.15))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("CompA hard score H per model")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def fig_probe_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(PROBE_KEYS), figsize=(3.0 * len(PROBE_KEYS), 3.0), sharey=True)
    h = df["H"].to_numpy(dtype=float)
    for ax, (key, label) in zip(axes, PROBE_KEYS):
        x = df[key].to_numpy(dtype=float)
        for m, marker in zip(MODELS, ("o", "s", "^")):
            mask = df["model"] == m
            ax.scatter(x[mask], h[mask], label=m.upper(), marker=marker, s=55)
        # Linear fit line (just for visual).
        if np.std(x) > 0:
            coeffs = np.polyfit(x, h, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("H (CompA)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(REPO_ROOT / "results/analysis/phase3_merged.csv"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    ns = ap.parse_args(argv)

    df = pd.read_csv(ns.merged)
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_h_bars(df, out_dir / "h_per_model")
    fig_probe_scatter(df, out_dir / "probes_vs_h")
    print(f"wrote {out_dir/'h_per_model.png'} and probes_vs_h.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
