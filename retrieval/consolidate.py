"""Aggregate Phase-1 JSON sidecars into a single summary CSV.

Usage:
    PYTHONPATH=. python -m eval.retrieval.consolidate \
        --results-dir results/retrieval \
        --out results/retrieval/phase1_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


COLUMNS = [
    "model",
    "bench",
    "n_rows",
    "F",
    "G",
    "H",
    "runtime_s",
    "ckpt_sha256",
    "git_sha",
]


def collect(results_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for jpath in sorted(results_dir.glob("*.json")):
        if jpath.name.endswith("_summary.json"):
            continue
        with jpath.open() as f:
            d = json.load(f)
        rows.append({k: d.get(k, "") for k in COLUMNS})
    return rows


def write_csv(rows: List[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/retrieval")
    p.add_argument("--out", default="results/retrieval/phase1_summary.csv")
    ns = p.parse_args(argv)

    rows = collect(Path(ns.results_dir))
    write_csv(rows, Path(ns.out))
    print(f"wrote {len(rows)} rows → {ns.out}")
    for r in rows:
        print(
            f"  {r['model']:>6} {r['bench']:>10} "
            f"F={float(r['F']):.4f} G={float(r['G']):.4f} H={float(r['H']):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
