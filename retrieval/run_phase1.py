"""Phase-1 retrieval driver: F/G/H on CompA-Order and CompA-Attribute.

The core entry point is :func:`evaluate_pairwise` which is pure and unit-tested
with a fake adapter. The :func:`run` CLI wraps it with config loading,
embedding caching, and result serialization.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from eval.adapters.base import CLAPAdapter
from eval.data.manifest import BenchmarkRow, load_benchmark
from eval.retrieval.fgh_scores import PairwiseScore, fgh_scores

REPO_ROOT = Path(__file__).resolve().parents[2]


def evaluate_pairwise(
    adapter: CLAPAdapter,
    rows: list[BenchmarkRow],
    audio_root: str | Path,
) -> tuple[PairwiseScore, dict[str, np.ndarray]]:
    """Encode pair + reversed text/audio for each row and compute F/G/H.

    Returns ``(score, {"text_pair", "text_rev", "audio_pair", "audio_rev"})``
    so callers can cache the embeddings to disk.
    """
    audio_root = Path(audio_root)
    texts_pair = [r.pair_caption for r in rows]
    texts_rev = [r.reversed_pair_caption for r in rows]
    audio_pair = [str(audio_root / r.pair_file) for r in rows]
    audio_rev = [str(audio_root / r.reversed_pair_file) for r in rows]

    text_pair_emb = adapter.embed_text(texts_pair)
    text_rev_emb = adapter.embed_text(texts_rev)
    audio_pair_emb = adapter.embed_audio(audio_pair)
    audio_rev_emb = adapter.embed_audio(audio_rev)

    score = fgh_scores(text_pair_emb, audio_pair_emb, text_rev_emb, audio_rev_emb)
    return score, {
        "text_pair": text_pair_emb,
        "text_rev": text_rev_emb,
        "audio_pair": audio_pair_emb,
        "audio_rev": audio_rev_emb,
    }


# --------------------------------------------------------------------------- CLI helpers


def _import_adapter(dotted: str) -> type[CLAPAdapter]:
    module_path, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(
    model_key: str,
    bench_key: str,
    models_cfg: Path,
    bench_cfg: Path,
    results_dir: Path,
) -> dict:
    models = _load_yaml(models_cfg)
    benches = _load_yaml(bench_cfg)
    if model_key not in models:
        raise KeyError(f"model '{model_key}' not in {models_cfg}; have {list(models)}")
    if bench_key not in benches:
        raise KeyError(f"bench '{bench_key}' not in {bench_cfg}; have {list(benches)}")

    mcfg = models[model_key]
    bcfg = benches[bench_key]

    ckpt = REPO_ROOT / mcfg["ckpt"]
    audio_root = REPO_ROOT / bcfg["audio_root"]
    csv_path = REPO_ROOT / bcfg["csv"]

    AdapterCls = _import_adapter(mcfg["adapter"])
    adapter = AdapterCls(ckpt_path=str(ckpt))

    rows = load_benchmark(csv_path)
    t0 = time.time()
    score, emb = evaluate_pairwise(adapter, rows, audio_root)
    runtime_s = time.time() - t0

    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_key}_{bench_key}"
    npz_path = results_dir / f"{tag}_embeddings.npz"
    np.savez_compressed(npz_path, **emb)

    sidecar = {
        "model": model_key,
        "model_name": mcfg.get("name"),
        "ckpt": mcfg["ckpt"],
        "ckpt_sha256": mcfg.get("sha256"),
        "bench": bench_key,
        "csv": bcfg["csv"],
        "audio_root": bcfg["audio_root"],
        "n_rows": score.n_pairs,
        "F": score.F,
        "G": score.G,
        "H": score.H,
        "runtime_s": runtime_s,
        "git_sha": _git_sha(),
        "embeddings_path": str(npz_path.relative_to(REPO_ROOT)),
    }
    json_path = results_dir / f"{tag}.json"
    with json_path.open("w") as f:
        json.dump(sidecar, f, indent=2)

    csv_out = results_dir / f"{tag}.csv"
    with csv_out.open("w") as f:
        f.write("model,bench,n_pairs,F,G,H,runtime_s\n")
        f.write(
            f"{model_key},{bench_key},{score.n_pairs},{score.F:.6f},{score.G:.6f},{score.H:.6f},{runtime_s:.2f}\n"
        )

    return sidecar


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase-1 CompA retrieval driver (F/G/H).")
    p.add_argument("--model", required=True, help="Key into configs/models.yaml")
    p.add_argument("--bench", required=True, help="Key into configs/benchmarks.yaml")
    p.add_argument("--models-cfg", default=str(REPO_ROOT / "configs/models.yaml"))
    p.add_argument("--bench-cfg", default=str(REPO_ROOT / "configs/benchmarks.yaml"))
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results/retrieval"))
    args = p.parse_args(list(argv) if argv is not None else None)

    sidecar = run(
        model_key=args.model,
        bench_key=args.bench,
        models_cfg=Path(args.models_cfg),
        bench_cfg=Path(args.bench_cfg),
        results_dir=Path(args.results_dir),
    )
    print(json.dumps({k: sidecar[k] for k in ("model", "bench", "n_rows", "F", "G", "H", "runtime_s")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
