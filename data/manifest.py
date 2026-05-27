"""CompA benchmark manifest loading + validation.

Handles both CompA-Order and CompA-Attribute CSV shapes. The Attribute CSV
ships with duplicate columns (``pair_file.1``, ``reversed_pair_file.1``) which
are ignored by :func:`load_benchmark`.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = (
    "pair_caption",
    "reversed_pair_caption",
    "pair_file",
    "reversed_pair_file",
)


@dataclass(frozen=True)
class BenchmarkRow:
    pair_caption: str
    reversed_pair_caption: str
    pair_file: str
    reversed_pair_file: str
    triplet_caption: str | None = None
    triplet_file: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    n_rows: int
    n_present: int
    missing: list[str]

    @property
    def is_complete(self) -> bool:
        return not self.missing


def load_benchmark(csv_path: str | Path) -> list[BenchmarkRow]:
    """Parse a CompA-Order or CompA-Attribute CSV into typed rows.

    Raises ``ValueError`` if any of :data:`REQUIRED_COLUMNS` is absent.

    The CompA-Attribute CSV ships with duplicate columns ``pair_file.1`` /
    ``reversed_pair_file.1`` that hold the *local* audio filenames; the
    primary ``pair_file`` / ``reversed_pair_file`` columns hold YouTube IDs
    that are not part of the released archive. When the ``.1`` variants are
    present, they take precedence.
    """
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing_cols:
            raise ValueError(
                f"{csv_path}: missing required columns: {missing_cols}"
            )
        pair_col = "pair_file.1" if "pair_file.1" in fieldnames else "pair_file"
        rev_col = (
            "reversed_pair_file.1"
            if "reversed_pair_file.1" in fieldnames
            else "reversed_pair_file"
        )
        rows: list[BenchmarkRow] = []
        for raw in reader:
            triplet_caption = (raw.get("triplet_caption") or "").strip()
            triplet_file = (raw.get("triplet_file") or "").strip()
            # CompA-Order uses "-" as a sentinel for rows lacking a triplet.
            if triplet_caption in ("", "-"):
                triplet_caption_val: str | None = None
            else:
                triplet_caption_val = triplet_caption
            if triplet_file in ("", "-"):
                triplet_file_val: str | None = None
            else:
                triplet_file_val = triplet_file
            rows.append(
                BenchmarkRow(
                    pair_caption=raw["pair_caption"],
                    reversed_pair_caption=raw["reversed_pair_caption"],
                    pair_file=raw[pair_col],
                    reversed_pair_file=raw[rev_col],
                    triplet_caption=triplet_caption_val,
                    triplet_file=triplet_file_val,
                )
            )
    return rows


def _iter_referenced_files(rows: Iterable[BenchmarkRow]) -> Iterable[str]:
    for r in rows:
        yield r.pair_file
        yield r.reversed_pair_file
        if r.triplet_file:
            yield r.triplet_file


def validate(csv_path: str | Path, audio_root: str | Path) -> ValidationReport:
    """Check that every audio file referenced in ``csv_path`` exists under ``audio_root``."""
    audio_root = Path(audio_root)
    rows = load_benchmark(csv_path)
    referenced = sorted(set(_iter_referenced_files(rows)))
    missing = [name for name in referenced if not (audio_root / name).is_file()]
    n_present = len(referenced) - len(missing)
    return ValidationReport(n_rows=len(rows), n_present=n_present, missing=missing)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate a CompA benchmark manifest.")
    parser.add_argument("--csv", required=True, help="Path to the benchmark CSV.")
    parser.add_argument("--audio-root", required=True, help="Directory holding the referenced audio files.")
    args = parser.parse_args()
    report = validate(args.csv, args.audio_root)
    print(
        f"rows={report.n_rows}  present={report.n_present}  missing={len(report.missing)}"
    )
    if report.missing:
        for m in report.missing[:20]:
            print(f"  MISSING: {m}")
        if len(report.missing) > 20:
            print(f"  ... and {len(report.missing) - 20} more")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
