"""Inspect a run directory produced by experiments-sac/run_all.py."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


NumericDiff = Tuple[float, float]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _pairwise_numeric_diff(a: pd.DataFrame, b: pd.DataFrame) -> NumericDiff:
    common = [c for c in _numeric_columns(a) if c in b.columns]
    if not common:
        return float("nan"), float("nan")
    diffs = (a[common] - b[common]).abs().stack()
    if diffs.empty:
        return 0.0, 0.0
    return float(diffs.max()), float(diffs.mean())


def inspect_run(run_dir: Path) -> Dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")

    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found at '{summary_path}'.")

    summary = pd.read_csv(summary_path)
    rounds: Dict[str, pd.DataFrame] = {}
    for path in sorted(run_dir.glob("*_rounds.csv")):
        mode = path.stem.replace("_rounds", "")
        df_round = pd.read_csv(path)
        if "mode" not in df_round.columns:
            df_round["mode"] = mode
        rounds[mode] = df_round

    return {"summary": summary, **rounds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Path to a run directory (e.g., runs/foo/bar)")
    args = parser.parse_args()

    tables = inspect_run(args.run_dir)
    summary = tables.pop("summary")
    print("Summary metrics (rounded to 3 decimals):")
    print(summary.round(3).to_string(index=False))

    modes = list(tables.keys())
    if len(modes) < 2:
        print("Only one mode recorded; skipping pairwise comparison.")
        return

    print("\nPairwise round-metric differences:")
    for m1, m2 in itertools.combinations(modes, 2):
        df1, df2 = tables[m1], tables[m2]
        max_diff, mean_diff = _pairwise_numeric_diff(df1, df2)
        if pd.isna(max_diff):
            status = "no numeric overlap"
        elif max_diff == 0:
            status = "identical (numeric)"
        else:
            status = "different"
        print(
            f"  {m1} vs {m2}: max |Δ| = {max_diff:.4g}, mean |Δ| = {mean_diff:.4g} -> {status}"
        )


if __name__ == "__main__":
    main()
