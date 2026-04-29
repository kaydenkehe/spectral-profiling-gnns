from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collate spectral run summaries.")
    parser.add_argument("run_dirs", nargs="*", help="Run directory names under spectral/runs.")
    parser.add_argument("--out", default="best_spectral.csv", help="Output CSV path under spectral/runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(__file__).resolve().parent
    run_dirs = args.run_dirs
    if not run_dirs:
        run_dirs = sorted(
            p.name for p in runs_root.iterdir()
            if p.is_dir() and (p / "summary.csv").exists()
        )
    if not run_dirs:
        raise SystemExit("No spectral run directories found.")

    frames = [
        pd.read_csv(runs_root / run_dir / "summary.csv", keep_default_na=False)
        for run_dir in run_dirs
    ]
    df = pd.concat(frames, ignore_index=True)
    df["test_acc"] = pd.to_numeric(df["test_acc"])

    group_cols = [
        "dataset",
        "model",
        "basis",
        "order",
        "hidden",
        "alpha",
        "jacobi_alpha",
        "jacobi_beta",
    ]
    agg = (
        df.groupby(group_cols)["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    best = agg.loc[agg.groupby(["dataset", "model"])["mean"].idxmax()]
    best = best.sort_values(["dataset", "model"]).reset_index(drop=True)
    best.to_csv(runs_root / args.out, index=False)
    print(f"Wrote {runs_root / args.out}")


if __name__ == "__main__":
    main()

