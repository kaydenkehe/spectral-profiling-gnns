import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Example: runs/paper_20260501-012345")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_dir / "summary.csv")
    df["test_acc"] = df["test_acc"].astype(float)

    group_cols = ["dataset", "model", "K", "hidden"]
    agg = (
        df.groupby(group_cols)["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )
    agg.to_csv(out / "aggregate_by_config.csv", index=False)

    winners = (
        agg.sort_values(["dataset", "mean"], ascending=[True, False])
        .groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )
    winners.to_csv(out / "winners_by_dataset.csv", index=False)

    model_wins = (
        winners.groupby("model")
        .size()
        .reset_index(name="n_dataset_wins")
        .sort_values("n_dataset_wins", ascending=False)
    )
    model_wins.to_csv(out / "model_win_counts.csv", index=False)

    # Best K per dataset/model.
    best_per_model = (
        agg.sort_values(["dataset", "model", "mean"], ascending=[True, True, False])
        .groupby(["dataset", "model"])
        .head(1)
        .reset_index(drop=True)
    )
    best_per_model.to_csv(out / "best_config_per_dataset_model.csv", index=False)

    print("\nWinners by dataset")
    print(winners[["dataset", "model", "K", "hidden", "mean", "std", "count"]].to_string(index=False))

    print("\nModel win counts")
    print(model_wins.to_string(index=False))

    print(f"\nWrote analysis tables to {out}")


if __name__ == "__main__":
    main()
