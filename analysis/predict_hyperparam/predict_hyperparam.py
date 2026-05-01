"""Compare SLP and homophily as hyperparameter selectors.

For each held-out dataset, selectors choose a full config or one hyperparameter
from the other datasets, then we measure regret against the held-out oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
RUNS = [
    "gcn_20260426-071840",
    "fagcn_20260426-203611",
    "mlp_20260426-230354",
    "hfgcn_20260429-135103",
]

DISPLAY_TO_DATASET = {
    "Amazon Photo": "Photo",
    "Amazon Computers": "Computers",
    "Coauthor CS": "CS",
}

SLP_COLS = ["slp_0_05", "slp_05_10", "slp_10_15", "slp_15_20"]
SELECTORS = {
    "homophily": ["homophily"],
    "slp": SLP_COLS,
    "homophily+slp": ["homophily", *SLP_COLS],
}


def load_features() -> pd.DataFrame:
    with open(ROOT / "analysis" / "metrics.json") as f:
        metrics = json.load(f)

    rows = []
    for display_name, m in metrics.items():
        grid = np.asarray(m["eigenvalues"], dtype=float)
        cdf = np.asarray(m["cdf"], dtype=float)
        cutoffs = np.interp([0.5, 1.0, 1.5], grid, cdf)
        bands = np.diff(np.r_[0.0, cutoffs, cdf[-1]])
        rows.append(
            {
                "dataset": DISPLAY_TO_DATASET.get(display_name, display_name),
                "homophily": float(m.get("class_homophily", m["homophily"])),
                **dict(zip(SLP_COLS, bands)),
            }
        )

    return pd.DataFrame(rows).set_index("dataset").sort_index()


def load_results() -> pd.DataFrame:
    frames = [
        pd.read_csv(ROOT / "spatial" / "runs" / run / "summary.csv", keep_default_na=False)
        for run in RUNS
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.assign(
        test_acc=pd.to_numeric(df["test_acc"]),
        depth=pd.to_numeric(df["depth"]),
        hidden=pd.to_numeric(df["hidden"]),
        eps=df["eps"].astype(str),
    )

    grouped = (
        df.groupby(["dataset", "model", "depth", "hidden", "eps"], as_index=False)["test_acc"]
        .agg(acc_mean="mean", acc_std="std", count="count")
    )
    grouped["config"] = list(zip(grouped["depth"], grouped["hidden"], grouped["eps"]))
    return grouped


def oracle_rows(results: pd.DataFrame) -> pd.DataFrame:
    return results.loc[results.groupby(["dataset", "model"])["acc_mean"].idxmax()]


def target_value(row: pd.Series, target: str):
    return row["config"] if target == "full" else row[target]


def default_choice(train_rows: pd.DataFrame, target: str):
    if target == "full":
        return train_rows.groupby("config")["acc_mean"].mean().idxmax()

    best_per_dataset = train_rows.groupby(["dataset", target])["acc_mean"].max().reset_index()
    return best_per_dataset.groupby(target)["acc_mean"].mean().idxmax()


def nearest_choice(
    features: pd.DataFrame,
    oracles: pd.DataFrame,
    train_datasets: list[str],
    test_dataset: str,
    model: str,
    cols: list[str],
    target: str,
):
    scaler = StandardScaler().fit(features.loc[train_datasets, cols])
    x_train = scaler.transform(features.loc[train_datasets, cols])
    x_test = scaler.transform(features.loc[[test_dataset], cols])
    source = train_datasets[int(np.linalg.norm(x_train - x_test, axis=1).argmin())]
    row = oracles[(oracles["dataset"] == source) & (oracles["model"] == model)].iloc[0]
    return target_value(row, target), source


def score_choice(test_rows: pd.DataFrame, target: str, choice):
    rows = test_rows[test_rows["config"] == choice] if target == "full" else test_rows[test_rows[target] == choice]
    if rows.empty:
        return np.nan, ()

    row = rows.loc[rows["acc_mean"].idxmax()]
    return float(row["acc_mean"]), row["config"]


def run_experiment(results: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    oracles = oracle_rows(results)
    records = []

    for model, model_rows in results.groupby("model", sort=True):
        datasets = sorted(set(model_rows["dataset"]) & set(features.index))
        targets = ["full", "depth", "hidden"] + (["eps"] if model == "fagcn" else [])

        for test_dataset in datasets:
            train_datasets = [d for d in datasets if d != test_dataset]
            train_rows = model_rows[model_rows["dataset"].isin(train_datasets)]
            test_rows = model_rows[model_rows["dataset"] == test_dataset]
            oracle = oracles[(oracles["dataset"] == test_dataset) & (oracles["model"] == model)].iloc[0]

            for target in targets:
                choices = {"default": (default_choice(train_rows, target), "")}
                choices.update(
                    {
                        name: nearest_choice(features, oracles, train_datasets, test_dataset, model, cols, target)
                        for name, cols in SELECTORS.items()
                    }
                )

                for selector, (choice, source) in choices.items():
                    chosen_acc, chosen_config = score_choice(test_rows, target, choice)
                    records.append(
                        {
                            "model": model,
                            "dataset": test_dataset,
                            "target": target,
                            "selector": selector,
                            "source_dataset": source,
                            "oracle_value": target_value(oracle, target),
                            "chosen_value": choice,
                            "oracle_config": oracle["config"],
                            "chosen_config": chosen_config,
                            "oracle_acc": float(oracle["acc_mean"]),
                            "chosen_acc": chosen_acc,
                            "regret": float(oracle["acc_mean"]) - chosen_acc,
                            "match": choice == target_value(oracle, target),
                        }
                    )

    return pd.DataFrame(records)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "mean_regret": ("regret", "mean"),
        "median_regret": ("regret", "median"),
        "exact_match": ("match", "mean"),
        "mean_chosen_acc": ("chosen_acc", "mean"),
        "mean_oracle_acc": ("oracle_acc", "mean"),
        "folds": ("dataset", "count"),
    }
    overall = details.groupby(["target", "selector"], as_index=False).agg(**metrics)
    overall.insert(0, "model", "ALL")
    by_model = details.groupby(["model", "target", "selector"], as_index=False).agg(**metrics)
    return pd.concat([overall, by_model], ignore_index=True).sort_values(
        ["target", "model", "mean_regret", "selector"]
    )


def main() -> None:
    details = run_experiment(load_results(), load_features())
    summary = summarize(details)

    summary_path = OUT_DIR / "summary.csv"
    details_path = OUT_DIR / "details.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)

    full = summary[(summary["model"] == "ALL") & (summary["target"] == "full")]
    focus = summary[(summary["model"] == "ALL") & (summary["target"].isin(["depth", "eps"]))]

    print("Overall full-config selection")
    print(full[["selector", "mean_regret", "median_regret", "exact_match", "folds"]].to_string(index=False))
    print("\nFilter-relevant hyperparameter targets")
    print(focus[["target", "selector", "mean_regret", "median_regret", "exact_match", "folds"]].to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {details_path}")


if __name__ == "__main__":
    main()
