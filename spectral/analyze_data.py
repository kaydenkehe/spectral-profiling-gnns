import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/spectral-profiling-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def analyze_train(run_dir, out):
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


def dataset_key(name):
    aliases = {
        "amazon photo": "photo",
        "amazon computers": "computers",
        "coauthor cs": "cs",
    }
    key = str(name).strip().lower()
    return aliases.get(key, key)


def resolve_existing_path(path):
    path = Path(path)
    if path.exists() or path.is_absolute():
        return path

    for base in (Path.cwd(), REPO_ROOT, SCRIPT_DIR):
        candidate = base / path
        if candidate.exists():
            return candidate

    return path


def make_jacobi_heatmaps(df, out):
    if not {"a", "b", "K", "mean_test_acc"}.issubset(df.columns):
        return []

    heatmap_dir = out / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for (dataset, K), group in df.groupby(["dataset", "K"]):
        table = group.pivot_table(
            index="b",
            columns="a",
            values="mean_test_acc",
            aggfunc="max",
        ).sort_index().sort_index(axis=1)

        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(
            table.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        ax.set_title(f"{dataset} Jacobi landscape, K={K}")
        ax.set_xlabel("a")
        ax.set_ylabel("b")

        x_positions = np.linspace(0, len(table.columns) - 1, min(len(table.columns), 8), dtype=int)
        y_positions = np.linspace(0, len(table.index) - 1, min(len(table.index), 8), dtype=int)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{table.columns[i]:.2g}" for i in x_positions], rotation=45, ha="right")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{table.index[i]:.2g}" for i in y_positions])

        best = group.loc[group["mean_test_acc"].idxmax()]
        best_x = list(table.columns).index(best["a"])
        best_y = list(table.index).index(best["b"])
        ax.scatter(best_x, best_y, marker="x", s=80, c="white", linewidths=2)

        fig.colorbar(image, ax=ax, label="mean test accuracy")
        fig.tight_layout()

        safe_dataset = str(dataset).replace(" ", "_").replace("/", "_")
        path = heatmap_dir / f"{safe_dataset}_K{K}_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)

        rows.append({
            "dataset": dataset,
            "K": K,
            "path": str(path),
            "best_a": best["a"],
            "best_b": best["b"],
            "best_mean_test_acc": best["mean_test_acc"],
        })

    if rows:
        pd.DataFrame(rows).to_csv(out / "jacobi_heatmap_index.csv", index=False)
    return rows


def slp_features(metrics_path, n_bins):
    with open(metrics_path) as f:
        metrics = json.load(f)

    edges = np.linspace(0.0, 2.0, n_bins + 1)
    rows = []

    for name, result in metrics.items():
        lambdas = np.asarray(result["eigenvalues"], dtype=float)
        cdf = np.asarray(result["cdf"], dtype=float)
        cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
        cdf_edges = np.interp(edges, lambdas, cdf)
        mass = np.diff(cdf_edges)

        row = {
            "dataset": name,
            "dataset_key": dataset_key(name),
            "homophily": result.get("homophily"),
            "num_nodes": result.get("num_nodes"),
            "num_edges": result.get("num_edges"),
        }
        for i, value in enumerate(mass):
            row[f"slp_bin_{i:02d}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def plot_regression_predictions(predictions, K, n_bins, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, target in zip(axes, ["a", "b"]):
        actual = predictions[f"actual_{target}"]
        predicted = predictions[f"pred_{target}"]
        ax.scatter(actual, predicted, s=40)
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(f"actual {target}")
        ax.set_ylabel(f"LOO predicted {target}")
        ax.set_title(f"{target}, K={K}, {n_bins} SLP bins")
        ax.grid(alpha=0.25)
        for _, row in predictions.iterrows():
            ax.annotate(row["dataset"], (row[f"actual_{target}"], row[f"pred_{target}"]), fontsize=7)
    fig.tight_layout()
    path = out / f"slp_regression_K{K}_bins_{n_bins}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_slp_regression(winners, metrics_path, out, min_bins, max_bins):
    metrics_path = resolve_existing_path(metrics_path)
    if not metrics_path.exists():
        print(f"\nSkipping SLP regression: missing {metrics_path}")
        return

    targets = winners.copy()
    targets["dataset_key"] = targets["dataset"].map(dataset_key)
    target_cols = ["a", "b"]
    alpha_grid = np.logspace(-4, 4, 17)
    summary_rows = []
    all_predictions = []
    coef_rows = []

    for K, k_targets in targets.groupby("K"):
        for n_bins in range(min_bins, max_bins + 1):
            features = slp_features(metrics_path, n_bins)
            feature_cols = [c for c in features.columns if c.startswith("slp_bin_")]
            merged = k_targets.merge(features, on="dataset_key", suffixes=("", "_metrics"))
            if len(merged) < 3:
                continue

            X = merged[feature_cols].to_numpy(dtype=float)
            y = merged[target_cols].to_numpy(dtype=float)

            model = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=alpha_grid),
            )
            model.fit(X, y)
            train_pred = model.predict(X)

            loo_pred = np.zeros_like(y)
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(X):
                loo_model = make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=alpha_grid),
                )
                loo_model.fit(X[train_idx], y[train_idx])
                loo_pred[test_idx] = loo_model.predict(X[test_idx])

            pred_rows = merged[["dataset", "K", "a", "b", "mean_test_acc"]].copy()
            pred_rows = pred_rows.rename(columns={"a": "actual_a", "b": "actual_b"})
            pred_rows["n_bins"] = n_bins
            pred_rows["pred_a"] = loo_pred[:, 0]
            pred_rows["pred_b"] = loo_pred[:, 1]
            pred_rows["abs_error_a"] = (pred_rows["actual_a"] - pred_rows["pred_a"]).abs()
            pred_rows["abs_error_b"] = (pred_rows["actual_b"] - pred_rows["pred_b"]).abs()
            all_predictions.append(pred_rows)
            pred_rows.to_csv(
                out / f"slp_regression_predictions_K{K}_bins_{n_bins}.csv",
                index=False,
            )
            plot_regression_predictions(pred_rows, K, n_bins, out)

            ridge = model.named_steps["ridgecv"]
            alpha = ridge.alpha_
            coef = ridge.coef_
            intercept = ridge.intercept_
            for target_idx, target_name in enumerate(target_cols):
                for feature_name, coef_value in zip(feature_cols, coef[target_idx]):
                    coef_rows.append({
                        "K": K,
                        "n_bins": n_bins,
                        "target": target_name,
                        "feature": feature_name,
                        "coef_standardized": coef_value,
                        "intercept": intercept[target_idx],
                        "alpha": alpha,
                    })

            summary_rows.append({
                "K": K,
                "n_bins": n_bins,
                "n_datasets": len(merged),
                "train_r2_a": r2_score(y[:, 0], train_pred[:, 0]),
                "train_r2_b": r2_score(y[:, 1], train_pred[:, 1]),
                "loo_mae_a": mean_absolute_error(y[:, 0], loo_pred[:, 0]),
                "loo_mae_b": mean_absolute_error(y[:, 1], loo_pred[:, 1]),
                "loo_rmse_a": np.sqrt(mean_squared_error(y[:, 0], loo_pred[:, 0])),
                "loo_rmse_b": np.sqrt(mean_squared_error(y[:, 1], loo_pred[:, 1])),
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values(["K", "loo_mae_a", "loo_mae_b"])
        summary.to_csv(out / "slp_regression_summary.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(
            out / "slp_regression_predictions.csv",
            index=False,
        )
        pd.DataFrame(coef_rows).to_csv(out / "slp_regression_coefficients.csv", index=False)

        print("\nSLP regression summary")
        print(summary.to_string(index=False))


def analyze_jacobi(csv_path, out, metrics_path, min_bins, max_bins, make_plots):
    df = pd.read_csv(csv_path)
    full_landscape = {"a", "b", "K", "mean_test_acc"}.issubset(df.columns)
    if "mean_test_acc" in df.columns:
        df["mean_test_acc"] = df["mean_test_acc"].astype(float)
        df["std_test_acc"] = df["std_test_acc"].astype(float)
    elif {"best_a", "best_b", "best_acc"}.issubset(df.columns):
        df = df.rename(columns={
            "best_a": "a",
            "best_b": "b",
            "best_acc": "mean_test_acc",
            "best_std": "std_test_acc",
        })
        df["K"] = -1
        df["n_seeds"] = np.nan
    else:
        raise ValueError("Jacobi mode expects columns a,b,mean_test_acc or best_a,best_b,best_acc.")

    sorted_results = df.sort_values(
        ["dataset", "mean_test_acc"], ascending=[True, False]
    ).reset_index(drop=True)
    sorted_results.to_csv(out / "jacobi_configs_ranked.csv", index=False)

    winners = (
        sorted_results.groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )
    winners.to_csv(out / "jacobi_winners_by_dataset.csv", index=False)

    best_by_dataset_k = (
        df.sort_values(["dataset", "K", "mean_test_acc"], ascending=[True, True, False])
        .groupby(["dataset", "K"])
        .head(1)
        .reset_index(drop=True)
    )
    best_by_dataset_k.to_csv(out / "jacobi_best_ab_by_dataset_k.csv", index=False)

    k_wins = (
        winners.groupby("K")
        .size()
        .reset_index(name="n_dataset_wins")
        .sort_values("n_dataset_wins", ascending=False)
    )
    k_wins.to_csv(out / "jacobi_k_win_counts.csv", index=False)

    ab_wins = (
        winners.groupby(["a", "b"])
        .size()
        .reset_index(name="n_dataset_wins")
        .sort_values("n_dataset_wins", ascending=False)
    )
    ab_wins.to_csv(out / "jacobi_ab_win_counts.csv", index=False)

    print("\nJacobi winners by dataset")
    cols = ["dataset", "K", "a", "b", "mean_test_acc", "std_test_acc", "n_seeds"]
    print(winners[cols].to_string(index=False))

    print("\nJacobi K win counts")
    print(k_wins.to_string(index=False))

    print("\nJacobi (a,b) win counts")
    print(ab_wins.to_string(index=False))

    if make_plots and full_landscape:
        heatmaps = make_jacobi_heatmaps(df, out)
        print(f"\nWrote {len(heatmaps)} Jacobi heatmaps")

    run_slp_regression(best_by_dataset_k, metrics_path, out, min_bins, max_bins)


def infer_mode(input_path, requested_mode):
    if requested_mode != "auto":
        return requested_mode
    if input_path.is_dir():
        return "train"

    cols = set(pd.read_csv(input_path, nrows=0).columns)
    if {"a", "b", "mean_test_acc"}.issubset(cols) or {"best_a", "best_b", "best_acc"}.issubset(cols):
        return "jacobi"
    return "train"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="train_spectral run dir or jacobi_ab_sweep CSV")
    p.add_argument("--mode", choices=["auto", "train", "jacobi"], default="auto")
    p.add_argument("--metrics", default="analysis/metrics.json")
    p.add_argument("--slp-min-bins", type=int, default=4)
    p.add_argument("--slp-max-bins", type=int, default=10)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    input_path = Path(args.input)
    mode = infer_mode(input_path, args.mode)
    default_out = input_path / "analysis" if input_path.is_dir() else input_path.with_suffix("") / "analysis"
    out = Path(args.out) if args.out else default_out
    out.mkdir(parents=True, exist_ok=True)

    if mode == "train":
        analyze_train(input_path, out)
    else:
        analyze_jacobi(
            input_path,
            out,
            Path(args.metrics),
            args.slp_min_bins,
            args.slp_max_bins,
            not args.no_plots,
        )

    print(f"\nWrote analysis tables to {out}")


if __name__ == "__main__":
    main()
