import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/spectral-profiling-matplotlib")
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def compute_homophily(edge_index, labels):
    import torch

    num_classes = int(labels.max().item()) + 1
    src, dst = edge_index
    idx = labels[src] * num_classes + labels[dst]
    co = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes).float()
    return (co.diag() / co.sum(dim=1).clamp(min=1)).mean().item()


def compute_spectrum(edge_index, n):
    import torch
    from torch_geometric.utils import get_laplacian, to_dense_adj

    edge_index, edge_weight = get_laplacian(
        edge_index,
        normalization="sym",
        num_nodes=n,
    )
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=n)[0]
    evals, evecs = torch.linalg.eigh(L)
    return evals, evecs


def compute_slp(evecs, labels, num_classes):
    import torch
    import torch.nn.functional as F

    Y = F.one_hot(labels, num_classes=num_classes).float()
    Y_tilde = Y - Y.mean(dim=0, keepdim=True)
    proj = (evecs.T @ Y_tilde) ** 2
    Y_norm = torch.norm(Y_tilde, dim=0) ** 2 + 1e-8
    pi_c = proj / Y_norm
    pi = pi_c.mean(dim=1)
    return torch.cumsum(pi, dim=0)


def compute_slp_profiles(dataset_names, out):
    from datasets import build_datasets

    dataset_map = {
        dataset_key(name): (name, dataset)
        for name, dataset in build_datasets()
    }
    profile_rows = []
    missing = []

    for dataset_name in sorted(set(dataset_names)):
        key = dataset_key(dataset_name)
        if key not in dataset_map:
            missing.append(dataset_name)
            continue

        canonical_name, dataset = dataset_map[key]
        data = dataset[0].cpu()
        num_classes = dataset.num_classes
        print(f"Computing SLP for {dataset_name}: {data.num_nodes} nodes")

        evals, evecs = compute_spectrum(data.edge_index, data.num_nodes)
        cdf = compute_slp(evecs, data.y, num_classes)
        homophily = compute_homophily(data.edge_index, data.y)

        for lambd, value in zip(evals.cpu().numpy(), cdf.cpu().numpy()):
            profile_rows.append({
                "dataset": dataset_name,
                "dataset_key": key,
                "canonical_dataset": canonical_name,
                "num_nodes": data.num_nodes,
                "num_edges": data.edge_index.size(1),
                "num_features": dataset.num_features,
                "num_classes": num_classes,
                "homophily": homophily,
                "eigenvalue": float(lambd),
                "slp_cdf": float(value),
            })

    if missing:
        print(f"\nMissing datasets for on-the-fly SLP: {', '.join(sorted(missing))}")

    profiles = pd.DataFrame(profile_rows)
    if not profiles.empty:
        profiles.to_csv(out / "slp_profiles.csv", index=False)
    return profiles


def make_jacobi_heatmaps(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def slp_features_from_profiles(profiles, n_bins, out=None):
    edges = np.linspace(0.0, 2.0, n_bins + 1)
    rows = []

    for key, group in profiles.groupby("dataset_key"):
        group = group.sort_values("eigenvalue")
        lambdas = group["eigenvalue"].to_numpy(dtype=float)
        cdf = group["slp_cdf"].to_numpy(dtype=float)
        cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
        cdf_edges = np.interp(edges, lambdas, cdf)
        mass = np.diff(cdf_edges)
        first = group.iloc[0]

        row = {
            "dataset": first["dataset"],
            "dataset_key": key,
            "homophily": first["homophily"],
            "num_nodes": first["num_nodes"],
            "num_edges": first["num_edges"],
            "num_features": first["num_features"],
            "num_classes": first["num_classes"],
            "n_bins": n_bins,
        }
        for i, value in enumerate(mass):
            row[f"slp_bin_{i:02d}"] = value
        rows.append(row)

    features = pd.DataFrame(rows)
    if out is not None and not features.empty:
        features.to_csv(out / f"slp_features_bins_{n_bins}.csv", index=False)
    return features


def select_feature_cols(features, feature_set):
    slp_cols = [c for c in features.columns if c.startswith("slp_bin_")]
    if feature_set == "slp":
        return slp_cols
    if feature_set == "homophily":
        return ["homophily"]
    if feature_set == "slp_homophily":
        return slp_cols + ["homophily"]
    raise ValueError(f"Unknown feature set: {feature_set}")


def plot_regression_predictions(predictions, K, n_bins, out, feature_set):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        ax.set_title(f"{target}, K={K}, {feature_set}, bins={n_bins}")
        ax.grid(alpha=0.25)
        for _, row in predictions.iterrows():
            ax.annotate(row["dataset"], (row[f"actual_{target}"], row[f"pred_{target}"]), fontsize=7)
    fig.tight_layout()
    path = out / f"regression_{feature_set}_K{K}_bins_{n_bins}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def lookup_nearest_config(landscape, dataset, K, pred_a, pred_b):
    choices = landscape[
        (landscape["dataset"] == dataset)
        & (landscape["K"] == K)
    ]
    if choices.empty:
        return np.nan, np.nan, np.nan

    distance = (choices["a"] - pred_a) ** 2 + (choices["b"] - pred_b) ** 2
    row = choices.loc[distance.idxmin()]
    return row["a"], row["b"], row["mean_test_acc"]


def run_slp_regression(winners, landscape, out, min_bins, max_bins, make_plots, feature_set):
    targets = winners.copy()
    targets["dataset_key"] = targets["dataset"].map(dataset_key)
    profiles = compute_slp_profiles(targets["dataset"], out)
    if profiles.empty:
        print("\nSkipping SLP regression: no SLP profiles were computed.")
        return

    target_cols = ["a", "b"]
    alpha_grid = np.logspace(-4, 4, 17)
    summary_rows = []
    all_predictions = []
    coef_rows = []

    for K, k_targets in targets.groupby("K"):
        for n_bins in range(min_bins, max_bins + 1):
            features = slp_features_from_profiles(profiles, n_bins, out=out)
            feature_cols = select_feature_cols(features, feature_set)
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
            loo_mean_pred = np.zeros_like(y)
            loo = LeaveOneOut()
            for train_idx, test_idx in loo.split(X):
                loo_model = make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=alpha_grid),
                )
                loo_model.fit(X[train_idx], y[train_idx])
                loo_pred[test_idx] = loo_model.predict(X[test_idx])
                loo_mean_pred[test_idx] = y[train_idx].mean(axis=0)

            pred_rows = merged[["dataset", "K", "a", "b", "mean_test_acc"]].copy()
            pred_rows = pred_rows.rename(columns={"a": "actual_a", "b": "actual_b"})
            pred_rows["feature_set"] = feature_set
            pred_rows["n_bins"] = n_bins
            pred_rows["pred_a"] = loo_pred[:, 0]
            pred_rows["pred_b"] = loo_pred[:, 1]
            pred_rows["mean_baseline_a"] = loo_mean_pred[:, 0]
            pred_rows["mean_baseline_b"] = loo_mean_pred[:, 1]
            pred_rows["abs_error_a"] = (pred_rows["actual_a"] - pred_rows["pred_a"]).abs()
            pred_rows["abs_error_b"] = (pred_rows["actual_b"] - pred_rows["pred_b"]).abs()
            pred_rows["mean_baseline_abs_error_a"] = (
                pred_rows["actual_a"] - pred_rows["mean_baseline_a"]
            ).abs()
            pred_rows["mean_baseline_abs_error_b"] = (
                pred_rows["actual_b"] - pred_rows["mean_baseline_b"]
            ).abs()

            snapped = []
            baseline_snapped = []
            for _, row in pred_rows.iterrows():
                snapped.append(lookup_nearest_config(
                    landscape,
                    row["dataset"],
                    row["K"],
                    row["pred_a"],
                    row["pred_b"],
                ))
                baseline_snapped.append(lookup_nearest_config(
                    landscape,
                    row["dataset"],
                    row["K"],
                    row["mean_baseline_a"],
                    row["mean_baseline_b"],
                ))

            pred_rows[["pred_grid_a", "pred_grid_b", "pred_grid_acc"]] = pd.DataFrame(
                snapped,
                index=pred_rows.index,
            )
            pred_rows[[
                "mean_baseline_grid_a",
                "mean_baseline_grid_b",
                "mean_baseline_grid_acc",
            ]] = pd.DataFrame(
                baseline_snapped,
                index=pred_rows.index,
            )
            pred_rows["pred_grid_regret"] = (
                pred_rows["mean_test_acc"] - pred_rows["pred_grid_acc"]
            )
            pred_rows["mean_baseline_grid_regret"] = (
                pred_rows["mean_test_acc"] - pred_rows["mean_baseline_grid_acc"]
            )
            pred_rows["exact_grid_match"] = (
                (pred_rows["pred_grid_a"] == pred_rows["actual_a"])
                & (pred_rows["pred_grid_b"] == pred_rows["actual_b"])
            )
            pred_rows["mean_baseline_exact_grid_match"] = (
                (pred_rows["mean_baseline_grid_a"] == pred_rows["actual_a"])
                & (pred_rows["mean_baseline_grid_b"] == pred_rows["actual_b"])
            )
            all_predictions.append(pred_rows)
            pred_rows.to_csv(
                out / f"slp_regression_predictions_K{K}_bins_{n_bins}.csv",
                index=False,
            )
            if make_plots:
                plot_regression_predictions(pred_rows, K, n_bins, out, feature_set)

            ridge = model.named_steps["ridgecv"]
            alpha = ridge.alpha_
            coef = ridge.coef_
            intercept = ridge.intercept_
            for target_idx, target_name in enumerate(target_cols):
                for feature_name, coef_value in zip(feature_cols, coef[target_idx]):
                    coef_rows.append({
                        "K": K,
                        "n_bins": n_bins,
                        "feature_set": feature_set,
                        "target": target_name,
                        "feature": feature_name,
                        "coef_standardized": coef_value,
                        "intercept": intercept[target_idx],
                        "alpha": alpha,
                    })

            summary_rows.append({
                "K": K,
                "n_bins": n_bins,
                "feature_set": feature_set,
                "n_datasets": len(merged),
                "train_r2_a": r2_score(y[:, 0], train_pred[:, 0]),
                "train_r2_b": r2_score(y[:, 1], train_pred[:, 1]),
                "loo_mae_a": mean_absolute_error(y[:, 0], loo_pred[:, 0]),
                "loo_mae_b": mean_absolute_error(y[:, 1], loo_pred[:, 1]),
                "mean_baseline_loo_mae_a": mean_absolute_error(y[:, 0], loo_mean_pred[:, 0]),
                "mean_baseline_loo_mae_b": mean_absolute_error(y[:, 1], loo_mean_pred[:, 1]),
                "loo_rmse_a": np.sqrt(mean_squared_error(y[:, 0], loo_pred[:, 0])),
                "loo_rmse_b": np.sqrt(mean_squared_error(y[:, 1], loo_pred[:, 1])),
                "mean_pred_grid_regret": pred_rows["pred_grid_regret"].mean(),
                "mean_baseline_grid_regret": pred_rows["mean_baseline_grid_regret"].mean(),
                "exact_grid_match_rate": pred_rows["exact_grid_match"].mean(),
                "mean_baseline_exact_grid_match_rate": pred_rows[
                    "mean_baseline_exact_grid_match"
                ].mean(),
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values(["K", "loo_mae_a", "loo_mae_b"])
        summary.to_csv(out / "slp_regression_summary.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(
            out / "slp_regression_predictions.csv",
            index=False,
        )
        pd.DataFrame(coef_rows).to_csv(out / "slp_regression_coefficients.csv", index=False)

        print("\nRegression summary")
        print(summary.to_string(index=False))


def analyze_jacobi(csv_path, out, min_bins, max_bins, make_plots, feature_set):
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

    run_slp_regression(
        best_by_dataset_k,
        df,
        out,
        min_bins,
        max_bins,
        make_plots,
        feature_set,
    )


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
    p.add_argument("--feature-set", choices=["slp", "homophily", "slp_homophily"],
                   default="slp_homophily")
    p.add_argument("--slp-min-bins", type=int, default=1)
    p.add_argument("--slp-max-bins", type=int, default=10)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    input_path = Path(args.input)
    mode = infer_mode(input_path, args.mode)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    analysis_root = input_path / "analysis" if input_path.is_dir() else input_path.with_suffix("") / "analysis"
    default_out = analysis_root / timestamp
    out = Path(args.out) if args.out else default_out
    out.mkdir(parents=True, exist_ok=True)

    if mode == "train":
        analyze_train(input_path, out)
    else:
        analyze_jacobi(
            input_path,
            out,
            args.slp_min_bins,
            args.slp_max_bins,
            not args.no_plots,
            args.feature_set,
        )

    print(f"\nWrote analysis tables to {out}")


if __name__ == "__main__":
    main()
