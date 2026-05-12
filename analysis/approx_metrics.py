"""Add approximate SLP-CDF entries to metrics.json for larger datasets.

The output format keeps the original metrics.json fields (num_classes, num_nodes, num_edges, homophily, eigenvalues,
cdf) and adds metadata describing the approximate SLP run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/spectral-profiling-matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from spatial import datasets as spatial_datasets
from sparsify_tradeoff import (
    cdf_on_grid,
    compute_lanczos_slp,
    graph_from_dataset,
    run_laplacians_sparsify,
)


DEFAULT_METRICS_PATH = Path(__file__).with_name("metrics.json")
DEFAULT_GRID_SIZE = 50
DEFAULT_STEPS = 64

METRICS_NAME = {
    "Photo": "Amazon Photo",
    "Computers": "Amazon Computers",
    "CS": "Coauthor CS",
    "RomanEmpire": "Roman Empire",
    "AmazonRatings": "Amazon Ratings",
}


def canonical_name(name: str) -> str:
    return name.replace("-", "").replace("_", "").replace(" ", "").lower()


def metrics_name(dataset_name: str) -> str:
    return METRICS_NAME.get(dataset_name, dataset_name)


def select_datasets(requested_names: list[str] | None, data_root: Path | None):
    if data_root is not None:
        spatial_datasets.data_dir = str(data_root)
        data_root.mkdir(parents=True, exist_ok=True)

    available = spatial_datasets.build_datasets()
    if requested_names is None:
        return available

    lookup = {}
    for name, dataset in available:
        lookup[canonical_name(name)] = (name, dataset)
        lookup[canonical_name(metrics_name(name))] = (name, dataset)

    selected = []
    missing = []
    for requested_name in requested_names:
        match = lookup.get(canonical_name(requested_name))
        if match is None:
            missing.append(requested_name)
        else:
            selected.append(match)

    if missing:
        choices = ", ".join(metrics_name(name) for name, _ in available)
        raise ValueError(f"unknown dataset(s): {', '.join(missing)}. Available: {choices}")

    return selected


def data_field(data, field: str):
    return data[field] if isinstance(data, dict) else getattr(data, field)


def dataset_num_classes(dataset, labels: torch.Tensor) -> int:
    value = getattr(dataset, "num_classes", None)
    if value is not None:
        return int(value)
    return int(labels.max().item()) + 1


def compute_homophily(edge_index: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    labels = labels.detach().cpu().long().reshape(-1)
    edge_index = edge_index.detach().cpu().long()
    src = edge_index[0]
    dst = edge_index[1]
    idx = labels[src] * num_classes + labels[dst]
    co = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes).float()
    return (co.diag() / co.sum(dim=1).clamp(min=1)).mean().item()


def approximate_slp_profile(
    dataset,
    grid: np.ndarray,
    epsilon: float | None,
    steps: int,
    julia_bin: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    adjacency, labels, num_classes = graph_from_dataset(dataset)
    full_edges = int(adjacency.nnz // 2)
    slp_adjacency = adjacency
    sparsify_time_sec = 0.0
    sparse_edges = full_edges

    if epsilon is not None:
        epsilon, sparsify_time_sec, slp_adjacency = run_laplacians_sparsify(
            adjacency,
            [epsilon],
            julia_bin,
            seed,
        )[0]
        sparse_edges = int(slp_adjacency.nnz // 2)

    evals, _, cdf, slp_time_sec = compute_lanczos_slp(
        slp_adjacency,
        labels,
        num_classes,
        steps,
    )
    profile = cdf_on_grid(evals, cdf, grid)
    method = "sparsified_lanczos" if epsilon is not None else "lanczos"

    return profile, {
        "slp_method": method,
        "slp_steps": int(steps),
        "slp_epsilon": None if epsilon is None else float(epsilon),
        "slp_graph_edges": sparse_edges,
        "slp_edge_fraction": sparse_edges / full_edges if full_edges else None,
        "sparsify_time_sec": float(sparsify_time_sec),
        "slp_time_sec": float(slp_time_sec),
        "pipeline_time_sec": float(sparsify_time_sec + slp_time_sec),
    }


def compute_dataset_metrics(
    dataset_name: str,
    dataset,
    grid: np.ndarray,
    epsilon: float | None,
    steps: int,
    julia_bin: str,
    seed: int,
) -> dict[str, object]:
    data = dataset[0]
    labels = data_field(data, "y")
    edge_index = data_field(data, "edge_index")
    num_classes = dataset_num_classes(dataset, labels)
    num_nodes = int(getattr(data, "num_nodes", 0) or labels.numel())
    num_edges = int(edge_index.size(1))
    homophily = compute_homophily(edge_index, labels, num_classes)
    profile, slp_metadata = approximate_slp_profile(dataset, grid, epsilon, steps, julia_bin, seed)

    return {
        "num_classes": int(num_classes),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "homophily": float(homophily),
        "eigenvalues": grid.tolist(),
        "cdf": profile.tolist(),
        **slp_metadata,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_existing_metrics(path: Path, replace: bool) -> dict[str, object]:
    if replace or not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_metrics(results: dict[str, dict[str, object]], path: Path) -> None:
    if not results:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = min(4, max(1, len(results)))
    n_rows = int(np.ceil(len(results) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.3 * n_cols, 2.8 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for ax, (name, row) in zip(axes_flat, results.items()):
        ax.step(row["eigenvalues"], row["cdf"], where="post", linewidth=1.4)
        method = row.get("slp_method", "unknown")
        ax.set_title(f"{name} ({method}, h={row['homophily']:.2f})", fontsize=9)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(results) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("lambda")
    for ax in axes[:, 0]:
        ax.set_ylabel("SLP CDF")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add approximate Lanczos SLP-CDF entries to a metrics.json file."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to compute. Accepts build_datasets names or metrics names.",
    )
    parser.add_argument(
        "--epsilon",
        "--eps",
        type=float,
        default=None,
        help="Optional Laplacians.jl sparsification epsilon before approximate SLP.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--julia-bin", default="julia")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Number of lambda grid points stored in metrics.json.",
    )
    parser.add_argument("--replace", action="store_true", help="Replace --out instead of adding/updating entries.")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only compute datasets not already present in --out.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--plot-out", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional override for spatial.datasets.data_dir before calling build_datasets().",
    )

    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.epsilon is not None and not (0.0 < args.epsilon < 1.0):
        parser.error("--epsilon must be in (0, 1)")
    if args.epsilon is not None and shutil.which(args.julia_bin) is None:
        parser.error(f"Julia executable not found: {args.julia_bin}")
    return args


def main() -> None:
    args = parse_args()
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    plot_out = args.plot_out or out.with_suffix(".png")
    plot_out = plot_out if plot_out.is_absolute() else REPO_ROOT / plot_out

    results = load_existing_metrics(out, args.replace)
    selected = select_datasets(args.datasets, args.data_root)
    grid = np.linspace(0.0, 2.0, num=args.grid_size)

    for dataset_name, dataset in selected:
        output_name = metrics_name(dataset_name)
        if args.missing_only and output_name in results:
            print(f"Skipping {output_name}: already in {out}")
            continue

        print(f"Computing {output_name} with approximate SLP")
        results[output_name] = compute_dataset_metrics(
            dataset_name,
            dataset,
            grid,
            args.epsilon,
            args.steps,
            args.julia_bin,
            args.seed,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    if not args.no_plot:
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plot_metrics(results, plot_out)

    print(f"Wrote {out}")
    if not args.no_plot:
        print(f"Wrote {plot_out}")


if __name__ == "__main__":
    main()
