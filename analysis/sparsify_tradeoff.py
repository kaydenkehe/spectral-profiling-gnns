"""
Ground-truth full-graph SLPs are loaded from analysis/metrics.json. The
experiment computes Lanczos spectral-measure approximations on the original
graph and on sparsified graphs, then compares those profiles to the stored
reference using 1-Wasserstein distance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/spectral-profiling-matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wasserstein_distance

from spatial import datasets as spatial_datasets


DEFAULT_EPSILONS = [0.3, 0.6, 0.9]
DEFAULT_STEP_VALUES = [16, 32, 64]
JULIA_SCRIPT = Path(__file__).with_name("laplacians_sparsify.jl")
DEFAULT_METRICS_PATH = Path(__file__).with_name("metrics.json")


def canonical_name(name: str) -> str:
    return name.replace("-", "").replace("_", "").replace(" ", "").lower()


def select_datasets(requested_names: list[str] | None):

    available = spatial_datasets.build_datasets()
    if requested_names is None:
        return available

    lookup = {canonical_name(name): (name, dataset) for name, dataset in available}
    selected = []
    missing = []
    for requested_name in requested_names:
        dataset = lookup.get(canonical_name(requested_name))
        if dataset is None:
            missing.append(requested_name)
        else:
            selected.append(dataset)

    if missing:
        choices = ", ".join(name for name, _ in available)
        raise ValueError(f"unknown dataset(s): {', '.join(missing)}. Available: {choices}")

    return selected


def data_field(data, field: str):
    return data[field] if isinstance(data, dict) else getattr(data, field)


def build_adjacency(edges: np.ndarray, weights: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    if edges.size == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.concatenate([weights, weights]).astype(np.float64, copy=False)
    adjacency = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    return adjacency.tocsr()


def graph_from_dataset(dataset) -> tuple[sp.csr_matrix, np.ndarray, int]:
    data = dataset[0]
    edge_index = data_field(data, "edge_index").detach().cpu().numpy()
    labels = data_field(data, "y").detach().cpu().numpy().astype(np.int64).reshape(-1)

    if np.any(labels < 0):
        raise ValueError("SLP labels must be non-negative class ids")

    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    non_self = src != dst
    src = src[non_self]
    dst = dst[non_self]

    edges = np.column_stack([np.minimum(src, dst), np.maximum(src, dst)])
    edges = np.unique(edges, axis=0)
    weights = np.ones(edges.shape[0], dtype=np.float64)

    num_nodes = int(getattr(data, "num_nodes", 0) or labels.shape[0])
    num_classes = int(labels.max()) + 1
    return build_adjacency(edges, weights, num_nodes), labels, num_classes


def normalized_laplacian(adjacency: sp.csr_matrix) -> sp.csr_matrix:
    adjacency = ((adjacency + adjacency.T) * 0.5).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.divide(
        1.0,
        np.sqrt(degree),
        out=np.zeros_like(degree, dtype=np.float64),
        where=degree > 0,
    )
    scale = sp.diags(inv_sqrt, format="csr")
    laplacian = sp.eye(adjacency.shape[0], format="csr", dtype=np.float64) - scale @ adjacency @ scale
    return ((laplacian + laplacian.T) * 0.5).tocsr()


def valid_steps(steps: int | None, n: int) -> int:
    if n < 2:
        raise ValueError("Lanczos SLP needs at least two nodes")

    if steps is None:
        steps = min(128, n)

    return max(1, min(int(steps), n))


def valid_step_values(step_values: list[int], n: int) -> list[int]:
    values = sorted({valid_steps(steps, n) for steps in step_values})
    if not values:
        raise ValueError("at least one Lanczos step value is required")
    return values


def lanczos_measure(
    laplacian: sp.csr_matrix,
    start_vector: np.ndarray,
    steps: int,
    breakdown_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(start_vector)
    if norm <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    q = start_vector.astype(np.float64, copy=True) / norm
    q_prev = np.zeros_like(q)
    beta = 0.0
    basis = []
    alphas = []
    betas = []

    for _ in range(steps):
        basis.append(q)
        z = laplacian @ q
        if betas:
            z = z - beta * q_prev
        alpha = float(np.dot(q, z))
        z = z - alpha * q

        # Full reorthogonalization is cheap for the step counts here and makes
        # the quadrature much less brittle on graphs with clustered eigenvalues.
        for basis_vector in basis:
            z = z - float(np.dot(basis_vector, z)) * basis_vector

        beta_next = float(np.linalg.norm(z))
        alphas.append(alpha)
        if beta_next <= breakdown_tol:
            break

        betas.append(beta_next)
        q_prev = q
        q = z / beta_next
        beta = beta_next

    tridiagonal_size = len(alphas)
    tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
    if tridiagonal_size > 1:
        offdiag = np.asarray(betas[: tridiagonal_size - 1], dtype=np.float64)
        tridiagonal += np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)

    nodes, eigenvectors = np.linalg.eigh(tridiagonal)
    weights = eigenvectors[0, :] ** 2
    weight_sum = weights.sum()
    if weight_sum > 0.0:
        weights = weights / weight_sum
    return nodes, weights


def compute_lanczos_slp(
    adjacency: sp.csr_matrix,
    labels: np.ndarray,
    num_classes: int,
    steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = time.perf_counter()

    laplacian = normalized_laplacian(adjacency)
    steps = valid_steps(steps, laplacian.shape[0])
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    centered_labels = one_hot - one_hot.mean(axis=0, keepdims=True)

    eigenvalue_parts = []
    mass_parts = []
    nonzero_classes = 0
    for class_idx in range(num_classes):
        class_signal = centered_labels[:, class_idx]
        if np.dot(class_signal, class_signal) <= 1e-12:
            continue
        nodes, weights = lanczos_measure(laplacian, class_signal, steps)
        if nodes.size == 0:
            continue
        eigenvalue_parts.append(np.clip(nodes, 0.0, 2.0))
        mass_parts.append(np.clip(weights, 0.0, None))
        nonzero_classes += 1

    if nonzero_classes == 0:
        raise ValueError("all centered label vectors have zero norm")

    eigenvalues = np.concatenate(eigenvalue_parts)
    mass = np.concatenate([weights / nonzero_classes for weights in mass_parts])
    total = mass.sum()
    if total > 0.0:
        mass = mass / total

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    mass = mass[order]
    cdf = np.cumsum(mass)

    elapsed_sec = time.perf_counter() - start
    return eigenvalues, mass, cdf, elapsed_sec


def write_upper_edges(adjacency: sp.csr_matrix, path: Path) -> None:
    upper = sp.triu(adjacency, k=1).tocoo()
    with path.open("w", encoding="utf-8") as handle:
        for row, col, weight in zip(upper.row, upper.col, upper.data):
            handle.write(f"{row + 1},{col + 1},{float(weight)}\n")


def read_upper_edges(path: Path, num_nodes: int) -> sp.csr_matrix:
    if path.stat().st_size == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    edge_data = np.loadtxt(path, delimiter=",", ndmin=2)
    edges = edge_data[:, :2].astype(np.int64) - 1
    weights = edge_data[:, 2].astype(np.float64)
    return build_adjacency(edges, weights, num_nodes)


def run_laplacians_sparsify(
    adjacency: sp.csr_matrix,
    epsilons: list[float],
    julia_bin: str,
    seed: int,
) -> list[tuple[float, float, sp.csr_matrix]]:
    if not JULIA_SCRIPT.exists():
        raise FileNotFoundError(f"Julia bridge not found: {JULIA_SCRIPT}")

    with tempfile.TemporaryDirectory(prefix="slp_laplacians_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "full_edges.csv"
        write_upper_edges(adjacency, input_path)

        command = [
            julia_bin,
            "--startup-file=no",
            "--history-file=no",
            str(JULIA_SCRIPT),
            str(input_path),
            str(tmp_dir),
            str(adjacency.shape[0]),
            ",".join(str(epsilon) for epsilon in epsilons),
            str(seed),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "Laplacians.jl sparsification failed. Install Julia and Laplacians.jl, then retry.\n"
                "Julia package install example: julia -e 'using Pkg; Pkg.add(\"Laplacians\")'\n\n"
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            )

        stats = pd.read_csv(tmp_dir / "sparsify_stats.csv")
        sparsified = []
        for _, row in stats.iterrows():
            sparse_adjacency = read_upper_edges(Path(row["edge_file"]), adjacency.shape[0])
            sparsified.append((float(row["epsilon"]), float(row["sparsify_time_sec"]), sparse_adjacency))
        return sparsified


def load_metrics_reference(metrics_path: Path, dataset_name: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    lookup = {canonical_name(name): (name, data) for name, data in metrics.items()}
    aliases = {
        canonical_name("Photo"): canonical_name("Amazon Photo"),
        canonical_name("Computers"): canonical_name("Amazon Computers"),
        canonical_name("CS"): canonical_name("Coauthor CS"),
    }

    key = canonical_name(dataset_name)
    candidates = [key]
    if key in aliases:
        candidates.append(aliases[key])

    match = None
    for candidate in candidates:
        if candidate in lookup:
            match = lookup[candidate]
            break

    if match is None:
        choices = ", ".join(metrics.keys())
        raise KeyError(f"{dataset_name!r} not found in {metrics_path}. Available metrics: {choices}")

    metrics_key, data = match
    support = np.asarray(data["eigenvalues"], dtype=np.float64)
    cdf = np.asarray(data["cdf"], dtype=np.float64)
    if support.shape != cdf.shape:
        raise ValueError(f"{metrics_key} has mismatched eigenvalue/CDF lengths in {metrics_path}")

    order = np.argsort(support)
    support = support[order]
    cdf = cdf[order]
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, None))
    mass = np.diff(np.concatenate(([0.0], cdf)))
    mass = np.clip(mass, 0.0, None)
    total = mass.sum()
    if total <= 0:
        raise ValueError(f"{metrics_key} has zero SLP mass in {metrics_path}")

    mass = mass / total
    normalized_cdf = np.cumsum(mass)
    return metrics_key, support, mass, normalized_cdf


def cdf_on_grid(eigenvalues: np.ndarray, cdf: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(eigenvalues, grid, side="right") - 1
    values = np.zeros_like(grid, dtype=np.float64)
    valid = idx >= 0
    values[valid] = cdf[idx[valid]]
    return values


def run_dataset(
    dataset_name: str,
    adjacency: sp.csr_matrix,
    labels: np.ndarray,
    num_classes: int,
    epsilons: list[float],
    step_values: list[int],
    julia_bin: str,
    seed: int,
    metrics_path: Path,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, object]]]:
    full_edges = int(sp.triu(adjacency, k=1).nnz)
    step_values = valid_step_values(step_values, adjacency.shape[0])
    reference_steps = max(step_values)
    metrics_reference = load_metrics_reference(metrics_path, dataset_name)
    ground_truth_key, full_evals, full_mass, full_cdf = metrics_reference

    rows = []
    cdf_curves = []

    def add_profile_rows(
        variant: str,
        epsilon: float | None,
        sparsify_time_sec: float,
        graph_edges: int,
        profiles: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    ) -> None:
        reference_profile = profiles[reference_steps]

        for steps in step_values:
            evals, mass, cdf, slp_time_sec = profiles[steps]
            pipeline_time_sec = sparsify_time_sec + slp_time_sec
            steps_wasserstein_to_reference = wasserstein_distance(
                reference_profile[0],
                evals,
                u_weights=reference_profile[1],
                v_weights=mass,
            )
            slp_wasserstein = wasserstein_distance(
                full_evals,
                evals,
                u_weights=full_mass,
                v_weights=mass,
            )
            approx_cdf_on_reference_grid = cdf_on_grid(evals, cdf, full_evals)
            cdf_delta = approx_cdf_on_reference_grid - full_cdf
            slp_cdf_l1 = float(np.mean(np.abs(cdf_delta)))
            slp_cdf_l2 = float(np.sqrt(np.mean(cdf_delta**2)))

            cdf_curves.append(
                {
                    "dataset": dataset_name,
                    "ground_truth_key": ground_truth_key,
                    "variant": variant,
                    "epsilon": np.nan if epsilon is None else epsilon,
                    "steps": steps,
                    "true_eigenvalues": full_evals,
                    "true_cdf": full_cdf,
                    "approx_eigenvalues": evals,
                    "approx_cdf": cdf,
                }
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant,
                    "epsilon": np.nan if epsilon is None else epsilon,
                    "steps": steps,
                    "reference_steps": reference_steps,
                    "seed": seed,
                    "num_nodes": adjacency.shape[0],
                    "num_classes": num_classes,
                    "full_edges": full_edges,
                    "graph_edges": graph_edges,
                    "edge_fraction": graph_edges / full_edges if full_edges else np.nan,
                    "ground_truth_key": ground_truth_key,
                    "sparsify_time_sec": sparsify_time_sec,
                    "slp_time_sec": slp_time_sec,
                    "pipeline_time_sec": pipeline_time_sec,
                    "steps_wasserstein_to_reference": float(steps_wasserstein_to_reference),
                    "slp_wasserstein": float(slp_wasserstein),
                    "slp_cdf_l1": slp_cdf_l1,
                    "slp_cdf_l2": slp_cdf_l2,
                }
            )

    original_profiles = {
        steps: compute_lanczos_slp(adjacency, labels, num_classes, steps)
        for steps in step_values
    }
    add_profile_rows(
        "original",
        None,
        0.0,
        full_edges,
        original_profiles,
    )

    for epsilon, sparsify_time_sec, sparse_adjacency in run_laplacians_sparsify(
        adjacency, epsilons, julia_bin, seed
    ):
        sparse_edges = int(sp.triu(sparse_adjacency, k=1).nnz)
        sparse_profiles = {
            steps: compute_lanczos_slp(sparse_adjacency, labels, num_classes, steps)
            for steps in step_values
        }
        add_profile_rows(
            "sparsified",
            epsilon,
            sparsify_time_sec,
            sparse_edges,
            sparse_profiles,
        )

    return rows, cdf_curves


def plot_tradeoff(results: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = results[results["variant"] == "sparsified"].copy()
    if results.empty:
        return

    datasets = list(results["dataset"].drop_duplicates())
    y_column = "slp_wasserstein"
    y_label = "Wasserstein to true SLP-CDF"

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.0 * len(datasets), 4.0),
        squeeze=False,
        sharey=False,
        constrained_layout=True,
    )

    scatter = None
    for ax, dataset_name in zip(axes.ravel(), datasets):
        subset = results[results["dataset"] == dataset_name].sort_values(["epsilon", "steps"])
        subset = subset[subset[y_column].notna()]
        sizes = 30 + 120 * subset["edge_fraction"].fillna(0.0)
        scatter = ax.scatter(
            subset["pipeline_time_sec"],
            subset[y_column],
            c=subset["epsilon"],
            s=sizes,
            cmap="viridis",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"eps={row['epsilon']:.2g}, s={int(row['steps'])}",
                (row["pipeline_time_sec"], row[y_column]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_title(dataset_name)
        ax.set_xlabel("sparsify + SLP time (sec)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("epsilon")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cdf_comparison(cdf_curves: list[dict[str, object]], path: Path) -> None:
    if not cdf_curves:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = []
    for curve in cdf_curves:
        dataset = curve["dataset"]
        if dataset not in datasets:
            datasets.append(dataset)

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.5 * len(datasets), 4.2),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )

    for ax, dataset_name in zip(axes.ravel(), datasets):
        curves = [curve for curve in cdf_curves if curve["dataset"] == dataset_name]
        first = curves[0]
        ax.step(
            first["true_eigenvalues"],
            first["true_cdf"],
            where="post",
            color="black",
            linewidth=2.0,
            label=f"true ({first['ground_truth_key']})",
        )

        eps_values = sorted(
            {float(curve["epsilon"]) for curve in curves if not np.isnan(float(curve["epsilon"]))}
        )
        color_lookup = {
            epsilon: plt.cm.viridis(i / max(1, len(eps_values) - 1))
            for i, epsilon in enumerate(eps_values)
        }
        for curve in curves:
            variant = str(curve["variant"])
            epsilon = float(curve["epsilon"])
            steps = int(curve["steps"])
            if variant == "original":
                color = "tab:orange"
                label = f"original, steps={steps}"
            else:
                color = color_lookup[epsilon]
                label = f"eps={epsilon:.2g}, steps={steps}"

            ax.step(
                curve["approx_eigenvalues"],
                curve["approx_cdf"],
                where="post",
                color=color,
                linewidth=1.2,
                alpha=0.78,
                label=label,
            )

        ax.set_title(dataset_name)
        ax.set_xlabel("lambda")
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        if ax is axes.ravel()[0]:
            ax.set_ylabel("SLP CDF")
        if len(curves) <= 12:
            ax.legend(fontsize=7, loc="lower right")

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep epsilon for Laplacians.jl sparsification and compare approximate "
            "Lanczos SLP-CDFs with Wasserstein distance."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Datasets to run. Defaults to datasets returned by "
            "spatial.datasets.build_datasets() that also have metrics.json ground truth."
        ),
    )
    parser.add_argument("--eps", "--epsilons", dest="epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    parser.add_argument(
        "--step-values",
        dest="step_values",
        nargs="+",
        type=int,
        default=DEFAULT_STEP_VALUES,
        help="Lanczos step sweep for approximate SLP. Defaults to 16 32 64.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--julia-bin", default="julia")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Ground-truth full-graph SLP-CDF JSON. Defaults to analysis/metrics.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "analysis" / "approximate_slp" / f"sparsify_tradeoff_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
    )
    parser.add_argument("--plot-out", type=Path, default=None)
    parser.add_argument(
        "--cdf-plot-out",
        type=Path,
        default=None,
        help="Output PNG for approximate CDFs overlaid with the metrics.json true CDF.",
    )
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()
    invalid = [epsilon for epsilon in args.epsilons if epsilon <= 0.0 or epsilon >= 1.0]
    if invalid:
        parser.error(f"all epsilons must be in (0, 1); got {invalid}")
    if any(steps <= 0 for steps in args.step_values):
        parser.error("all Lanczos step values must be positive")
    metrics_path = args.metrics_json if args.metrics_json.is_absolute() else REPO_ROOT / args.metrics_json
    if not metrics_path.exists():
        parser.error(f"--metrics-json does not exist: {metrics_path}")
    return args


def main() -> None:
    args = parse_args()

    if shutil.which(args.julia_bin) is None:
        raise SystemExit(
            f"Julia executable not found: {args.julia_bin}. Install Julia and Laplacians.jl, "
            "or pass --julia-bin /path/to/julia."
        )

    rows = []
    cdf_curves = []
    for dataset_name, dataset in select_datasets(args.datasets):
        adjacency, labels, num_classes = graph_from_dataset(dataset)
        full_edges = int(sp.triu(adjacency, k=1).nnz)
        step_values = valid_step_values(args.step_values, adjacency.shape[0])
        print(
            f"{dataset_name}: n={adjacency.shape[0]}, edges={full_edges}, "
            f"eps={args.epsilons}, steps={step_values}"
        )
        try:
            dataset_rows, dataset_cdf_curves = run_dataset(
                dataset_name,
                adjacency,
                labels,
                num_classes,
                args.epsilons,
                step_values,
                args.julia_bin,
                args.seed,
                args.metrics_json if args.metrics_json.is_absolute() else REPO_ROOT / args.metrics_json,
            )
        except KeyError as error:
            if args.datasets is not None:
                raise
            print(f"Skipping {dataset_name}: {error}")
            continue
        rows.extend(dataset_rows)
        cdf_curves.extend(dataset_cdf_curves)

    if not rows:
        raise SystemExit("No results were produced.")

    results = pd.DataFrame(rows)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    plot_out = None
    cdf_plot_out = None
    if not args.no_plot:
        plot_out = args.plot_out or out.with_suffix(".png")
        plot_out = plot_out if plot_out.is_absolute() else REPO_ROOT / plot_out
        plot_out.parent.mkdir(parents=True, exist_ok=True)
        plot_tradeoff(results, plot_out)

        if cdf_curves:
            cdf_plot_out = args.cdf_plot_out or out.with_name(f"{out.stem}_cdf.png")
            cdf_plot_out = cdf_plot_out if cdf_plot_out.is_absolute() else REPO_ROOT / cdf_plot_out
            cdf_plot_out.parent.mkdir(parents=True, exist_ok=True)
            plot_cdf_comparison(cdf_curves, cdf_plot_out)

    print("\nResults")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    print(f"\nWrote {out}")
    if plot_out is not None:
        print(f"Wrote {plot_out}")
    if cdf_plot_out is not None:
        print(f"Wrote {cdf_plot_out}")


if __name__ == "__main__":
    main()
