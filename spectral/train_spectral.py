from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPATIAL_DIR = REPO_ROOT / "spatial"
DEFAULT_DATA_ROOT = REPO_ROOT / "graph_data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"


def parse_float_pair(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Jacobi pairs must look like alpha,beta")
    return float(parts[0]), float(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-grounded polynomial spectral GNN sweeps on the spatial dataset pool."
    )
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names or 'all'.")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Dataset names or 'all'.")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--orders", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--jacobi-pairs", nargs="+", type=parse_float_pair, default=[(0.0, 0.0)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny Cora-only sanity check.")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return requested


def load_datasets(data_root: Path):
    sys.path.insert(0, str(SPATIAL_DIR))
    cwd = os.getcwd()
    try:
        os.chdir(SPATIAL_DIR)
        import datasets as spatial_datasets
    finally:
        os.chdir(cwd)

    spatial_datasets.data_dir = str(data_root.resolve())
    return spatial_datasets.build_datasets()


def select_datasets(datasets, requested: list[str]):
    if requested == ["all"]:
        return datasets
    requested_set = set(requested)
    selected = [(name, dataset) for name, dataset in datasets if name in requested_set]
    missing = sorted(requested_set - {name for name, _ in selected})
    if missing:
        raise ValueError(f"Unknown dataset names: {missing}")
    return selected


def expand_models(requested: list[str], supported: tuple[str, ...]) -> list[str]:
    if requested == ["all"]:
        return list(supported)
    unknown = sorted(set(requested) - set(supported))
    if unknown:
        raise ValueError(f"Unknown spectral model names: {unknown}. Supported: {supported}")
    return requested


def format_optional(value) -> str:
    return "na" if value is None else str(value)


def row_id(dataset_name: str, config, seed: int) -> str:
    alpha = format_optional(config.alpha)
    ja = format_optional(config.jacobi_alpha)
    jb = format_optional(config.jacobi_beta)
    return (
        f"{dataset_name}_{config.model}_K{config.order}_H{config.hidden}"
        f"_a{alpha}_ja{ja}_jb{jb}_s{seed}"
    )


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.datasets = ["Cora"]
    args.hidden_dims = [16]
    args.orders = [2]
    args.alphas = [0.1]
    args.jacobi_pairs = [(0.0, 0.0)]
    args.seeds = [0]
    args.max_epochs = min(args.max_epochs, 2)
    args.patience = min(args.patience, 1)


def write_config(run_dir: Path, args: argparse.Namespace, models: list[str], datasets: list[str], device: str) -> None:
    payload = {
        "models": models,
        "datasets": datasets,
        "hidden_dims": args.hidden_dims,
        "orders": args.orders,
        "alphas": args.alphas,
        "jacobi_pairs": args.jacobi_pairs,
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": device,
        "data_root": str(args.data_root),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    apply_smoke_overrides(args)

    from poly_harness import SUPPORTED_MODELS, expand_configs, run_single

    models = expand_models(args.models, SUPPORTED_MODELS)
    device = resolve_device(args.device)
    datasets = select_datasets(load_datasets(args.data_root), args.datasets)
    configs = expand_configs(models, args.hidden_dims, args.orders, args.alphas, args.jacobi_pairs)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or timestamp
    run_dir = args.output_root / run_name
    curves_dir = run_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir, args, models, [name for name, _ in datasets], device)

    fieldnames = [
        "id",
        "dataset",
        "model",
        "basis",
        "order",
        "hidden",
        "alpha",
        "jacobi_alpha",
        "jacobi_beta",
        "seed",
        "test_acc",
        "best_val",
        "epochs",
    ]

    total = len(datasets) * len(configs) * len(args.seeds)
    done = 0
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_name, dataset in datasets:
            for config in configs:
                for seed in args.seeds:
                    acc, best_val, history = run_single(
                        dataset=dataset,
                        config=config,
                        seed=seed,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        dropout=args.dropout,
                        device=device,
                    )
                    rid = row_id(dataset_name, config, seed)
                    row = {
                        "id": rid,
                        "dataset": dataset_name,
                        "model": config.model,
                        "basis": config.basis,
                        "order": config.order,
                        "hidden": config.hidden,
                        "alpha": format_optional(config.alpha),
                        "jacobi_alpha": format_optional(config.jacobi_alpha),
                        "jacobi_beta": format_optional(config.jacobi_beta),
                        "seed": seed,
                        "test_acc": f"{acc:.4f}",
                        "best_val": f"{best_val:.4f}",
                        "epochs": len(history),
                    }
                    writer.writerow(row)
                    f.flush()

                    with open(curves_dir / f"{rid}.csv", "w", newline="") as curve_f:
                        curve_writer = csv.writer(curve_f)
                        curve_writer.writerow(["epoch", "val_acc"])
                        for epoch, val_acc in enumerate(history):
                            curve_writer.writerow([epoch, f"{val_acc:.4f}"])

                    done += 1
                    print(
                        f"[{done}/{total}] {dataset_name} {config.model} "
                        f"K={config.order} H={config.hidden} seed={seed} "
                        f"test={acc:.4f} best_val={best_val:.4f}",
                        flush=True,
                    )

    print(f"Wrote spectral run to {run_dir}")


if __name__ == "__main__":
    main()
