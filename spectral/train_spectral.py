import argparse
import csv
import importlib
import json
from datetime import datetime
from pathlib import Path

from datasets import build_datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--harness", choices=["poly", "paper"], default="poly")
    p.add_argument("--models", nargs="+", default=["GPRGNN", "ChebGNN", "BernNet", "JacobiConv"])
    p.add_argument("--k", nargs="+", type=int, default=[4, 8, 10])
    p.add_argument("--hidden", nargs="+", type=int, default=[64])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--device", default=None)  # none, harness chooses cuda/cpu
    p.add_argument("--epochs", nargs="+", type=int, default=[300])
    p.add_argument("--patience", nargs="+", type=int, default=[20])
    p.add_argument("--lr", nargs="+", type=float, default=[1e-2])
    p.add_argument("--weight-decay", nargs="+", type=float, default=[5e-4])
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--dprate", type=float, default=0.0)
    p.add_argument("--out-dir", default="runs")
    return p.parse_args()


def load_train_sweep(harness):
    module_name = {
        "poly": "poly_harness",
        "paper": "paper_faithful_harness",
    }[harness]
    return importlib.import_module(module_name).train_sweep


def write_outputs(results, args, run_dir):
    curves_dir = run_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset_name, configs in results.items():
        for config_key, (accs, val_histories) in configs.items():
            model_name, K, hidden_dim, lr, weight_decay, epochs, patience = config_key
            for seed, (acc, history) in enumerate(zip(accs, val_histories)):
                row_id = (
                    f"{dataset_name}_{model_name}_K{K}_H{hidden_dim}"
                    f"_lr{lr:g}_wd{weight_decay:g}_e{epochs}_p{patience}_s{seed}"
                )

                rows.append({
                    "id": row_id,
                    "dataset": dataset_name,
                    "model": model_name,
                    "K": K,
                    "hidden": hidden_dim,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "patience": patience,
                    "seed": seed,
                    "test_acc": f"{acc:.4f}",
                })

                with open(curves_dir / f"{row_id}.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "val_acc"])
                    for epoch, val_acc in enumerate(history):
                        writer.writerow([epoch, f"{val_acc:.4f}"])

    fieldnames = [
        "id", "dataset", "model", "K", "hidden",
        "lr", "weight_decay", "epochs", "patience",
        "seed", "test_acc",
    ]
    with open(run_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    return rows


def main():
    args = parse_args()
    train_sweep = load_train_sweep(args.harness)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) / f"{args.harness}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets()

    results = train_sweep(
        datasets,
        models=args.models,
        K=args.k,
        hidden_dims=args.hidden,
        n_runs=args.runs,
        dropout=args.dropout,
        dprate=args.dprate,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )

    write_outputs(results, args, run_dir)

    print(f"Wrote run to {run_dir}")


if __name__ == "__main__":
    main()
