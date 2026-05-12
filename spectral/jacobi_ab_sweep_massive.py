import argparse
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets import build_datasets
from jacobi_ab_sweep import (
    BatchedJacobiConv,
    DEFAULT_MASK_DIR,
    Tee,
    build_A_norm,
    get_split_masks,
    load_spatial_masks,
    write_csv,
)


def masked_cross_entropy(logits, y, mask):
    tasks, num_nodes, num_classes = logits.shape
    loss = F.cross_entropy(
        logits.reshape(tasks * num_nodes, num_classes),
        y.repeat(tasks),
        reduction="none",
    ).reshape(tasks, num_nodes)
    denom = mask.sum(dim=1).clamp(min=1).float()
    return (loss * mask.float()).sum(dim=1) / denom


def masked_accuracy(pred, y, mask):
    correct = (pred == y.view(1, -1)) & mask
    denom = mask.sum(dim=1).clamp(min=1).float()
    return correct.float().sum(dim=1) / denom


def train_task_batch(
    model,
    x,
    y,
    A_norm,
    train_masks,
    val_masks,
    test_masks,
    lr=0.01,
    weight_decay=5e-4,
    epochs=300,
    patience=50,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tasks = model.B
    best_val = torch.full((tasks,), -1.0, device=x.device)
    best_test = torch.zeros(tasks, device=x.device)
    stale = torch.zeros(tasks, dtype=torch.long, device=x.device)
    finished = torch.zeros(tasks, dtype=torch.bool, device=x.device)

    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        model.train()
        optimizer.zero_grad()
        logits = model(x, A_norm)
        loss_per = masked_cross_entropy(logits, y, train_masks)

        finite = torch.isfinite(loss_per)
        active = (~finished) & finite
        if not active.any():
            break

        loss = loss_per[active].sum()
        if not torch.isfinite(loss):
            break

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, A_norm)
            pred = logits.argmax(dim=-1)
            val_acc = masked_accuracy(pred, y, val_masks)
            test_acc = masked_accuracy(pred, y, test_masks)

        improved = val_acc > best_val
        best_val = torch.where(improved, val_acc, best_val)
        best_test = torch.where(improved, test_acc, best_test)
        stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
        finished |= stale >= patience
        if finished.all():
            break

    return best_val.cpu().numpy(), best_test.cpu().numpy(), epochs_ran


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help='Datasets to run, or "all" for the train_spectral dataset list.',
    )
    p.add_argument("--a-min", type=float, default=-0.99)
    p.add_argument("--a-max", type=float, default=4.0)
    p.add_argument("--b-min", type=float, default=-0.99)
    p.add_argument("--b-max", type=float, default=4.0)
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--K", nargs="+", type=int, default=[4, 10])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--dprate", type=float, default=0.0)
    p.add_argument(
        "--max-task-batch",
        "--max-batch",
        dest="max_task_batch",
        type=int,
        default=1000,
        help="Max seed/config tasks trained together in one vectorized batch.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="jacobi_ab_sweep_massive")
    p.add_argument(
        "--out",
        default="summary.csv",
        help="Summary CSV filename written inside the timestamped output directory.",
    )
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--train-r", type=float, default=0.8)
    p.add_argument("--val-r", type=float, default=0.1)
    p.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    return p.parse_args()


def select_datasets(requested):
    all_datasets = build_datasets()
    if requested == ["all"]:
        return all_datasets

    wanted = {name.lower() for name in requested}
    selected = [
        (name, dataset)
        for name, dataset in all_datasets
        if name.lower() in wanted
    ]
    found = {name.lower() for name, _ in selected}
    missing = sorted(wanted - found)
    if missing:
        available = ", ".join(name for name, _ in all_datasets)
        raise ValueError(f"Unknown dataset(s): {', '.join(missing)}. Available: {available}")
    return selected


def stack_masks(seed_masks, seeds, index, device):
    return torch.stack([seed_masks[seed][index] for seed in seeds]).to(device)


def main():
    args = parse_args()
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    out_name = Path(args.out).name
    out_path = run_dir / out_name
    details_path = out_path.with_name(f"{out_path.stem}_details{out_path.suffix}")
    config_path = out_path.with_name(f"{out_path.stem}_config.json")
    log_path = run_dir / "progress.log"
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    a_vals = np.arange(args.a_min, args.a_max + 1e-9, args.step)
    b_vals = np.arange(args.b_min, args.b_max + 1e-9, args.step)
    ab_pairs = [(float(a), float(b)) for a, b in product(a_vals, b_vals)]
    selected_datasets = select_datasets(args.datasets)

    with open(config_path, "w") as f:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, f, indent=2)

    print(f"device: {device}", flush=True)
    print(f"grid: {len(a_vals)} x {len(b_vals)} = {len(ab_pairs)} configs", flush=True)
    print(f"K values: {args.K}", flush=True)
    print(f"seeds: {args.seeds}", flush=True)
    print(f"max task batch: {args.max_task_batch}", flush=True)
    print(f"spatial masks: {args.mask_dir}", flush=True)
    print(f"output directory: {run_dir}", flush=True)
    print(f"progress log: {log_path}", flush=True)

    rows = []
    detail_rows = []

    for dataset_idx, (ds_name, dataset) in enumerate(selected_datasets, start=1):
        data = dataset[0]
        A_norm = build_A_norm(data, device)
        x = data.x.to(device)
        y = data.y.to(device)

        print(
            f"\n=== [{dataset_idx}/{len(selected_datasets)}] {ds_name}: "
            f"{data.num_nodes} nodes, {data.num_edges} edges ===",
            flush=True,
        )

        spatial_masks = load_spatial_masks(ds_name, args.mask_dir)
        seed_masks = {seed: spatial_masks for seed in args.seeds}

        for K in args.K:
            print(f"  K={K}", flush=True)
            per_pair = {pair: {"val": [], "test": []} for pair in ab_pairs}
            tasks = [(pair, seed) for pair in ab_pairs for seed in args.seeds]
            num_chunks = (len(tasks) + args.max_task_batch - 1) // args.max_task_batch

            for chunk_id, start in enumerate(range(0, len(tasks), args.max_task_batch), start=1):
                chunk = tasks[start:start + args.max_task_batch]
                chunk_pairs = [pair for pair, _ in chunk]
                chunk_seeds = [seed for _, seed in chunk]

                train_masks = stack_masks(seed_masks, chunk_seeds, 0, device)
                val_masks = stack_masks(seed_masks, chunk_seeds, 1, device)
                test_masks = stack_masks(seed_masks, chunk_seeds, 2, device)

                torch.manual_seed(args.init_seed + dataset_idx * 1_000_003 + K * 10_007 + start)
                t0 = time.time()
                model = BatchedJacobiConv(
                    in_dim=dataset.num_features,
                    num_classes=dataset.num_classes,
                    K=K,
                    ab_pairs=chunk_pairs,
                    dropout=args.dropout,
                    dprate=args.dprate,
                ).to(device)

                val_accs, test_accs, epochs_ran = train_task_batch(
                    model,
                    x,
                    y,
                    A_norm,
                    train_masks,
                    val_masks,
                    test_masks,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    patience=args.patience,
                )

                for (pair, seed), val_acc, test_acc in zip(chunk, val_accs, test_accs):
                    per_pair[pair]["val"].append(float(val_acc))
                    per_pair[pair]["test"].append(float(test_acc))
                    detail_rows.append({
                        "dataset": ds_name,
                        "num_nodes": data.num_nodes,
                        "num_edges": data.num_edges,
                        "num_features": dataset.num_features,
                        "num_classes": dataset.num_classes,
                        "K": K,
                        "a": pair[0],
                        "b": pair[1],
                        "seed": seed,
                        "best_val_acc": float(val_acc),
                        "best_test_acc": float(test_acc),
                        "epochs_ran": epochs_ran,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "dropout": args.dropout,
                        "dprate": args.dprate,
                        "patience": args.patience,
                        "max_epochs": args.epochs,
                        "parallel_mode": "seed_config_batched",
                    })

                del model, train_masks, val_masks, test_masks
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                print(
                    f"    chunk {chunk_id}/{num_chunks}: "
                    f"{len(chunk)} seed/config tasks, {time.time() - t0:.1f}s, "
                    f"epochs_ran={epochs_ran}",
                    flush=True,
                )

            for (a, b), accs in per_pair.items():
                rows.append({
                    "dataset": ds_name,
                    "num_nodes": data.num_nodes,
                    "num_edges": data.num_edges,
                    "num_features": dataset.num_features,
                    "num_classes": dataset.num_classes,
                    "a": a,
                    "b": b,
                    "K": K,
                    "mean_val_acc": float(np.mean(accs["val"])),
                    "std_val_acc": float(np.std(accs["val"])),
                    "mean_test_acc": float(np.mean(accs["test"])),
                    "std_test_acc": float(np.std(accs["test"])),
                    "n_seeds": len(accs["test"]),
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "dprate": args.dprate,
                    "patience": args.patience,
                    "max_epochs": args.epochs,
                    "parallel_mode": "seed_config_batched",
                })

            write_csv(out_path, rows)
            write_csv(details_path, detail_rows)
            print(f"  wrote {len(rows)} summary rows to {out_path}", flush=True)
            print(f"  wrote {len(detail_rows)} detail rows to {details_path}", flush=True)


if __name__ == "__main__":
    main()
