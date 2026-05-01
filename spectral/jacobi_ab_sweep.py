import argparse
import csv
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from datasets import build_datasets


class BatchedJacobiConv(nn.Module):
    """Train many paper-style JacobiConv configs in one lockstep model."""

    def __init__(self, in_dim, num_classes, K, ab_pairs, dropout=0.5, dprate=0.0):
        super().__init__()
        self.K = K
        self.B = len(ab_pairs)
        self.C = num_classes
        self.dropout = dropout
        self.dprate = dprate

        a = torch.tensor([p[0] for p in ab_pairs], dtype=torch.float)
        b = torch.tensor([p[1] for p in ab_pairs], dtype=torch.float)
        if torch.any(a <= -1.0) or torch.any(b <= -1.0):
            raise ValueError("Jacobi a and b must be greater than -1.")
        self.register_buffer("a", a)
        self.register_buffer("b", b)

        # Paper JacobiConv: X_hat = XW + b, then one filter per output channel.
        self.W = nn.Parameter(torch.empty(self.B, in_dim, num_classes))
        self.bias = nn.Parameter(torch.zeros(self.B, num_classes))
        nn.init.xavier_uniform_(self.W)

        self.alpha = nn.Parameter(torch.randn(self.B, K + 1, num_classes) * 0.1)

    def _linear(self, x):
        return torch.einsum("ni,bic->bnc", x, self.W) + self.bias.unsqueeze(1)

    def _spmm_batched(self, A_norm, h):
        B, N, C = h.shape
        h_flat = h.permute(1, 0, 2).reshape(N, B * C)
        out = torch.sparse.mm(A_norm, h_flat)
        return out.reshape(N, B, C).permute(1, 0, 2)

    def _jacobi_step(self, k, prevprev, prev, A_norm):
        a, b = self.a, self.b

        if k == 1:
            c0 = ((a - b) / 2).view(self.B, 1, 1)
            c1 = ((a + b + 2) / 2).view(self.B, 1, 1)
            return c0 * prev + c1 * self._spmm_batched(A_norm, prev)

        theta = ((2 * k + a + b) * (2 * k + a + b - 1)) / (2 * k * (k + a + b))
        theta_prime = (
            (2 * k + a + b - 1) * (a**2 - b**2)
        ) / (2 * k * (k + a + b) * (2 * k + a + b - 2))
        theta_double = (
            (k + a - 1) * (k + b - 1) * (2 * k + a + b)
        ) / (k * (k + a + b) * (2 * k + a + b - 2))

        theta = theta.view(self.B, 1, 1)
        theta_prime = theta_prime.view(self.B, 1, 1)
        theta_double = theta_double.view(self.B, 1, 1)

        return (
            theta * self._spmm_batched(A_norm, prev)
            + theta_prime * prev
            - theta_double * prevprev
        )

    def forward(self, x, A_norm):
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self._linear(x)
        h = F.dropout(h, p=self.dprate, training=self.training)

        p_prev = h
        z = self.alpha[:, 0].unsqueeze(1) * p_prev
        if self.K == 0:
            return z

        p_curr = self._jacobi_step(1, None, p_prev, A_norm)
        z = z + self.alpha[:, 1].unsqueeze(1) * p_curr
        for k in range(2, self.K + 1):
            p_next = self._jacobi_step(k, p_prev, p_curr, A_norm)
            z = z + self.alpha[:, k].unsqueeze(1) * p_next
            p_prev, p_curr = p_curr, p_next
        return z


def get_split_masks(data, split_idx=0):
    if hasattr(data, "train_mask") and data.train_mask is not None:
        def select_mask(mask):
            if mask.dim() == 2:
                return mask[:, split_idx % mask.size(1)]
            return mask

        return (
            select_mask(data.train_mask),
            select_mask(data.val_mask),
            select_mask(data.test_mask),
        )

    n = data.num_nodes
    perm = torch.randperm(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


def build_A_norm(data, device):
    edge_index, edge_weight = gcn_norm(
        data.edge_index,
        num_nodes=data.num_nodes,
        add_self_loops=False,
        dtype=torch.float,
    )
    return torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(data.num_nodes, data.num_nodes),
    ).coalesce().to(device)


def train_batched(model, x, y, A_norm, train_mask, val_mask, test_mask,
                  lr=0.01, weight_decay=5e-4, epochs=300, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    B = model.B
    best_val = torch.full((B,), -1.0)
    best_test = torch.zeros(B)
    stale = torch.zeros(B, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool)

    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        model.train()
        optimizer.zero_grad()
        logits = model(x, A_norm)
        loss_per = F.cross_entropy(
            logits[:, train_mask].reshape(-1, model.C),
            y[train_mask].repeat(B),
            reduction="none",
        ).reshape(B, -1).mean(dim=1)

        active = (~finished).to(loss_per.device)
        finite = torch.isfinite(loss_per)
        loss = loss_per[active & finite].sum()
        if loss.numel() == 0 or not torch.isfinite(loss):
            break

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, A_norm)
            pred = logits.argmax(dim=-1)
            val_acc = (pred[:, val_mask] == y[val_mask]).float().mean(dim=1).cpu()
            test_acc = (pred[:, test_mask] == y[test_mask]).float().mean(dim=1).cpu()

        improved = val_acc > best_val
        best_val = torch.where(improved, val_acc, best_val)
        best_test = torch.where(improved, test_acc, best_test)
        stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
        finished |= stale >= patience
        if finished.all():
            break

    return best_val.numpy(), best_test.numpy(), epochs_ran


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["all"],
                   help='Datasets to run, or "all" for the train_spectral dataset list.')
    p.add_argument("--a-min", type=float, default=-0.99)
    p.add_argument("--a-max", type=float, default=4.0)
    p.add_argument("--b-min", type=float, default=-0.99)
    p.add_argument("--b-max", type=float, default=4.0)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--K", nargs="+", type=int, default=[4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--dprate", type=float, default=0.0)
    p.add_argument("--max-batch", type=int, default=10000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="jacobi_ab_sweep")
    p.add_argument("--out", default="summary.csv",
                   help="Summary CSV filename written inside the timestamped output directory.")
    return p.parse_args()


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

    a_vals = np.arange(args.a_min, args.a_max + 1e-9, args.step)
    b_vals = np.arange(args.b_min, args.b_max + 1e-9, args.step)
    ab_pairs = [(float(a), float(b)) for a, b in product(a_vals, b_vals)]

    rows = []
    detail_rows = []
    print(f"device: {device}")
    print(f"grid: {len(a_vals)} x {len(b_vals)} = {len(ab_pairs)} configs")
    print(f"K values: {args.K}")
    print(f"output directory: {run_dir}")

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    all_datasets = build_datasets()
    if args.datasets == ["all"]:
        selected_datasets = all_datasets
    else:
        wanted = {name.lower() for name in args.datasets}
        selected_datasets = [
            (name, dataset) for name, dataset in all_datasets
            if name.lower() in wanted
        ]
        found = {name.lower() for name, _ in selected_datasets}
        missing = sorted(wanted - found)
        if missing:
            available = ", ".join(name for name, _ in all_datasets)
            raise ValueError(f"Unknown dataset(s): {', '.join(missing)}. Available: {available}")

    for ds_name, dataset in selected_datasets:
        data = dataset[0]
        A_norm = build_A_norm(data, device)
        x = data.x.to(device)
        y = data.y.to(device)

        print(f"\n=== {ds_name}: {data.num_nodes} nodes, {data.num_edges} edges ===")

        for K in args.K:
            print(f"  K={K}")
            per_pair = {pair: {"val": [], "test": []} for pair in ab_pairs}

            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                train_mask, val_mask, test_mask = get_split_masks(data, split_idx=seed % 10)
                train_mask = train_mask.to(device)
                val_mask = val_mask.to(device)
                test_mask = test_mask.to(device)

                t0 = time.time()
                for start in range(0, len(ab_pairs), args.max_batch):
                    chunk = ab_pairs[start:start + args.max_batch]
                    model = BatchedJacobiConv(
                        in_dim=dataset.num_features,
                        num_classes=dataset.num_classes,
                        K=K,
                        ab_pairs=chunk,
                        dropout=args.dropout,
                        dprate=args.dprate,
                    ).to(device)

                    val_accs, test_accs, epochs_ran = train_batched(
                        model,
                        x,
                        y,
                        A_norm,
                        train_mask,
                        val_mask,
                        test_mask,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        epochs=args.epochs,
                        patience=args.patience,
                    )
                    for pair, val_acc, test_acc in zip(chunk, val_accs, test_accs):
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
                        })

                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                print(f"    seed {seed}: {time.time() - t0:.1f}s")

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
                })

        write_csv(out_path, rows)
        write_csv(details_path, detail_rows)
        print(f"  wrote {len(rows)} summary rows to {out_path}")
        print(f"  wrote {len(detail_rows)} detail rows to {details_path}")


if __name__ == "__main__":
    main()
