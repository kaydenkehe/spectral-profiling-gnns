import argparse
import csv
import json
import math
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian

from datasets import build_datasets
from jacobi_ab_sweep import DEFAULT_MASK_DIR, Tee, load_spatial_masks, write_csv


def values(x):
    return x if isinstance(x, (list, tuple)) else (x,)


def init_xavier_slices_(tensor):
    with torch.no_grad():
        if tensor.dim() == 3:
            for i in range(tensor.size(0)):
                nn.init.xavier_uniform_(tensor[i])
        elif tensor.dim() == 4:
            for i in range(tensor.size(0)):
                for j in range(tensor.size(1)):
                    nn.init.xavier_uniform_(tensor[i, j])
        else:
            nn.init.xavier_uniform_(tensor)


def spmm_batched(A, h):
    tasks, num_nodes, channels = h.shape
    h_flat = h.permute(1, 0, 2).reshape(num_nodes, tasks * channels)
    out = torch.sparse.mm(A, h_flat)
    return out.reshape(num_nodes, tasks, channels).permute(1, 0, 2)


class BatchedMLP(nn.Module):
    def __init__(self, tasks, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(tasks, in_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(tasks, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(tasks, hidden_dim, out_dim))
        self.b2 = nn.Parameter(torch.zeros(tasks, out_dim))
        init_xavier_slices_(self.W1)
        init_xavier_slices_(self.W2)

    def forward(self, x, dropout, training):
        tasks = self.W1.size(0)
        h = x.unsqueeze(0).expand(tasks, -1, -1)
        h = F.dropout(h, p=dropout, training=training)
        h = torch.einsum("tni,tih->tnh", h, self.W1) + self.b1.unsqueeze(1)
        h = F.relu(h)
        h = F.dropout(h, p=dropout, training=training)
        return torch.einsum("tnh,thc->tnc", h, self.W2) + self.b2.unsqueeze(1)


class BatchedGPRGNN(nn.Module):
    operator = "A_hat"

    def __init__(self, tasks, in_dim, hidden_dim, num_classes, k_val,
                 dropout=0.5, dprate=0.0):
        super().__init__()
        self.K = k_val
        self.dropout = dropout
        self.dprate = dprate
        self.encoder = BatchedMLP(tasks, in_dim, hidden_dim, num_classes)

        alpha = 0.1
        gammas = alpha * (1 - alpha) ** torch.arange(self.K + 1, dtype=torch.float)
        gammas[-1] = (1 - alpha) ** self.K
        self.gamma = nn.Parameter(gammas.unsqueeze(0).repeat(tasks, 1))

    def forward(self, x, A_hat):
        h = self.encoder(x, self.dropout, self.training)
        h = F.dropout(h, p=self.dprate, training=self.training)
        z = self.gamma[:, 0].view(-1, 1, 1) * h
        for k in range(self.K):
            h = spmm_batched(A_hat, h)
            z = z + self.gamma[:, k + 1].view(-1, 1, 1) * h
        return z


class BatchedBernNet(nn.Module):
    operator = "L_sym"

    def __init__(self, tasks, in_dim, hidden_dim, num_classes, k_val,
                 dropout=0.5, dprate=0.0):
        super().__init__()
        self.K = k_val
        self.dropout = dropout
        self.dprate = dprate
        self.encoder = BatchedMLP(tasks, in_dim, hidden_dim, num_classes)
        self.theta = nn.Parameter(torch.ones(tasks, self.K + 1))
        self.register_buffer(
            "binom",
            torch.tensor([math.comb(self.K, k) for k in range(self.K + 1)], dtype=torch.float),
        )

    def forward(self, x, L_sym):
        h = self.encoder(x, self.dropout, self.training)
        h = F.dropout(h, p=self.dprate, training=self.training)

        l_powers = [h]
        for _ in range(self.K):
            l_powers.append(spmm_batched(L_sym, l_powers[-1]))

        theta = F.relu(self.theta)
        z = torch.zeros_like(h)
        for k in range(self.K + 1):
            term = l_powers[k]
            for _ in range(self.K - k):
                term = 2 * term - spmm_batched(L_sym, term)
            scale = theta[:, k].view(-1, 1, 1) * self.binom[k] / (2 ** self.K)
            z = z + scale * term
        return z


class BatchedChebLayer(nn.Module):
    def __init__(self, tasks, in_dim, out_dim, k_val):
        super().__init__()
        self.K = k_val
        self.weight = nn.Parameter(torch.empty(tasks, self.K, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(tasks, out_dim))
        init_xavier_slices_(self.weight)

    def _linear(self, x, k):
        return torch.einsum("tni,tio->tno", x, self.weight[:, k])

    def forward(self, x, L_cheb):
        tx_0 = x
        tx_1 = x
        out = self._linear(tx_0, 0)

        if self.K > 1:
            tx_1 = spmm_batched(L_cheb, x)
            out = out + self._linear(tx_1, 1)

        for k in range(2, self.K):
            tx_2 = 2.0 * spmm_batched(L_cheb, tx_1) - tx_0
            out = out + self._linear(tx_2, k)
            tx_0, tx_1 = tx_1, tx_2

        return out + self.bias.unsqueeze(1)


class BatchedChebNet(nn.Module):
    operator = "L_cheb"

    def __init__(self, tasks, in_dim, hidden_dim, num_classes, k_val,
                 dropout=0.5, dprate=0.0):
        super().__init__()
        self.dropout = dropout
        self.dprate = dprate
        self.conv1 = BatchedChebLayer(tasks, in_dim, hidden_dim, k_val)
        self.conv2 = BatchedChebLayer(tasks, hidden_dim, num_classes, k_val)

    def forward(self, x, L_cheb):
        tasks = self.conv1.weight.size(0)
        x = x.unsqueeze(0).expand(tasks, -1, -1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, L_cheb)
        x = F.relu(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        return self.conv2(x, L_cheb)


class BatchedJacobiConv(nn.Module):
    operator = "A_norm"

    def __init__(self, tasks, in_dim, hidden_dim, num_classes, k_val,
                 a=1.0, b=1.0, dropout=0.5, dprate=0.0):
        super().__init__()
        self.K = k_val
        self.a = a
        self.b = b
        self.dropout = dropout
        self.dprate = dprate
        self.W = nn.Parameter(torch.empty(tasks, in_dim, num_classes))
        self.bias = nn.Parameter(torch.zeros(tasks, num_classes))
        init_xavier_slices_(self.W)
        self.alpha = nn.Parameter(torch.randn(tasks, self.K + 1, num_classes) * 0.1)

    def _step(self, k, prevprev, prev, A_norm):
        a, b = self.a, self.b
        if k == 1:
            return ((a - b) / 2) * prev + ((a + b + 2) / 2) * spmm_batched(A_norm, prev)

        theta = ((2 * k + a + b) * (2 * k + a + b - 1)) / (2 * k * (k + a + b))
        theta_prime = (
            (2 * k + a + b - 1) * (a**2 - b**2)
        ) / (2 * k * (k + a + b) * (2 * k + a + b - 2))
        theta_double = (
            (k + a - 1) * (k + b - 1) * (2 * k + a + b)
        ) / (k * (k + a + b) * (2 * k + a + b - 2))
        return theta * spmm_batched(A_norm, prev) + theta_prime * prev - theta_double * prevprev

    def forward(self, x, A_norm):
        tasks = self.W.size(0)
        h = x.unsqueeze(0).expand(tasks, -1, -1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.einsum("tni,tic->tnc", h, self.W) + self.bias.unsqueeze(1)
        h = F.dropout(h, p=self.dprate, training=self.training)

        p_prev = h
        z = self.alpha[:, 0].unsqueeze(1) * p_prev
        if self.K == 0:
            return z

        p_curr = self._step(1, None, p_prev, A_norm)
        z = z + self.alpha[:, 1].unsqueeze(1) * p_curr
        for k in range(2, self.K + 1):
            p_next = self._step(k, p_prev, p_curr, A_norm)
            z = z + self.alpha[:, k].unsqueeze(1) * p_next
            p_prev, p_curr = p_curr, p_next
        return z


BATCHED_MODELS = {
    "GPRGNN": BatchedGPRGNN,
    "ChebGNN": BatchedChebNet,
    "BernNet": BatchedBernNet,
    "JacobiConv": BatchedJacobiConv,
}
MODELS = BATCHED_MODELS


class TaskAdam:
    def __init__(self, params, lr_values, wd_values, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = torch.as_tensor(lr_values, dtype=torch.float)
        self.wd = torch.as_tensor(wd_values, dtype=torch.float)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, active):
        self.t += 1
        active = active.to(dtype=torch.float)
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            shape = (p.size(0),) + (1,) * (p.dim() - 1)
            active_view = active.view(shape)
            lr_view = self.lr.to(p.device).view(shape)
            wd_view = self.wd.to(p.device).view(shape)

            grad = (p.grad + wd_view * p.data) * active_view
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            update = (m_hat / (v_hat.sqrt() + self.eps)) * active_view
            p.data.add_(lr_view * update, alpha=-1.0)


def get_splits(data, seed, train_r=0.8, val_r=0.1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = data.num_nodes
    perm = torch.randperm(n)
    n_tr = int(train_r * n)
    n_val = int(val_r * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr:n_tr + n_val]] = True
    test_mask[perm[n_tr + n_val:]] = True
    return train_mask, val_mask, test_mask


def make_operators(data, device):
    n = data.num_nodes
    ei_hat, ew_hat = gcn_norm(data.edge_index, num_nodes=n, add_self_loops=True, dtype=torch.float)
    A_hat = torch.sparse_coo_tensor(ei_hat, ew_hat, (n, n)).coalesce().to(device)

    ei_norm, ew_norm = gcn_norm(data.edge_index, num_nodes=n, add_self_loops=False, dtype=torch.float)
    A_norm = torch.sparse_coo_tensor(ei_norm, ew_norm, (n, n)).coalesce().to(device)

    ei_lap, ew_lap = get_laplacian(data.edge_index, normalization="sym", num_nodes=n)
    L_sym = torch.sparse_coo_tensor(ei_lap, ew_lap.float(), (n, n)).coalesce().to(device)

    ew_cheb = ew_lap.float().clone()
    lambda_max = 2.0 * ew_cheb.max()
    ew_cheb = (2.0 * ew_cheb) / lambda_max
    ew_cheb.masked_fill_(ew_cheb == float("inf"), 0)
    cheb_loop_mask = ei_lap[0] == ei_lap[1]
    ew_cheb[cheb_loop_mask] -= 1.0
    L_cheb = torch.sparse_coo_tensor(ei_lap, ew_cheb, (n, n)).coalesce().to(device)

    self_loop_mask = ei_lap[0] == ei_lap[1]
    ew_tilde = ew_lap.float().clone()
    ew_tilde[self_loop_mask] -= 1.0
    L_tilde = torch.sparse_coo_tensor(ei_lap, ew_tilde, (n, n)).coalesce().to(device)

    return {
        "A_hat": A_hat,
        "A_norm": A_norm,
        "L_sym": L_sym,
        "L_tilde": L_tilde,
        "L_cheb": L_cheb,
    }


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


def train_task_batch(model, x, y, op, train_masks, val_masks, test_masks,
                     lr_values, wd_values, epochs, patience):
    optimizer = TaskAdam(model.parameters(), lr_values, wd_values)
    tasks = len(lr_values)
    best_val = torch.full((tasks,), -1.0, device=x.device)
    best_test = torch.zeros(tasks, device=x.device)
    stale = torch.zeros(tasks, dtype=torch.long, device=x.device)
    finished = torch.zeros(tasks, dtype=torch.bool, device=x.device)
    finished_epoch = torch.zeros(tasks, dtype=torch.long, device=x.device)
    histories = [[] for _ in range(tasks)]

    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        model.train()
        optimizer.zero_grad()
        logits = model(x, op)
        loss_per = masked_cross_entropy(logits, y, train_masks)

        finite = torch.isfinite(loss_per)
        active = (~finished) & finite
        if not active.any():
            break

        loss = loss_per[active].sum()
        if not torch.isfinite(loss):
            break

        loss.backward()
        optimizer.step(active)

        model.eval()
        with torch.no_grad():
            logits = model(x, op)
            pred = logits.argmax(dim=-1)
            val_acc = masked_accuracy(pred, y, val_masks)
            test_acc = masked_accuracy(pred, y, test_masks)

        for task_idx, value in enumerate(val_acc.detach().cpu().tolist()):
            if not finished[task_idx]:
                histories[task_idx].append(value)

        improved = (val_acc > best_val) & (~finished)
        best_val = torch.where(improved, val_acc, best_val)
        best_test = torch.where(improved, test_acc, best_test)
        stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
        newly_finished = (~finished) & (stale >= patience)
        finished_epoch = torch.where(
            newly_finished,
            torch.full_like(finished_epoch, epoch + 1),
            finished_epoch,
        )
        finished |= newly_finished
        if finished.all():
            break

    finished_epoch = torch.where(
        finished_epoch == 0,
        torch.full_like(finished_epoch, epochs_ran),
        finished_epoch,
    )
    return (
        best_val.detach().cpu().numpy(),
        best_test.detach().cpu().numpy(),
        finished_epoch.detach().cpu().numpy(),
        histories,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--models", nargs="+", default=["GPRGNN", "ChebGNN", "BernNet", "JacobiConv"])
    p.add_argument("--k", nargs="+", type=int, default=[4, 10])
    p.add_argument("--hidden", nargs="+", type=int, default=[64, 128])
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--epochs", nargs="+", type=int, default=[500])
    p.add_argument("--patience", nargs="+", type=int, default=[50])
    p.add_argument("--lr", nargs="+", type=float, default=[1e-2, 5e-3, 1e-3])
    p.add_argument("--weight-decay", nargs="+", type=float, default=[5e-4, 1e-4, 0.0])
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--dprate", type=float, default=0.0)
    p.add_argument("--max-task-batch", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--no-curves", action="store_true")
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
    selected = [(name, dataset) for name, dataset in all_datasets if name.lower() in wanted]
    found = {name.lower() for name, _ in selected}
    missing = sorted(wanted - found)
    if missing:
        available = ", ".join(name for name, _ in all_datasets)
        raise ValueError(f"Unknown dataset(s): {', '.join(missing)}. Available: {available}")
    return selected


def write_outputs(summary_path, rows, curves_dir=None, curve_rows=None):
    write_csv(summary_path, rows)
    if curves_dir is None or curve_rows is None:
        return
    curves_dir.mkdir(parents=True, exist_ok=True)
    for row_id, history in curve_rows:
        with open(curves_dir / f"{row_id}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_acc"])
            for epoch, val_acc in enumerate(history):
                writer.writerow([epoch, f"{val_acc:.4f}"])


def append_result(rows, curve_rows, dataset_name, model_name, k_val, hidden_dim,
                  lr_value, wd_value, epoch_value, patience_value, seed,
                  best_val, test_acc, epochs_ran, history, parallel_mode,
                  curves_dir):
    row_id = (
        f"{dataset_name}_{model_name}_K{k_val}_H{hidden_dim}"
        f"_lr{lr_value:g}_wd{wd_value:g}_e{epoch_value}_p{patience_value}_s{seed}"
    )
    rows.append({
        "id": row_id,
        "dataset": dataset_name,
        "model": model_name,
        "K": k_val,
        "hidden": hidden_dim,
        "lr": lr_value,
        "weight_decay": wd_value,
        "epochs": epoch_value,
        "patience": patience_value,
        "seed": seed,
        "best_val_acc": f"{best_val:.4f}",
        "test_acc": f"{test_acc:.4f}",
        "epochs_ran": int(epochs_ran),
        "parallel_mode": parallel_mode,
    })
    if curves_dir is not None:
        curve_rows.append((row_id, history))


def main():
    args = parse_args()
    unsupported = sorted(set(args.models) - set(MODELS))
    if unsupported:
        supported = ", ".join(MODELS)
        raise ValueError(
            f"Unsupported model(s) for massive mode: {', '.join(unsupported)}. "
            f"Supported: {supported}."
        )

    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) / f"paper_massive_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.csv"
    curves_dir = None if args.no_curves else run_dir / "curves"
    log_path = run_dir / "progress.log"
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    with open(run_dir / "config.json", "w") as f:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, f, indent=2)

    datasets = select_datasets(args.datasets)
    rows = []
    curve_rows = []

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Datasets: {len(datasets)}", flush=True)
    print(
        f"Sweep: models={args.models}, k={args.k}, hidden={args.hidden}, runs={args.runs}, "
        f"epochs={args.epochs}, patience={args.patience}, lr={args.lr}, "
        f"weight_decay={args.weight_decay}, max_task_batch={args.max_task_batch}, "
        f"device={args.device}, mask_dir={args.mask_dir}",
        flush=True,
    )

    for dataset_idx, (dataset_name, dataset) in enumerate(datasets, start=1):
        graph = dataset[0]
        operators = make_operators(graph, device)
        x = graph.x.to(device)
        y = graph.y.to(device)
        split_masks = [load_spatial_masks(dataset_name, args.mask_dir) for _ in range(args.runs)]

        print(
            f"\n[{dataset_idx}/{len(datasets)}] {dataset_name}: "
            f"{graph.num_nodes} nodes, {graph.num_edges} edges, {dataset.num_classes} classes",
            flush=True,
        )

        for model_name in args.models:
            ModelClass = BATCHED_MODELS[model_name]
            for k_val in args.k:
                for hidden_dim in args.hidden:
                    for epoch_value in values(args.epochs):
                        for patience_value in values(args.patience):
                            all_tasks = list(product(values(args.lr), values(args.weight_decay), range(args.runs)))
                            for start in range(0, len(all_tasks), args.max_task_batch):
                                task_chunk = all_tasks[start:start + args.max_task_batch]
                                lr_values = [task[0] for task in task_chunk]
                                wd_values = [task[1] for task in task_chunk]
                                seeds = [task[2] for task in task_chunk]
                                task_count = len(task_chunk)

                                train_masks = torch.stack([split_masks[s][0] for s in seeds]).to(device)
                                val_masks = torch.stack([split_masks[s][1] for s in seeds]).to(device)
                                test_masks = torch.stack([split_masks[s][2] for s in seeds]).to(device)

                                torch.manual_seed(
                                    args.init_seed
                                    + dataset_idx * 1_000_003
                                    + k_val * 10_007
                                    + hidden_dim * 101
                                    + start
                                )
                                model = ModelClass(
                                    tasks=task_count,
                                    in_dim=graph.x.size(1),
                                    hidden_dim=hidden_dim,
                                    num_classes=dataset.num_classes,
                                    k_val=k_val,
                                    dropout=args.dropout,
                                    dprate=args.dprate,
                                ).to(device)

                                op = operators[ModelClass.operator]
                                print(
                                    f"  {dataset_name} | {model_name} | K={k_val} | "
                                    f"hidden={hidden_dim} | epochs={epoch_value} | "
                                    f"patience={patience_value} | tasks {start + 1}-"
                                    f"{start + task_count}/{len(all_tasks)}",
                                    flush=True,
                                )

                                best_val, best_test, epochs_ran, histories = train_task_batch(
                                    model,
                                    x,
                                    y,
                                    op,
                                    train_masks,
                                    val_masks,
                                    test_masks,
                                    lr_values=lr_values,
                                    wd_values=wd_values,
                                    epochs=epoch_value,
                                    patience=patience_value,
                                )

                                for task_idx, (lr_value, wd_value, seed) in enumerate(task_chunk):
                                    append_result(
                                        rows,
                                        curve_rows,
                                        dataset_name,
                                        model_name,
                                        k_val,
                                        hidden_dim,
                                        lr_value,
                                        wd_value,
                                        epoch_value,
                                        patience_value,
                                        seed,
                                        best_val[task_idx],
                                        best_test[task_idx],
                                        epochs_ran[task_idx],
                                        histories[task_idx],
                                        "lr_wd_seed_batched",
                                        curves_dir,
                                    )

                                write_outputs(summary_path, rows, curves_dir, curve_rows)
                                curve_rows = []
                                del model, train_masks, val_masks, test_masks
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()

    print(f"\nWrote run to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
