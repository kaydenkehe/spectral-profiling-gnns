from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, ChebConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


SUPPORTED_MODELS = ("gprgnn", "appnp", "chebnet", "bernnet", "jacobiconv")


@dataclass(frozen=True)
class SpectralConfig:
    model: str
    basis: str
    order: int
    hidden: int
    alpha: float | None = None
    jacobi_alpha: float | None = None
    jacobi_beta: float | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_splits(data, train_r: float = 0.6, val_r: float = 0.2):
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


def clean_labels(y: torch.Tensor) -> torch.Tensor:
    if y.dim() > 1:
        y = y.squeeze()
    return y.long()


class PredictionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class SymmetricAdjacency(MessagePassing):
    def __init__(self, add_self_loops: bool):
        super().__init__(aggr="add")
        self.add_self_loops = add_self_loops

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, norm = gcn_norm(
            edge_index,
            edge_weight=None,
            num_nodes=x.size(0),
            improved=False,
            add_self_loops=self.add_self_loops,
            flow=self.flow,
            dtype=x.dtype,
        )
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class GPRPropagation(nn.Module):
    """GPR-GNN propagation with learnable generalized PageRank weights."""

    def __init__(self, order: int, alpha: float):
        super().__init__()
        self.order = order
        self.propagate_once = SymmetricAdjacency(add_self_loops=True)
        self.weights = nn.Parameter(self._ppr_initialization(order, alpha))

    @staticmethod
    def _ppr_initialization(order: int, alpha: float) -> torch.Tensor:
        if order == 0:
            return torch.ones(1)
        weights = [alpha * (1.0 - alpha) ** k for k in range(order)]
        weights.append((1.0 - alpha) ** order)
        return torch.tensor(weights, dtype=torch.float)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = x
        out = self.weights[0] * hidden
        for k in range(1, self.order + 1):
            hidden = self.propagate_once(hidden, edge_index)
            out = out + self.weights[k] * hidden
        return out


class GPRGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        order: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.mlp = PredictionMLP(in_dim, hidden_dim, out_dim, dropout)
        self.propagation = GPRPropagation(order, alpha)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        return self.propagation(logits, edge_index)


class APPNPNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        order: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.mlp = PredictionMLP(in_dim, hidden_dim, out_dim, dropout)
        self.propagation = APPNP(K=order, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagation(self.mlp(x), edge_index)


class ChebNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        order: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = ChebConv(in_dim, hidden_dim, K=order, normalization="sym")
        self.conv2 = ChebConv(hidden_dim, out_dim, K=order, normalization="sym")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index, lambda_max=2.0)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, lambda_max=2.0)


class LaplacianOps(nn.Module):
    def __init__(self):
        super().__init__()
        self.adj = SymmetricAdjacency(add_self_loops=False)

    def adjoint(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.adj(x, edge_index)

    def laplacian(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return x - self.adjoint(x, edge_index)

    def two_minus_laplacian(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return x + self.adjoint(x, edge_index)

    def scaled_laplacian(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.laplacian(x, edge_index) - x


class BernPropagation(nn.Module):
    """Bernstein basis over the normalized Laplacian spectrum in [0, 2]."""

    def __init__(self, order: int):
        super().__init__()
        self.order = order
        self.ops = LaplacianOps()
        self.coeffs = nn.Parameter(torch.ones(order + 1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        l_powers = [x]
        for _ in range(self.order):
            l_powers.append(self.ops.laplacian(l_powers[-1], edge_index))

        out = x.new_zeros(x.shape)
        scale = float(2 ** self.order)
        for k in range(self.order + 1):
            y = l_powers[k]
            for _ in range(self.order - k):
                y = self.ops.two_minus_laplacian(y, edge_index)
            basis_scale = math.comb(self.order, k) / scale
            out = out + self.coeffs[k] * basis_scale * y
        return out


class BernNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        order: int,
        dropout: float,
    ):
        super().__init__()
        self.mlp = PredictionMLP(in_dim, hidden_dim, out_dim, dropout)
        self.propagation = BernPropagation(order)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        return self.propagation(logits, edge_index)


class JacobiPropagation(nn.Module):
    """Jacobi polynomial basis on the scaled normalized Laplacian."""

    def __init__(self, order: int, alpha: float, beta: float):
        super().__init__()
        if alpha <= -1.0 or beta <= -1.0:
            raise ValueError("Jacobi alpha and beta must both be greater than -1.")
        self.order = order
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ops = LaplacianOps()
        self.coeffs = nn.Parameter(torch.ones(order + 1) / float(order + 1))

    def _scaled_op(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.ops.scaled_laplacian(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        b = self.beta
        polys = [x]

        if self.order >= 1:
            p1 = 0.5 * ((a - b) * x + (a + b + 2.0) * self._scaled_op(x, edge_index))
            polys.append(p1)

        for n in range(2, self.order + 1):
            prev = polys[-1]
            prevprev = polys[-2]
            denom = 2.0 * n * (n + a + b) * (2.0 * n + a + b - 2.0)
            c0 = (2.0 * n + a + b - 1.0) * (a * a - b * b)
            c1 = (
                (2.0 * n + a + b - 1.0)
                * (2.0 * n + a + b)
                * (2.0 * n + a + b - 2.0)
            )
            c2 = 2.0 * (n + a - 1.0) * (n + b - 1.0) * (2.0 * n + a + b)
            pn = (c0 * prev + c1 * self._scaled_op(prev, edge_index) - c2 * prevprev) / denom
            polys.append(pn)

        out = x.new_zeros(x.shape)
        for k, basis_value in enumerate(polys):
            out = out + self.coeffs[k] * basis_value
        return out


class JacobiConvNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        order: int,
        alpha: float,
        beta: float,
        dropout: float,
    ):
        super().__init__()
        self.mlp = PredictionMLP(in_dim, hidden_dim, out_dim, dropout)
        self.propagation = JacobiPropagation(order, alpha, beta)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(x)
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        return self.propagation(logits, edge_index)


def build_model(
    config: SpectralConfig,
    in_dim: int,
    out_dim: int,
    dropout: float,
) -> nn.Module:
    if config.model == "gprgnn":
        return GPRGNN(in_dim, config.hidden, out_dim, config.order, config.alpha or 0.1, dropout)
    if config.model == "appnp":
        return APPNPNet(in_dim, config.hidden, out_dim, config.order, config.alpha or 0.1, dropout)
    if config.model == "chebnet":
        return ChebNet(in_dim, config.hidden, out_dim, config.order, dropout)
    if config.model == "bernnet":
        return BernNet(in_dim, config.hidden, out_dim, config.order, dropout)
    if config.model == "jacobiconv":
        return JacobiConvNet(
            in_dim,
            config.hidden,
            out_dim,
            config.order,
            config.jacobi_alpha if config.jacobi_alpha is not None else 0.0,
            config.jacobi_beta if config.jacobi_beta is not None else 0.0,
            dropout,
        )
    raise ValueError(f"Unsupported spectral model: {config.model}")


def expand_configs(
    models: Iterable[str],
    hidden_dims: Iterable[int],
    orders: Iterable[int],
    alpha_values: Iterable[float],
    jacobi_pairs: Iterable[tuple[float, float]],
) -> list[SpectralConfig]:
    configs: list[SpectralConfig] = []
    for model in models:
        for order in orders:
            for hidden in hidden_dims:
                if model in {"gprgnn", "appnp"}:
                    for alpha in alpha_values:
                        basis = "monomial" if model == "gprgnn" else "ppr"
                        configs.append(SpectralConfig(model, basis, order, hidden, alpha=alpha))
                elif model == "chebnet":
                    configs.append(SpectralConfig(model, "chebyshev", order, hidden))
                elif model == "bernnet":
                    configs.append(SpectralConfig(model, "bernstein", order, hidden))
                elif model == "jacobiconv":
                    for jacobi_alpha, jacobi_beta in jacobi_pairs:
                        configs.append(
                            SpectralConfig(
                                model,
                                "jacobi",
                                order,
                                hidden,
                                jacobi_alpha=jacobi_alpha,
                                jacobi_beta=jacobi_beta,
                            )
                        )
                else:
                    raise ValueError(f"Unsupported spectral model: {model}")
    return configs


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    pred = logits.argmax(dim=1)
    return (pred[mask] == labels[mask]).float().mean().item()


def fit(
    model: nn.Module,
    data,
    masks,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: str,
):
    tr_mask, va_mask, te_mask = (m.to(device) for m in masks)
    data = data.clone().to(device)
    data.x = data.x.float()
    labels = clean_labels(data.y)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -np.inf
    best_state = None
    patience_counter = 0
    val_history: list[float] = []

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[tr_mask], labels[tr_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_acc = accuracy(logits, labels, va_mask)
            val_history.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_acc = accuracy(logits, labels, te_mask)

    return test_acc, float(best_val), val_history


def run_single(
    dataset,
    config: SpectralConfig,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    dropout: float,
    device: str,
):
    set_seed(seed)
    graph = dataset[0]
    masks = get_splits(graph)
    model = build_model(
        config=config,
        in_dim=graph.x.size(1),
        out_dim=dataset.num_classes,
        dropout=dropout,
    )
    return fit(model, graph, masks, lr, weight_decay, max_epochs, patience, device)
