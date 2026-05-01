from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
from torch_geometric.nn import ChebConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian


def spmm(A, x):
    return torch.sparse.mm(A, x) if A.is_sparse else A @ x


class GPRGNN(nn.Module):
    operator = "A_hat"

    def __init__(self, in_dim, hidden_dim, num_classes, k_val,
                 init="PPR", alpha=0.1, dropout=0.5, dprate=0.0):
        super().__init__()
        self.K = k_val
        self.dropout = dropout
        self.dprate = dprate

        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if init == "PPR":
            gammas = alpha * (1 - alpha) ** torch.arange(self.K + 1, dtype=torch.float)
            gammas[-1] = (1 - alpha) ** self.K
        elif init == "Random":
            bound = (3 / (self.K + 1)) ** 0.5
            gammas = torch.empty(self.K + 1).uniform_(-bound, bound)
            gammas = gammas / gammas.abs().sum()
        else:
            raise ValueError(f"Unknown GPR init: {init}")

        self.gamma = nn.Parameter(gammas)

    def forward(self, x, A_hat):
        h = self.encoder(x)
        h = F.dropout(h, p=self.dprate, training=self.training)
        z = self.gamma[0] * h
        for k in range(self.K):
            h = spmm(A_hat, h)
            z = z + self.gamma[k + 1] * h
        return z


class ChebNet(nn.Module):
    operator = "edge_index"

    def __init__(self, in_dim, hidden_dim, num_classes, k_val,
                 dropout=0.5, dprate=0.0):
        super().__init__()
        self.dropout = dropout
        self.dprate = dprate
        self.conv1 = ChebConv(in_dim, hidden_dim, K=k_val, normalization="sym")
        self.conv2 = ChebConv(hidden_dim, num_classes, K=k_val, normalization="sym")

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dprate, training=self.training)
        return self.conv2(x, edge_index)


class BernNet(nn.Module):
    operator = "L_sym"

    def __init__(self, in_dim, hidden_dim, num_classes, k_val,
                 dropout=0.5, dprate=0.0):
        super().__init__()
        self.K = k_val
        self.dropout = dropout
        self.dprate = dprate

        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.theta = nn.Parameter(torch.ones(self.K + 1))
        self.register_buffer(
            "binom",
            torch.tensor([comb(self.K, k) for k in range(self.K + 1)], dtype=torch.float),
        )

    def forward(self, x, L_sym):
        h = self.encoder(x)
        h = F.dropout(h, p=self.dprate, training=self.training)

        l_powers = [h]
        for _ in range(self.K):
            l_powers.append(spmm(L_sym, l_powers[-1]))

        theta = F.relu(self.theta)
        z = torch.zeros_like(h)
        for k in range(self.K + 1):
            term = l_powers[k]
            for _ in range(self.K - k):
                term = 2 * term - spmm(L_sym, term)
            z = z + theta[k] * self.binom[k] * term / (2 ** self.K)
        return z


class JacobiConv(nn.Module):
    operator = "A_norm"

    def __init__(self, in_dim, hidden_dim, num_classes, k_val,
                 a=1.0, b=1.0, dropout=0.5, dprate=0.0):
        super().__init__()
        if a <= -1.0 or b <= -1.0:
            raise ValueError("Jacobi a and b must be greater than -1.")
        self.K = k_val
        self.a = a
        self.b = b
        self.dropout = dropout
        self.dprate = dprate

        # JacobiConv: linear transform first, then filter each output channel.
        self.lin = nn.Linear(in_dim, num_classes)
        self.alpha = nn.Parameter(torch.randn(self.K + 1, num_classes) * 0.1)

    def _step(self, k, prevprev, prev, A_norm):
        a, b = self.a, self.b
        if k == 1:
            return ((a - b) / 2) * prev + ((a + b + 2) / 2) * spmm(A_norm, prev)

        theta = ((2 * k + a + b) * (2 * k + a + b - 1)) / (2 * k * (k + a + b))
        theta_prime = (
            (2 * k + a + b - 1) * (a**2 - b**2)
        ) / (2 * k * (k + a + b) * (2 * k + a + b - 2))
        theta_double = (
            (k + a - 1) * (k + b - 1) * (2 * k + a + b)
        ) / (k * (k + a + b) * (2 * k + a + b - 2))

        return theta * spmm(A_norm, prev) + theta_prime * prev - theta_double * prevprev

    def forward(self, x, A_norm):
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.lin(x)
        h = F.dropout(h, p=self.dprate, training=self.training)

        p_prev = h
        z = self.alpha[0] * p_prev
        if self.K == 0:
            return z

        p_curr = self._step(1, None, p_prev, A_norm)
        z = z + self.alpha[1] * p_curr
        for k in range(2, self.K + 1):
            p_next = self._step(k, p_prev, p_curr, A_norm)
            z = z + self.alpha[k] * p_next
            p_prev, p_curr = p_curr, p_next
        return z


def get_splits(data, train_r=0.6, val_r=0.2):
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

    return {"A_hat": A_hat, "A_norm": A_norm, "L_sym": L_sym}


def fit(model, data, masks, operators, lr=1e-2, weight_decay=5e-4,
        max_epochs=300, patience=20, device="cuda"):
    tr_mask, va_mask, te_mask = (m.to(device) for m in masks)
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_va = -np.inf
    best_state = None
    patience_counter = 0
    val_history = []

    op_name = type(model).operator
    op = data.edge_index if op_name == "edge_index" else operators[op_name]

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, op)
        loss = F.cross_entropy(out[tr_mask], data.y[tr_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, op)
            va_acc = (out.argmax(dim=1)[va_mask] == data.y[va_mask]).float().mean().item()
            val_history.append(va_acc)

        if va_acc > best_va:
            best_va = va_acc
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, op)
        te_acc = (out.argmax(dim=1)[te_mask] == data.y[te_mask]).float().mean().item()

    return te_acc, val_history


MODELS = {
    "GPRGNN": GPRGNN,
    "ChebGNN": ChebNet,
    "BernNet": BernNet,
    "JacobiConv": JacobiConv,
}


def train_sweep(
    datasets,
    models=None,
    K=(4, 8, 10),
    hidden_dims=(64,),
    n_runs=3,
    dropout=0.5,
    dprate=0.0,
    lr=1e-2,
    weight_decay=5e-4,
    max_epochs=300,
    patience=20,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if models is None:
        models = tuple(MODELS.keys())

    results = {}
    for name, dataset in datasets:
        graph = dataset[0]
        operators = make_operators(graph, device)
        results[name] = {}

        for model_name in models:
            ModelClass = MODELS[model_name]
            for k in K:
                for hidden_dim in hidden_dims:
                    accs = []
                    val_histories = []
                    for seed in range(n_runs):
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        masks = get_splits(graph)

                        model = ModelClass(
                            in_dim=graph.x.size(1),
                            hidden_dim=hidden_dim,
                            num_classes=dataset.num_classes,
                            k_val=k,
                            dropout=dropout,
                            dprate=dprate,
                        )
                        acc, history = fit(
                            model,
                            graph,
                            masks,
                            operators,
                            lr=lr,
                            weight_decay=weight_decay,
                            max_epochs=max_epochs,
                            patience=patience,
                            device=device,
                        )
                        accs.append(acc)
                        val_histories.append(history)

                    results[name][(model_name, k, hidden_dim)] = (accs, val_histories)

    return results
