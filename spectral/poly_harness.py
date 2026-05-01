from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian

# Polynomial Basis Setup - Bernstein, Chebyshev, etc.
def spmm(A, x): return torch.sparse.mm(A, x) if A.is_sparse else A @ x # helper

def values(x):
    return x if isinstance(x, (list, tuple)) else (x,)

class GPRGNN(nn.Module):
    operator = 'A_hat'
    def __init__(self, in_dim, hidden_dim, num_classes,  k_val, init="PPR", alpha=0.1, dropout=0.5, dprate=0):
        super().__init__()
        self.dropout = dropout
        self.K = k_val
        self.dprate = dprate

        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        if init == 'PPR':
            gammas = alpha * (1 - alpha) ** torch.arange(self.K + 1, dtype=torch.float)
            gammas[-1] = (1 - alpha) ** self.K
        elif init == 'Random':
            bound = (3 / (self.K + 1)) ** 0.5
            gammas = torch.empty(self.K + 1).uniform_(-bound, bound)
            gammas = gammas / gammas.abs().sum()
        elif init == 'SGC':
            # alpha is interpreted as an integer index here
            gammas = torch.zeros(self.K + 1)
            gammas[int(alpha)] = 1.0
        else:
            raise ValueError(f"Unknown init: {init}")
        self.gamma = nn.Parameter(gammas)


    def forward(self, x, A_hat):
        h = self.encoder(x)
        h = F.dropout(h, p=self.dprate, training=self.training)
        z = self.gamma[0] * h
        for k in range(self.K):
           h = spmm(A_hat, h)
           z = z + self.gamma[k+1]*h # handles the iterative adding, just do it in parallel rather than storing
        return z

class ChebNet(nn.Module):
    operator = 'L_tilde'
    def __init__(self, in_dim, hidden_dim, num_classes,  k_val, alpha=0.1, dropout=0.5, dprate=0):
        super().__init__()
        self.dropout = dropout
        self.K = k_val
        assert self.K >= 1
        self.dprate = dprate

        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.gamma = nn.Parameter(torch.randn(self.K + 1) * 0.1)


    def forward(self, x, L_tilde):
        h = self.encoder(x)
        h = F.dropout(h, p=self.dprate, training=self.training)

        t_prev = h
        t_curr = spmm(L_tilde, h)
        z = self.gamma[0] * t_prev + self.gamma[1] * t_curr
        for k in range(2, self.K + 1):
            t_next = 2 * spmm(L_tilde, t_curr) - t_prev
            z = z + self.gamma[k] * t_next
            t_prev, t_curr = t_curr, t_next
        return z

class BernNet(nn.Module):
    operator = 'L_sym'
    binom: torch.Tensor
    def __init__(self, in_dim, hidden_dim, num_classes, k_val, dropout=0.5, dprate=0.0):
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

        # theta init: ones, per the BernNet paper (gives flat all-pass at init).
        self.theta = nn.Parameter(torch.ones(self.K + 1))

        # precompute binomial coefficients to be faster during fwd passes
        self.register_buffer(
            'binom', torch.tensor([comb(self.K, k) for k in range(self.K + 1)], dtype=torch.float)
        )

    def forward(self, x, L_sym):
        h = self.encoder(x)
        h = F.dropout(h, p=self.dprate, training=self.training)

        # L_powers[k] = L^k h
        L_powers = [h]
        for _ in range(self.K):
            L_powers.append(spmm(L_sym, L_powers[-1]))

        theta = F.relu(self.theta) # important, makes it less expressive
        z = torch.zeros_like(h)
        for k in range(self.K + 1):
            term = L_powers[k]
            for _ in range(self.K - k):
                term = 2 * term - spmm(L_sym, term)
            coef = theta[k] * self.binom[k] / (2 ** self.K)
            z = z + coef * term
        return z

class JacobiConv(nn.Module):
    operator = 'A_norm'
    def __init__(self, in_dim, hidden_dim, num_classes, k_val,
                 a=1.0, b=1.0,
                 dropout=0.5, dprate=0.0):
        super().__init__()
        if a <= -1.0 or b <= -1.0:
            raise ValueError("Jacobi a and b must be greater than -1.")
        self.K = k_val
        self.a, self.b = a, b
        self.dropout = dropout
        self.dprate = dprate

        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # JacobiConv learns one polynomial filter per output channel.
        self.gamma = nn.Parameter(torch.randn(self.K + 1, num_classes) * 0.1)

    def _jacobi_step(self, k, P_prev, P_curr, A_norm):
        """Compute P_L from P_{L-1} (P_curr) and P_{L-2} (P_prev)."""
        a, b = self.a, self.b

        if k == 1:
            return ((a - b) / 2) * P_curr + ((a + b + 2) / 2) * spmm(A_norm, P_curr)

        theta = ((2 * k + a + b) * (2 * k + a + b - 1)) / (2 * k * (k + a + b))
        theta_prime = (
            (2 * k + a + b - 1) * (a**2 - b**2)
        ) / (2 * k * (k + a + b) * (2 * k + a + b - 2))
        theta_double = (
            (k + a - 1) * (k + b - 1) * (2 * k + a + b)
        ) / (k * (k + a + b) * (2 * k + a + b - 2))

        return theta * spmm(A_norm, P_curr) + theta_prime * P_curr - theta_double * P_prev

    def forward(self, x, A_norm):
        h = self.encoder(x)
        h = F.dropout(h, p=self.dprate, training=self.training)

        # P_0(A_norm) h = h
        P_prev = h
        P_curr = self._jacobi_step(1, None, P_prev, A_norm)
        z = self.gamma[0] * P_prev + self.gamma[1] * P_curr
        for k in range(2, self.K + 1):
            P_next = self._jacobi_step(k, P_prev, P_curr, A_norm)
            z = z + self.gamma[k] * P_next
            P_prev, P_curr = P_curr, P_next
        return z

# get train, val, test masks

def get_splits(data, train_r = 0.6, val_r = 0.2):
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

# train one model. val early stopping. return test acc
def make_operators(data, device):
    """Build operators used by the polynomial-basis models."""
    N = data.num_nodes

    # A_hat = D̃^{-1/2}(A+I)D̃^{-1/2}, used by GPR-style propagation.
    ei, ew = gcn_norm(data.edge_index, num_nodes=N, add_self_loops=True, dtype=torch.float)
    A_hat = torch.sparse_coo_tensor(ei, ew, (N, N)).coalesce().to(device)

    # A_norm = D^{-1/2}AD^{-1/2}, used by JacobiConv as P_k(I - L_sym).
    ei_A, ew_A = gcn_norm(data.edge_index, num_nodes=N, add_self_loops=False, dtype=torch.float)
    A_norm = torch.sparse_coo_tensor(ei_A, ew_A, (N, N)).coalesce().to(device)

    # L_sym = I - D^{-1/2} A D^{-1/2}, eigenvalues in [0, 2], no self-loops
    ei_L, ew_L = get_laplacian(data.edge_index, normalization='sym', num_nodes=N)
    L_sym = torch.sparse_coo_tensor(ei_L, ew_L.float(), (N, N)).coalesce().to(device)

    # L_tilde = L_sym - I (assuming lambda_max = 2), eigenvalues in [-1, 1]
    self_loop_mask = ei_L[0] == ei_L[1]
    ew_tilde = ew_L.float().clone()
    ew_tilde[self_loop_mask] -= 1.0
    L_tilde = torch.sparse_coo_tensor(ei_L, ew_tilde, (N, N)).coalesce().to(device)

    return {'A_hat': A_hat, 'A_norm': A_norm, 'L_sym': L_sym, 'L_tilde': L_tilde}


def fit(model, data, masks, operators, lr=1e-2, weight_decay=5e-4, max_epochs=300, patience=20, device='cuda'):
    op = operators[type(model).operator]
    tr_mask, va_mask, te_mask = (m.to(device) for m in masks)
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_va = -np.inf
    best_state = None
    patience_counter = 0
    val_history = []

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

# run for all datasets

MODELS = {
    'GPRGNN': GPRGNN,
    'ChebGNN': ChebNet,
    'BernNet': BernNet,
    'JacobiConv': JacobiConv,
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
    verbose=True,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}
    if models is None:
        models = tuple(MODELS.keys())

    total_datasets = len(datasets)
    for dataset_idx, (name, d) in enumerate(datasets, start=1):
        g = d[0]
        if verbose:
            print(
                f"[{dataset_idx}/{total_datasets}] {name}: "
                f"{g.num_nodes} nodes, {g.num_edges} edges, "
                f"{d.num_classes} classes",
                flush=True,
            )
        operators = make_operators(g, device)
        results[name] = {}

        for model_name in models:
            ModelClass = MODELS[model_name]
            for k in K:
                for hidden_dim in hidden_dims:
                    for lr_value in values(lr):
                        for wd_value in values(weight_decay):
                            for epoch_value in values(max_epochs):
                                for patience_value in values(patience):
                                    if verbose:
                                        print(
                                            f"  {name} | {model_name} | K={k} | "
                                            f"hidden={hidden_dim} | lr={lr_value:g} | "
                                            f"wd={wd_value:g} | epochs={epoch_value} | "
                                            f"patience={patience_value}",
                                            flush=True,
                                        )
                                    accs = []
                                    val_histories = []
                                    for i in range(n_runs):
                                        torch.manual_seed(i)
                                        np.random.seed(i)
                                        masks = get_splits(g)

                                        model = ModelClass(
                                            in_dim=g.x.size(1),
                                            hidden_dim=hidden_dim,
                                            num_classes=d.num_classes,
                                            k_val=k,
                                            dropout=dropout,
                                            dprate=dprate
                                        )
                                        acc, history = fit(
                                            model,
                                            g,
                                            masks,
                                            operators,
                                            lr=lr_value,
                                            weight_decay=wd_value,
                                            max_epochs=epoch_value,
                                            patience=patience_value,
                                            device=device,
                                        )
                                        accs.append(acc)
                                        val_histories.append(history)
                                        if verbose:
                                            print(
                                                f"    seed {i + 1}/{n_runs}: "
                                                f"test={acc:.4f}, epochs_ran={len(history)}",
                                                flush=True,
                                            )

                                    key = (
                                        model_name, k, hidden_dim,
                                        lr_value, wd_value, epoch_value, patience_value,
                                    )
                                    results[name][key] = (accs, val_histories)

    return results

if __name__ == '__main__':
    from datasets import build_datasets
    datasets = build_datasets()
    results = train_sweep(datasets, K=(4,), hidden_dims=(64,), n_runs=3)

    for dataset_name, configs in results.items():
        for (model_name, k, hidden_dim, lr, wd, epochs, patience), (accs, _) in configs.items():
            print(f"{dataset_name:15s} {model_name:12s} h={hidden_dim} "
                  f"K={k} lr={lr:g} wd={wd:g} epochs={epochs} patience={patience} "
                  f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")
