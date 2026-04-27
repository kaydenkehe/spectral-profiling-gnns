import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

# similar MLP setup

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_dim, hidden_dim))
        self.layers.append(Linear(hidden_dim, out_dim))
        self.dropout = dropout 
        

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x

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

def fit(model, data, masks, lr=1e-2, weight_decay=5e-4, max_epochs=300, patience=20, device='cuda'):
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
        out = model(data.x)
        loss = F.cross_entropy(out[tr_mask], data.y[tr_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x)
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
        out = model(data.x)
        te_acc = (out.argmax(dim=1)[te_mask] == data.y[te_mask]).float().mean().item()

    return te_acc, val_history

# run for all datasets

def train_sweep(datasets, depths=(2, 3, 4), hidden_dims=(16, 32, 64), n_runs=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for name, d in datasets:
        g = d[0]
        results[name] = {}
        for depth in depths:
            for hidden_dim in hidden_dims:
                accs = []
                val_histories = []

                for i in range(n_runs):
                    torch.manual_seed(i)
                    np.random.seed(i)
                    masks = get_splits(g)

                    model = MLP(
                        in_dim=g.x.size(1),
                        hidden_dim=hidden_dim,
                        out_dim=d.num_classes,
                        num_layers=depth
                    )

                    acc, val_history = fit(model, g, masks, device=device)
                    accs.append(acc)
                    val_histories.append(val_history)
                    # print(f"{name} L={depth} H={hidden_dim} test={acc:.4f}")

                results[name][(depth, hidden_dim)] = (accs, val_histories)
                # print(f"{name} L={depth} H={hidden_dim} mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")

    return results


