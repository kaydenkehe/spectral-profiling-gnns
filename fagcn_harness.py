import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FAConv
from copy import deepcopy
import numpy as np

# FAGCN setup adapted from paper repo

class FAGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, eps, num_layers=2):
        super().__init__()
        self.eps = eps
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(FAConv(hidden_dim, eps=eps, dropout=dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h, edge_index):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, raw, edge_index)
        h = self.t2(h)
        return F


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
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[tr_mask], data.y[tr_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
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
        out = model(data.x, data.edge_index)
        te_acc = (out.argmax(dim=1)[te_mask] == data.y[te_mask]).float().mean().item()

    return te_acc, val_history

# run for all datasets

def train_sweep(datasets, depths=(2, 3, 4), hidden_dims=(16, 32, 64), eps_values=(0.1, 0.3, 0.5), dropout=0.5, n_runs=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    for name, d in datasets:
        g = d[0]
        results[name] = {}
        for depth in depths:
            for hidden_dim in hidden_dims:
                for eps in eps_values:
                    accs = []
                    val_histories = []

                    for i in range(n_runs):
                        torch.manual_seed(i)
                        np.random.seed(i)
                        masks = get_splits(g)

                        model = FAGCN(
                            in_dim=g.x.size(1),
                            hidden_dim=hidden_dim,
                            out_dim=d.num_classes,
                            dropout=dropout,
                            eps=eps,
                            num_layers=depth,
                        )

                        acc, val_history = fit(model, g, masks, device=device)
                        accs.append(acc)
                        val_histories.append(val_history)
                        # print(f"{name} L={depth} H={hidden_dim} eps={eps} "
                        #       f"seed={i} test={acc:.4f}")

                    results[name][(depth, hidden_dim, eps)] = (accs, val_histories)
                    # print(f"{name} L={depth} H={hidden_dim} eps={eps} "
                    #       f"mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")

    return results




