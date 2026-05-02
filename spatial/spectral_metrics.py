import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj, subgraph
from datasets import build_datasets

# homophily

def compute_homophily(edge_index, labels):
    N = labels.shape[0]
    num_classes = labels.max().item() + 1
    A = to_dense_adj(edge_index, max_num_nodes=N)[0]
    src, dst = A.nonzero(as_tuple=True)
    idx = labels[src] * num_classes + labels[dst]
    co = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes).float()

    return (co.diag() / co.sum(dim=1).clamp(min=1)).mean().item()

# spectrum of normalized Laplacian

def compute_spectrum(edge_index, n):
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=n)
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=n)[0]
    evals, evecs = torch.linalg.eigh(L)

    return evals, evecs

# spectral label profile

def compute_slp(evecs, labels, num_classes):
    Y = F.one_hot(labels, num_classes=num_classes).float()
    Y_tilde = Y - Y.mean(dim=0, keepdim=True)
    proj = (evecs.T @ Y_tilde) ** 2
    Y_norm = torch.norm(Y_tilde, dim=0) ** 2 + 1e-8
    pi_c = proj / Y_norm
    pi = pi_c.mean(dim=1)
    cdf = torch.cumsum(pi, dim=0)

    return cdf

# walk datasets, load saved train mask, compute metrics on the train subgraph,
# write a single metrics.json (same format as analysis/metrics.json).

masks_dir = Path(__file__).resolve().parent / 'masks'
out_path = Path(__file__).resolve().parent / 'metrics.json'
lambd_grid = np.linspace(0, 2, num=50)

results = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

datasets = build_datasets()
for name, d in datasets:
    g = d[0]
    mask_file = masks_dir / f'{name}.pt'
    masks = torch.load(mask_file)
    train_mask = masks['train_mask']

    sub_ei, _ = subgraph(
        train_mask, g.edge_index,
        relabel_nodes=True, num_nodes=g.num_nodes,
    )
    sub_y = g.y[train_mask]
    n_sub = int(train_mask.sum().item())

    sub_ei = sub_ei.to(device)
    sub_y = sub_y.to(device)

    evals, evecs = compute_spectrum(sub_ei, n_sub)
    cdf = compute_slp(evecs, sub_y, d.num_classes)
    h = compute_homophily(sub_ei, sub_y)

    evals_np = evals.cpu().numpy()
    cdf_np = cdf.cpu().numpy()
    idx = np.searchsorted(evals_np, lambd_grid, side='right') - 1
    profile = np.where(idx >= 0, cdf_np[idx.clip(min=0)], 0.0)

    results[name] = {
        'num_classes': int(d.num_classes),
        'num_nodes': n_sub,
        'num_edges': int(sub_ei.size(1)),
        'homophily': float(h),
        'eigenvalues': lambd_grid.tolist(),
        'cdf': profile.tolist(),
    }

with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

