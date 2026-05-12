# UNFINISHED. testing how good low dim projections are 

import os
import time
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork,
    Actor, HeterophilousGraphDataset,
)
from torch_geometric.utils import get_laplacian, to_dense_adj, to_undirected

# collect graphs

data_dir = '../graph_data'

cora_dataset = Planetoid(root=data_dir, name='Cora')
amazon_dataset = Amazon(root=data_dir, name='Photo')
texas_dataset = WebKB(root=data_dir, name='Texas')
chameleon_dataset = WikipediaNetwork(root=data_dir, name='Chameleon', geom_gcn_preprocess=True)
citeseer = Planetoid(root=data_dir, name='CiteSeer')
amazon_comp = Amazon(root=data_dir, name='Computers')
coauthor_cs = Coauthor(root=data_dir, name='CS')
cornell = WebKB(root=data_dir, name='Cornell')
wisconsin = WebKB(root=data_dir, name='Wisconsin')
actor = Actor(root=os.path.join(data_dir, 'actor'))
squirrel = WikipediaNetwork(root=data_dir, name='Squirrel', geom_gcn_preprocess=True)
minesweeper = HeterophilousGraphDataset(root=data_dir, name='Minesweeper', pre_transform=T.ToUndirected())
tolokers = HeterophilousGraphDataset(root=data_dir, name='Tolokers', pre_transform=T.ToUndirected())

names = ['Cora', 'Amazon Photo', 'Texas', 'Chameleon', 'CiteSeer', 'Amazon Computers', 'Coauthor CS', 'Cornell', 'Wisconsin', 'Actor', 'Squirrel', 'Minesweeper', 'Tolokers']
datasets = [cora_dataset, amazon_dataset, texas_dataset, chameleon_dataset, citeseer, amazon_comp, coauthor_cs, cornell, wisconsin, actor, squirrel, minesweeper, tolokers]
graphs = [dataset[0] for dataset in datasets]
labels = [graph.y for graph in graphs]

# homophily

def compute_homophily(edge_index, labels):
    N = labels.shape[0]
    num_classes = labels.max().item() + 1
    A = to_dense_adj(edge_index, max_num_nodes=N)[0]
    src, dst = A.nonzero(as_tuple=True)
    idx = labels[src] * num_classes + labels[dst]
    co = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes).float()
 
    return (co.diag() / co.sum(dim=1).clamp(min=1)).mean().item()

# compute slp from evecs and labels

def compute_slp(evecs, labels, num_classes):
    Y = F.one_hot(labels, num_classes=num_classes).float()
    Y_tilde = Y - Y.mean(dim=0, keepdim=True)
    proj = (evecs.T @ Y_tilde) ** 2
    Y_norm = torch.norm(Y_tilde, dim=0) ** 2 + 1e-8
    pi_c = proj / Y_norm
    pi = pi_c.mean(dim=1)
    pi = pi / pi.sum().clamp(min=1e-12)
    cdf = torch.cumsum(pi, dim=0)
 
    return cdf

# --- METHODS ---

# full eigh computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def full_eigh(edge_index, labels, num_classes, n):
    edge_index = to_undirected(edge_index, num_nodes=n)
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=n)
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=n)[0]
    if n * n > 400_000_000:
        L = L.cpu()
    else:
        L = L.to(device)
    evals, evecs = torch.linalg.eigh(L)
    cdf = compute_slp(evecs, labels.to(evecs.device), num_classes)

    return evals.cpu().numpy(), cdf.cpu().numpy()


methods = {
    'full_eigh': full_eigh,
}

# run & time each method

results = {}

for d, g, name in zip(datasets, graphs, names):
    C = d.num_classes
    n = g.num_nodes
    h = compute_homophily(g.edge_index, g.y)

    method_results = {}
    for m_name, m in methods.items():
        start_time = time.perf_counter()
        evals, cdf = m(g.edge_index, g.y, C, n)
        elapsed = time.perf_counter() - start_time
        method_results[m_name] = {
            'evals': evals,
            'cdf': cdf,
            'time': elapsed,
        }

    results[name] = {
        'homophily': h,
        'num_nodes': n,
        'methods': method_results,
    }

# write times to csv

method_names = list(methods.keys())
with open('timings.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dataset', 'num_nodes', 'homophily'] + method_names)
    for name in names:
        r = results[name]
        row = [name, f'{r["homophily"]:.2f}', r['num_nodes']]
        row += [f'{r["methods"][m]["time"]:.3f}' for m in method_names]
        w.writerow(row)

# plot CDFs

n_rows, n_cols = 2, 7
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3 * n_cols, 3 * n_rows),
                         sharex=True, sharey=True)
axes_flat = axes.flatten()
 
for ax, (name, r) in zip(axes_flat, results.items()):
    for m in method_names:
        evals = r['methods'][m]['evals']
        cdf = r['methods'][m]['cdf']
        ax.step(evals, cdf, where='post', label=m, linewidth=1.2, alpha=0.8)
    ax.set_title(f'{name} (h={r["homophily"]:.2f})', fontsize=10)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
 
for ax in axes_flat[len(results):]:
    ax.set_visible(False)
 
for ax in axes[-1, :]:
    ax.set_xlabel(r'$\lambda$')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\Pi(\lambda)$')
 
axes_flat[n_cols - 1].legend(loc='lower right', fontsize=8)
fig.tight_layout()
plt.savefig('slp_comparison.png', dpi=150)
plt.close(fig)











