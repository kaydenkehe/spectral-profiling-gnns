import os
import sys
from pathlib import Path

# Add parent directory to path to import from common
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import torch
torch.backends.cuda.preferred_linalg_library('magma')
from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork,
    Actor, HeterophilousGraphDataset, WikiCS,
)
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj, subgraph
from common.datasets import FixedWikipediaNetwork
import matplotlib.pyplot as plt

# collect graphs

data_dir = '../graph_data'

cora_dataset = Planetoid(root=data_dir, name='Cora')
pubmed = Planetoid(root=data_dir, name='PubMed')
amazon_dataset = Amazon(root=data_dir, name='Photo')
texas_dataset = WebKB(root=data_dir, name='Texas')
chameleon_dataset = FixedWikipediaNetwork(root=data_dir, name='Chameleon')
citeseer = Planetoid(root=data_dir, name='CiteSeer')
amazon_comp = Amazon(root=data_dir, name='Computers')
coauthor_cs = Coauthor(root=data_dir, name='CS')
# coauthor_physics = Coauthor(root=data_dir, name='Physics')
cornell = WebKB(root=data_dir, name='Cornell')
wisconsin = WebKB(root=data_dir, name='Wisconsin')
actor = Actor(root=os.path.join(data_dir, 'actor'))
wikics = WikiCS(root=os.path.join(data_dir, 'wikics'))
squirrel = FixedWikipediaNetwork(root=data_dir, name='Squirrel')
roman_empire = HeterophilousGraphDataset(root=data_dir, name='Roman-empire')
amazon_ratings = HeterophilousGraphDataset(root=data_dir, name='Amazon-ratings')
minesweeper = HeterophilousGraphDataset(root=data_dir, name='Minesweeper', pre_transform=T.ToUndirected())
tolokers = HeterophilousGraphDataset(root=data_dir, name='Tolokers', pre_transform=T.ToUndirected())
# questions = HeterophilousGraphDataset(root=data_dir, name='Questions', pre_transform=T.ToUndirected())


names = ['Cora', 'PubMed', 'Amazon Photo', 'Texas', 'Chameleon', 'CiteSeer', 'Amazon Computers', 'Coauthor CS',
         'Cornell', 'Wisconsin', 'Actor', 'WikiCS', 'Squirrel', 'Roman Empire', 'Amazon Ratings', 'Minesweeper', 'Tolokers']
datasets = [cora_dataset, pubmed, amazon_dataset, texas_dataset, chameleon_dataset, citeseer, amazon_comp, coauthor_cs,
            cornell, wisconsin, actor, wikics, squirrel, roman_empire, amazon_ratings, minesweeper, tolokers]
graphs = [dataset[0] for dataset in datasets]
labels = [graph.y for graph in graphs]

# compute homophily
# this is a vectorized version given to me by AI - non-vectorized in homophily.py
# computed by counting co-occurrences of class labels across edges,
# then dividing the diagonal (same-class edges) by the row sums (total edges per class)
# to get per-class homophily. Then we average across classes to get an overall homophily score.

def compute_homophily(edge_index, labels):
    N = labels.shape[0]
    num_classes = labels.max().item() + 1
    A = to_dense_adj(edge_index, max_num_nodes=N)[0]
    src, dst = A.nonzero(as_tuple=True)
    idx = labels[src] * num_classes + labels[dst]
    co = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes).float()

    return (co.diag() / co.sum(dim=1).clamp(min=1)).mean().item()

# compute spectra

def compute_spectrum(edge_index, n):
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=n)
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=n)[0]
    evals, evecs = torch.linalg.eigh(L)

    return evals, evecs

# compute SLP

def compute_slp(evecs, labels, num_classes):
    N = labels.shape[0]

    Y = F.one_hot(labels, num_classes=num_classes).float()
    Y_tilde = Y - Y.mean(dim=0, keepdim=True)
    proj = (evecs.T @ Y_tilde) ** 2
    Y_norm = torch.norm(Y_tilde, dim=0) ** 2 + 1e-8
    pi_c = proj / Y_norm
    pi = pi_c.mean(dim=1)
    cdf = torch.cumsum(pi, dim=0)

    return cdf

# compute metrics, write to json

lambd_grid = np.linspace(0, 2, num=50)
results = {}
full = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for d, g, name in zip(datasets, graphs, names):
    C = d.num_classes
    n = g.num_nodes
    g = g.to(device)

    evals_f, evecs_f = compute_spectrum(g.edge_index, n)
    cdf_f = compute_slp(evecs_f, g.y, C)
    h = compute_homophily(g.edge_index, g.y)

    evals_np = evals_f.cpu().numpy()
    cdf_np = cdf_f.cpu().numpy()
    full[name] = (evals_np, cdf_np)

    # gives number of evals <= each lambd in the grid,
    # -1, gives index of largest eval <= lambd 
    idx = np.searchsorted(evals_np, lambd_grid, side='right') - 1

    # look up cdf at those indices
    profile = np.where(idx >= 0, cdf_np[idx.clip(min=0)], 0.0)

    results[name] = {
        'num_classes': C,
        'num_nodes': n,
        'num_edges': g.edge_index.size(1),
        'homophily': h,
        'eigenvalues': lambd_grid.tolist(),
        'cdf': profile.tolist(),
    }

with open('metrics.json', 'w') as f:
    json.dump(results, f)

# plot

n_rows, n_cols = 3, 7
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3 * n_cols, 3 * n_rows),
                         sharex=True, sharey=True)
axes_flat = axes.flatten()

for ax, (name, r) in zip(axes_flat, results.items()):
    evals_np, cdf_np = full[name]
    ax.step(evals_np, cdf_np, where='post', label='full', linewidth=1.5)
    ax.set_title(f'{name} (h={r["homophily"]:.2f})', fontsize=10)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

for ax in axes_flat[len(results):]:
    ax.set_visible(False)

for ax in axes[-1, :]:
    ax.set_xlabel(r'$\lambda^*$')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\Pi(\lambda^*)$')

axes_flat[n_cols - 1].legend(loc='lower right', fontsize=8)
fig.tight_layout()
plt.savefig('metrics.png', dpi=150)
plt.show()




