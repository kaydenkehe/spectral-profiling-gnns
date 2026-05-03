import os
import sys
from pathlib import Path

# Add parent directory to path to import from common
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.backends.cuda.preferred_linalg_library('magma')
import torch.nn.functional as F
from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, WebKB,
    Actor, HeterophilousGraphDataset, WikiCS,
)
import torch_geometric.transforms as T
from torch_geometric.utils import get_laplacian, to_dense_adj, subgraph
import matplotlib.pyplot as plt
from common.datasets import FixedWikipediaNetwork

torch.manual_seed(777)
data_dir = '../graph_data'
os.makedirs(data_dir, exist_ok=True)

# collect graphs

cora_dataset = Planetoid(root=data_dir, name='Cora')
pubmed = Planetoid(root=data_dir, name='PubMed')
amazon_dataset = Amazon(root=data_dir, name='Photo')
texas_dataset = WebKB(root=data_dir, name='Texas')
chameleon_dataset = FixedWikipediaNetwork(root=data_dir, name='Chameleon')
citeseer = Planetoid(root=data_dir, name='CiteSeer')
amazon_comp = Amazon(root=data_dir, name='Computers')
coauthor_cs = Coauthor(root=data_dir, name='CS')
cornell = WebKB(root=data_dir, name='Cornell')
wisconsin = WebKB(root=data_dir, name='Wisconsin')
actor = Actor(root=os.path.join(data_dir, 'actor'))
wikics = WikiCS(root=os.path.join(data_dir, 'wikics'))
squirrel = FixedWikipediaNetwork(root=data_dir, name='Squirrel')
roman_empire = HeterophilousGraphDataset(root=data_dir, name='Roman-empire')
amazon_ratings = HeterophilousGraphDataset(root=data_dir, name='Amazon-ratings')
minesweeper = HeterophilousGraphDataset(root=data_dir, name='Minesweeper', pre_transform=T.ToUndirected())
tolokers = HeterophilousGraphDataset(root=data_dir, name='Tolokers', pre_transform=T.ToUndirected())

names = ['Cora', 'PubMed', 'Amazon Photo', 'Texas', 'Chameleon', 'CiteSeer', 'Amazon Computers', 'Coauthor CS',
         'Cornell', 'Wisconsin', 'Actor', 'WikiCS', 'Squirrel', 'Roman Empire', 'Amazon Ratings', 'Minesweeper', 'Tolokers']
datasets = [cora_dataset, pubmed, amazon_dataset, texas_dataset, chameleon_dataset, citeseer, amazon_comp, coauthor_cs,
            cornell, wisconsin, actor, wikics, squirrel, roman_empire, amazon_ratings, minesweeper, tolokers]
graphs = [dataset[0] for dataset in datasets]
labels = [graph.y for graph in graphs]

# compute spectra

def compute_spectrum(edge_index, n):
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=n)
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=n)[0]
    evals, evecs = torch.linalg.eigh(L)
    return evals, evecs

# compute profile

def compute_slp(evals, evecs, labels, num_classes):
    N = labels.shape[0]

    Y = F.one_hot(labels, num_classes=num_classes).float()
    Y_tilde = Y - Y.mean(dim=0, keepdim=True)
    proj = (evecs.T @ Y_tilde) ** 2
    Y_norm = torch.norm(Y_tilde, dim=0) ** 2 + 1e-8
    pi_c = proj / Y_norm
    pi = pi_c.mean(dim=1)
    cdf = torch.cumsum(pi, dim=0)

    return cdf

results = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
masks_dir = Path(__file__).resolve().parents[1] / 'spatial' / 'masks'

# display name -> mask filename (mask filenames have no spaces / hyphens)
mask_name = {
    'Cora': 'Cora', 'PubMed': 'PubMed', 'Amazon Photo': 'Photo',
    'Texas': 'Texas', 'Chameleon': 'Chameleon', 'CiteSeer': 'CiteSeer',
    'Amazon Computers': 'Computers', 'Coauthor CS': 'CS',
    'Cornell': 'Cornell', 'Wisconsin': 'Wisconsin', 'Actor': 'Actor',
    'WikiCS': 'WikiCS', 'Squirrel': 'Squirrel',
    'Roman Empire': 'RomanEmpire', 'Amazon Ratings': 'AmazonRatings',
    'Minesweeper': 'Minesweeper', 'Tolokers': 'Tolokers',
}

for d, g, name in zip(datasets, graphs, names):
    C = d.num_classes
    n = g.num_nodes
    g = g.to(device)

    evals_f, evecs_f = compute_spectrum(g.edge_index, n)

    masks = torch.load(masks_dir / f'{mask_name[name]}.pt')
    train_mask = masks['train_mask'].to(device)
    sub_edge_index, _ = subgraph(train_mask, g.edge_index, num_nodes=n, relabel_nodes=True)
    sub_n = int(train_mask.sum().item())
    evals_s, evecs_s = compute_spectrum(sub_edge_index, sub_n)

    cdf_f = compute_slp(evals_f, evecs_f, g.y.to(evecs_f.device), C)
    cdf_s = compute_slp(evals_s, evecs_s, g.y[train_mask].to(evecs_s.device), C)

    results[name] = (evals_f, cdf_f, evals_s, cdf_s)
    print(name)

# plot

n_rows, n_cols = 3, 7
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3 * n_cols, 3 * n_rows),
                         sharex=True, sharey=True)
axes_flat = axes.flatten()

for ax, (name, (evals_f, cdf_f, evals_s, cdf_s)) in zip(axes_flat, results.items()):
    ax.step(evals_f.cpu().numpy(), cdf_f.cpu().numpy(),
            where='post', label='full', linewidth=1.5)
    ax.step(evals_s.cpu().numpy(), cdf_s.cpu().numpy(),
            where='post', label='train subgraph', linewidth=1.5, alpha=0.8)
    ax.set_title(name, fontsize=10)
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
plt.savefig('slp_comparison.png', dpi=150)
plt.show()







