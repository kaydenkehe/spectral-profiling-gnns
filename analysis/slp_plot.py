import os
from torch_geometric.datasets import (   
    Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork,
    Actor, HeterophilousGraphDataset,
)
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj, subgraph
import torch
import matplotlib.pyplot as plt


# collect graphs

data_dir = './graph_data'

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


# compute spectra

def compute_spectrum(edge_index, n):
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=n)
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=n)[0]
    evals, evecs = torch.linalg.eigh(L)
    return evals, evecs

# compute profile

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

results = {}

for d, g, name in zip(datasets, graphs, names):
    C = d.num_classes
    n = g.num_nodes

    evals_f, evecs_f = compute_spectrum(g.edge_index, n)
    cdf_f = compute_slp(evecs_f, g.y, C)

    results[name] = (evals_f, cdf_f)

# plot

n_rows, n_cols = 2, 7
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3 * n_cols, 3 * n_rows),
                         sharex=True, sharey=True)
axes_flat = axes.flatten()

for ax, (name, (evals_f, cdf_f)) in zip(axes_flat, results.items()):
    ax.step(evals_f.cpu().numpy(), cdf_f.cpu().numpy(),
            where='post', label='full', linewidth=1.5)
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
plt.savefig('all_slp.png', dpi=150)
plt.show()




