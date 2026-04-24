import os
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, WebKB, WikipediaNetwork
from torch_geometric.utils import get_laplacian, to_dense_adj, subgraph
import torch
import matplotlib.pyplot as plt

torch.manual_seed(777)
data_dir = './graph_data'
os.makedirs(data_dir, exist_ok=True)

# collect graphs

cora_dataset = Planetoid(root=data_dir, name='Cora')
# pubmed_dataset = Planetoid(root=data_dir, name='PubMed')
amazon_dataset = Amazon(root=data_dir, name='Photo')
texas_dataset = WebKB(root=data_dir, name='Texas')
chameleon_dataset = WikipediaNetwork(root=data_dir, name='Chameleon')

names = ['Cora', 'Amazon Photo', 'Texas', 'Chameleon']#, 'PubMed']
datasets = [cora_dataset, amazon_dataset, texas_dataset, chameleon_dataset]#, pubmed_dataset]
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

for d, g, name in zip(datasets, graphs, names):
    C = d.num_classes
    n = g.num_nodes
    evals_f, evecs_f = compute_spectrum(g.edge_index, n)

    sub_n = int(0.9 * n)
    subset_nodes = torch.randperm(n)[:sub_n]
    sub_edge_index, _ = subgraph(subset_nodes, g.edge_index, num_nodes=n, relabel_nodes=True)
    evals_s, evecs_s = compute_spectrum(sub_edge_index, sub_n)

    cdf_f = compute_slp(evals_f, evecs_f, g.y, C)
    cdf_s = compute_slp(evals_s, evecs_s, g.y[subset_nodes], C)

    results[name] = (evals_f, cdf_f, evals_s, cdf_s)

# plot

fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.5), sharey=True)
if len(results) == 1:
    axes = [axes]

for ax, (name, (evals_f, cdf_f, evals_s, cdf_s)) in zip(axes, results.items()):
    ax.step(evals_f.cpu().numpy(), cdf_f.cpu().numpy(),
            where='post', label='full', linewidth=1.5)
    ax.step(evals_s.cpu().numpy(), cdf_s.cpu().numpy(),
            where='post', label='90% subgraph', linewidth=1.5, alpha=0.8)

    ax.set_title(name)
    ax.set_xlabel(r'$\lambda^*$')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

axes[0].set_ylabel(r'$\Pi(\lambda^*)$')
axes[-1].legend(loc='lower right', fontsize=8)
fig.tight_layout()
plt.savefig('slp_comparison.png', dpi=150)
plt.show()







