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

# compute homophily

def compute_homophily(edge_index, labels):
    N = labels.shape[0]
    num_classes = labels.max().item() + 1

    A = to_dense_adj(edge_index, max_num_nodes=N)[0]

    src, dst = A.nonzero(as_tuple=True)
    idx = labels[src] * num_classes + labels[dst]
    co_occurrence = torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes).float()
    homophily = co_occurrence.diag() / co_occurrence.sum(dim=1).clamp(min=1)

#     co_occurrence = torch.zeros((num_classes, num_classes))
#     for i in range(N):
#         for j in range(N):
#             if A[i, j] > 0:
#                 co_occurrence[labels[i], labels[j]] += 1
# 
#     homophily = torch.zeros(num_classes)
#     for c in range(num_classes):
#         if co_occurrence[c].sum() > 0:
#             homophily[c] = co_occurrence[c, c] / co_occurrence[c].sum()

    return homophily

# compute and write to csv 

import csv

summary_rows = []
detail_rows = []

for name, graph in zip(names, graphs):
    homophily = compute_homophily(graph.edge_index, graph.y)
    mean_h = homophily.mean().item()
    print(f'{name} homophily: {mean_h:.4f}')

    summary_rows.append({
        'dataset': name,
        'num_classes': homophily.shape[0],
        'num_nodes': graph.num_nodes,
        'num_edges': graph.edge_index.shape[1],
        'mean_homophily': mean_h,
    })
    for c, h in enumerate(homophily.tolist()):
        detail_rows.append({'dataset': name, 'class': c, 'homophily': h})

# deprecated - done in metrics.py
# summary: one row per dataset
# with open('homophily_summary.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
#     writer.writeheader()
#     writer.writerows(summary_rows)

# detail: one row per (dataset, class)
with open('homophily_per_class.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
    writer.writeheader()
    writer.writerows(detail_rows)













