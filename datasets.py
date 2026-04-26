import os
from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork,
    Actor, HeterophilousGraphDataset,
)
import torch_geometric.transforms as T

data_dir = './graph_data'
os.makedirs(data_dir, exist_ok=True)

def build_datasets():
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

    return [
        ('Cora', cora_dataset),
        ('Photo', amazon_dataset),
        ('Texas', texas_dataset),
        ('Chameleon', chameleon_dataset),
        ('CiteSeer', citeseer),
        ('Computers', amazon_comp),
        ('CS', coauthor_cs),
        ('Cornell', cornell),
        ('Wisconsin', wisconsin),
        ('Actor', actor),
        ('Squirrel', squirrel),
        ('Minesweeper', minesweeper),
        ('Tolokers', tolokers)
    ]

