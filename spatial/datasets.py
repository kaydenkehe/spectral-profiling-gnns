import os
import sys
from pathlib import Path

# Add parent directory to path to import from common
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_geometric.datasets import (
    Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork,
    Actor, HeterophilousGraphDataset, WikiCS,
)
import torch_geometric.transforms as T
from common.datasets import FixedWikipediaNetwork

data_dir = str(Path(__file__).resolve().parents[1] / 'graph_data')
os.makedirs(data_dir, exist_ok=True)

def build_datasets():
    cora_dataset = Planetoid(root=data_dir, name='Cora')
    pubmed = Planetoid(root=data_dir, name='PubMed')
    amazon_dataset = Amazon(root=data_dir, name='Photo')
    texas_dataset = WebKB(root=data_dir, name='Texas')
    chameleon_dataset = FixedWikipediaNetwork(root=data_dir, name='Chameleon')
    citeseer = Planetoid(root=data_dir, name='CiteSeer')
    amazon_comp = Amazon(root=data_dir, name='Computers')
    coauthor_cs = Coauthor(root=data_dir, name='CS')
#     coauthor_physics = Coauthor(root=data_dir, name='Physics')
    cornell = WebKB(root=data_dir, name='Cornell')
    wisconsin = WebKB(root=data_dir, name='Wisconsin')
    actor = Actor(root=os.path.join(data_dir, 'actor'))
    wikics = WikiCS(root=os.path.join(data_dir, 'wikics'))
    squirrel = FixedWikipediaNetwork(root=data_dir, name='Squirrel')
    roman_empire = HeterophilousGraphDataset(root=data_dir, name='Roman-empire')
    amazon_ratings = HeterophilousGraphDataset(root=data_dir, name='Amazon-ratings')
    minesweeper = HeterophilousGraphDataset(root=data_dir, name='Minesweeper', pre_transform=T.ToUndirected())
    tolokers = HeterophilousGraphDataset(root=data_dir, name='Tolokers', pre_transform=T.ToUndirected())
#     questions = HeterophilousGraphDataset(root=data_dir, name='Questions', pre_transform=T.ToUndirected())

    return [
        ('Cora', cora_dataset),
        ('PubMed', pubmed),
        ('Photo', amazon_dataset),
        ('Texas', texas_dataset),
        ('Chameleon', chameleon_dataset),
        ('CiteSeer', citeseer),
        ('Computers', amazon_comp),
        ('CS', coauthor_cs),
#         ('Physics', coauthor_physics),
        ('Cornell', cornell),
        ('Wisconsin', wisconsin),
        ('Actor', actor),
        ('WikiCS', wikics),
        ('Squirrel', squirrel),
        ('RomanEmpire', roman_empire),
        ('AmazonRatings', amazon_ratings),
        ('Minesweeper', minesweeper),
        ('Tolokers', tolokers),
#         ('Questions', questions),
]

