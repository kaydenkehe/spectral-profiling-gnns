import os
import numpy as np
import urllib.request
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected


class FixedWikipediaNetwork(Dataset):
    """Fixed versions of Squirrel and Chameleon datasets from Yandex Research.
    
    These datasets address issues with duplicate nodes found in the original
    WikipediaNetwork datasets. Source: https://github.com/yandex-research/heterophilous-graphs
    """
    
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel'], f"Dataset {name} not available"
        self.url = f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/data/{self.name}_filtered.npz"
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return [f"{self.name}_filtered.npz"]
    
    @property
    def processed_file_names(self):
        return [f"data_{self.name}.pt"]
    
    def download(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(self.url, file_path)
    
    def process(self):
        data = np.load(os.path.join(self.raw_dir, self.raw_file_names[0]))
        
        node_features = torch.from_numpy(data['node_features']).float()
        node_labels = torch.from_numpy(data['node_labels']).long()
        edges = torch.from_numpy(data['edges']).long().t().contiguous()
        edges = to_undirected(edges, num_nodes=node_features.size(0))
        
        # Use the first split (index 0)
        train_mask = torch.from_numpy(data['train_masks'][0]).bool()
        val_mask = torch.from_numpy(data['val_masks'][0]).bool()
        test_mask = torch.from_numpy(data['test_masks'][0]).bool()
        
        pyg_data = Data(
            x=node_features,
            y=node_labels,
            edge_index=edges,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        if self.pre_transform is not None:
            pyg_data = self.pre_transform(pyg_data)
        
        torch.save(pyg_data, os.path.join(self.processed_dir, self.processed_file_names[0]))
    
    def len(self):
        return 1
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]), weights_only=False)
        return data if self.transform is None else self.transform(data)
