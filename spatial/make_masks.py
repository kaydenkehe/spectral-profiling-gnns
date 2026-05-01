from pathlib import Path
import torch
from datasets import build_datasets

train_r = 0.8
val_r = 0.1
seed = 0

masks_dir = Path(__file__).resolve().parent / 'masks'
masks_dir.mkdir(exist_ok=True)

datasets = build_datasets()
for name, d in datasets:
    g = d[0]
    n = g.num_nodes

    torch.manual_seed(seed)
    perm = torch.randperm(n)
    n_tr = int(train_r * n)
    n_val = int(val_r * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr:n_tr + n_val]] = True
    test_mask[perm[n_tr + n_val:]] = True

    out_path = masks_dir / f'{name}.pt'
    torch.save(
        {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask},
        out_path,
    )

