from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import stochastic_blockmodel_graph


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "graph_data" / "sbms"

N_CLASSES = 4
BLOCK_SIZE = 250
FEATURE_DIM = 16
SEEDS = range(5)

HIGH = 0.06
LOW = 0.006
ER_P = 0.02


def matrices():
    hom = torch.full((N_CLASSES, N_CLASSES), LOW)
    hom.fill_diagonal_(HIGH)

    hetero = torch.full((N_CLASSES, N_CLASSES), LOW)
    hetero[0, 1] = hetero[1, 0] = HIGH
    hetero[2, 3] = hetero[3, 2] = HIGH

    er = torch.full((N_CLASSES, N_CLASSES), ER_P)

    mixed = torch.full((N_CLASSES, N_CLASSES), LOW)
    mixed[0, 0] = mixed[1, 1] = HIGH
    mixed[2, 3] = mixed[3, 2] = HIGH

    return {
        "homophilic": hom,
        "paired_heterophilic": hetero,
        "er_no_alignment": er,
        "mixed": mixed,
    }


def make_graph(family, probs, seed):
    torch.manual_seed(seed)
    block_sizes = [BLOCK_SIZE] * N_CLASSES
    y = torch.cat([
        torch.full((size,), label, dtype=torch.long)
        for label, size in enumerate(block_sizes)
    ])

    edge_index = stochastic_blockmodel_graph(
        block_sizes=block_sizes,
        edge_probs=probs,
        directed=False,
    )

    means = F.one_hot(torch.arange(N_CLASSES), num_classes=FEATURE_DIM).float() * 1.5
    x = means[y] + torch.randn(y.numel(), FEATURE_DIM)

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=y.numel())
    data.family = family
    data.seed = seed
    data.block_probs = probs
    data.num_classes = N_CLASSES
    return data


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for family, probs in matrices().items():
        for seed in SEEDS:
            data = make_graph(family, probs, seed)
            torch.save(data, OUT_DIR / f"{family}_seed{seed}.pt")
            print(f"wrote {family}_seed{seed}.pt edges={data.edge_index.size(1)}")


if __name__ == "__main__":
    main()
