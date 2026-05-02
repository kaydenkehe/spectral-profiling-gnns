from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "graph_data" / "sbms"
OUT_PATH = Path(__file__).resolve().parent / "sbm_slps.png"
FAMILIES = ["homophilic", "paired_heterophilic", "er_no_alignment", "mixed"]


def edge_homophily(edge_index, y):
    src, dst = edge_index
    return (y[src] == y[dst]).float().mean().item()


def spectral_label_profile(data):
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        normalization="sym",
        num_nodes=data.num_nodes,
    )
    laplacian = to_dense_adj(
        edge_index,
        edge_attr=edge_weight,
        max_num_nodes=data.num_nodes,
    )[0]
    evals, evecs = torch.linalg.eigh(laplacian)

    y_onehot = F.one_hot(data.y, num_classes=data.num_classes).float()
    y_centered = y_onehot - y_onehot.mean(dim=0, keepdim=True)
    proj = (evecs.T @ y_centered) ** 2
    class_norm = torch.norm(y_centered, dim=0) ** 2 + 1e-8
    energy = (proj / class_norm).mean(dim=1)
    cdf = torch.cumsum(energy, dim=0)
    return evals.numpy(), cdf.numpy()


def load_graphs():
    graphs = {family: [] for family in FAMILIES}
    for path in sorted(DATA_DIR.glob("*.pt")):
        data = torch.load(path, weights_only=False)
        graphs[data.family].append(data)
    return graphs


def main():
    graphs = load_graphs()
    n_cols = max(len(items) for items in graphs.values())
    fig, axes = plt.subplots(
        len(FAMILIES),
        n_cols,
        figsize=(3 * n_cols, 2.5 * len(FAMILIES)),
        sharex=True,
        sharey=True,
    )

    diag_x = np.linspace(0, 2, 100)
    for row, family in enumerate(FAMILIES):
        for col, data in enumerate(graphs[family]):
            ax = axes[row, col]
            evals, cdf = spectral_label_profile(data)
            h = edge_homophily(data.edge_index, data.y)
            ax.step(evals, cdf, where="post", linewidth=1.4)
            ax.plot(diag_x, diag_x / 2, color="0.7", linestyle="--", linewidth=0.8)
            ax.set_title(f"seed={data.seed}, h={h:.2f}", fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 1.02)
            if col == 0:
                ax.set_ylabel(f"{family}\n$\\Pi(\\lambda^*)$")
            if row == len(FAMILIES) - 1:
                ax.set_xlabel("$\\lambda^*$")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
