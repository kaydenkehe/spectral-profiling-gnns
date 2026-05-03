from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, stochastic_blockmodel_graph, to_dense_adj


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "graph_data" / "sbms_matched"
OUT_DIR = Path(__file__).resolve().parent

K = 4
BLOCK = 250
SEEDS = range(5)
TARGET_H = 0.42
AVG_DEG = 20
HIGH = 0.06
LOW = 0.006


def probs():
    p_in = AVG_DEG * TARGET_H / BLOCK
    p_out = AVG_DEG * (1 - TARGET_H) / (BLOCK * (K - 1))

    uniform = torch.full((K, K), p_out)
    uniform.fill_diagonal_(p_in)

    mixed = torch.full((K, K), LOW)
    mixed[0, 0] = mixed[1, 1] = HIGH
    mixed[2, 3] = mixed[3, 2] = HIGH

    return {"uniform_mid": uniform, "mixed": mixed}


def make_graph(family, P, seed):
    torch.manual_seed(seed)
    y = torch.cat([torch.full((BLOCK,), c, dtype=torch.long) for c in range(K)])
    edge_index = stochastic_blockmodel_graph([BLOCK] * K, P, directed=False)
    x = 1.5 * F.one_hot(y, num_classes=K).float() + torch.randn(y.numel(), K)
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=y.numel())
    data.family = family
    data.seed = seed
    data.num_classes = K
    data.block_probs = P
    return data


def homophily(data):
    src, dst = data.edge_index
    return (data.y[src] == data.y[dst]).float().mean().item()


def slp(data):
    edge_index, edge_weight = get_laplacian(data.edge_index, normalization="sym", num_nodes=data.num_nodes)
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=data.num_nodes)[0]
    evals, evecs = torch.linalg.eigh(L)

    Y = F.one_hot(data.y, num_classes=K).float()
    Y = Y - Y.mean(dim=0, keepdim=True)
    energy = ((evecs.T @ Y) ** 2 / (Y.norm(dim=0) ** 2 + 1e-8)).mean(dim=1)
    return evals.numpy(), torch.cumsum(energy, dim=0).numpy()


def cdf_at(evals, cdf, threshold):
    idx = np.searchsorted(evals, threshold, side="right") - 1
    return float(cdf[idx]) if idx >= 0 else 0.0


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    graphs = {family: [] for family in probs()}
    rows = ["family,seed,homophily,num_edges,pi_0.5,pi_1.0,pi_1.5"]

    for family, P in probs().items():
        for seed in SEEDS:
            data = make_graph(family, P, seed)
            torch.save(data, DATA_DIR / f"{family}_seed{seed}.pt")
            evals, cdf = slp(data)
            graphs[family].append((data, evals, cdf))
            rows.append(
                f"{family},{seed},{homophily(data):.4f},{data.edge_index.size(1)},"
                f"{cdf_at(evals, cdf, 0.5):.4f},{cdf_at(evals, cdf, 1.0):.4f},{cdf_at(evals, cdf, 1.5):.4f}"
            )

    (OUT_DIR / "matched_homophily_results.csv").write_text("\n".join(rows) + "\n")

    fig, axes = plt.subplots(2, len(SEEDS), figsize=(3 * len(SEEDS), 5), sharex=True, sharey=True)
    for r, (family, items) in enumerate(graphs.items()):
        for c, (data, evals, cdf) in enumerate(items):
            ax = axes[r, c]
            ax.step(evals, cdf, where="post", linewidth=1.4)
            ax.step(evals, np.arange(1, len(evals) + 1) / len(evals), where="post", color="0.7", linestyle="--", linewidth=0.8)
            ax.set_title(f"seed={data.seed}, h={homophily(data):.2f}", fontsize=9)
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(f"{family}\n$\\Pi(\\lambda^*)$")
            if r == 1:
                ax.set_xlabel("$\\lambda^*$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "matched_homophily_slps.png", dpi=150)

    print(f"wrote {OUT_DIR / 'matched_homophily_results.csv'}")
    print(f"wrote {OUT_DIR / 'matched_homophily_slps.png'}")


if __name__ == "__main__":
    main()
