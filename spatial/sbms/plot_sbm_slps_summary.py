from pathlib import Path

import matplotlib.pyplot as plt
import torch

from plot_sbm_slps import edge_homophily, spectral_label_profile


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "graph_data" / "sbms"
OUT_PATH = Path(__file__).resolve().parent / "sbm_slps_seed0.png"
SEED = 0

PANELS = [
    ("homophilic", "homophilic"),
    ("paired_heterophilic", "paired heterophilic"),
    ("er_no_alignment", "erdos renyi"),
    ("mixed", "mixed SBM"),
]


fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharex=True, sharey=True)
axes = axes.ravel()

for ax, (family, title) in zip(axes, PANELS):
    path = DATA_DIR / f"{family}_seed{SEED}.pt"
    data = torch.load(path, weights_only=False)
    evals, cdf, random_label_cdf = spectral_label_profile(data)
    h = edge_homophily(data.edge_index, data.y)

    ax.step(
        evals,
        random_label_cdf,
        where="post",
        color="0.65",
        linestyle="--",
        linewidth=1.4,
        label="random-label baseline",
    )
    ax.step(
        evals,
        cdf,
        where="post",
        color="#1f77b4",
        linewidth=2.2,
        label="SLP",
    )
    ax.set_title(f"{title}\n$h={h:.2f}$", fontsize=11)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)

for ax in axes:
    ax.set_xlabel(r"normalized Laplacian frequency $\lambda$")
axes[0].set_ylabel(r"cumulative label energy $\Pi(\lambda)$")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(OUT_PATH, dpi=200)
print(f"wrote {OUT_PATH}")
