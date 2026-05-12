"""Replot metrics.png from cached spectra (no spectrum recomputation)."""
import json
import pickle
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw_doodle(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    blobs = [
        ('L', 0.35, 0.65, '#70df2f'),
        ('K', 0.65, 0.65, '#d82f68'),
        ('E', 0.35, 0.35, '#24d63a'),
        ('D', 0.65, 0.35, '#2299d6'),
    ]
    radius = 0.22
    for letter, x, y, color in blobs:
        circ = mpatches.Circle((x, y), radius,
                               facecolor=color, edgecolor='black',
                               linewidth=1.8, zorder=2)
        ax.add_patch(circ)
        ax.text(x, y, letter, ha='center', va='center',
                fontsize=22, color='black', zorder=3)


with open('metrics.json') as f:
    results = json.load(f)

cache_path = 'spectra_cache_undirected_v2.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        full = pickle.load(f)
else:
    full = None

n_rows, n_cols = 3, 6
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.5 * n_cols, 3.2 * n_rows),
                         sharex=False, sharey=False)
axes_flat = axes.flatten()

for ax, (name, r) in zip(axes_flat, results.items()):
    if full is not None and name in full:
        evals_np, cdf_np = full[name]
    else:
        evals_np, cdf_np = r['eigenvalues'], r['cdf']
    ax.step(evals_np, cdf_np, where='post', label='full', linewidth=1.5)
    ax.set_title(f'{name} (h={r["homophily"]:.2f})', fontsize=10)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

for i, ax in enumerate(axes_flat[len(results):]):
    if i == 0:
        draw_doodle(ax)
    else:
        ax.set_visible(False)

# only edge plots get tick labels and axis labels
for ax in axes_flat:
    ax.tick_params(labelbottom=False, labelleft=False)

# bottom row of actual data plots gets x-axis labels
# (row 2, cols 0-4 -- col 5 is the doodle)
for ax in axes[-1, :-1]:
    ax.tick_params(labelbottom=True)
    ax.set_xlabel(r'$\lambda^*$')

# left column gets y-axis labels
for ax in axes[:, 0]:
    ax.tick_params(labelleft=True)
    ax.set_ylabel(r'$\Pi(\lambda^*)$')

axes_flat[n_cols - 1].legend(loc='lower right', fontsize=8)
fig.tight_layout()
plt.savefig('metrics.png', dpi=150)
print('wrote metrics.png')
