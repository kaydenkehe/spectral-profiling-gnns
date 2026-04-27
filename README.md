# Spectral Label Profiles for GNN Filter Selection

CS 2252 final project. We test whether the **spectral label profile**, the
CDF of label energy across the normalized-Laplacian spectrum of an input graph,
predicts which GNN architecture (GCN, FAGCN, MLP, etc.) gives the best 
node-classification accuracy. The pipeline trains each architecture on a pool of
standard node-classification graphs, then compares the empirically best
architecture against what the spectral profile prescribes.

## Layout

- `spatial/` — training harnesses for the spatial models we sweep.
  - `datasets.py` — loads the graph benchmarks via PyG.
  - `gcn_harness.py`, `fagcn_harness.py`, `mlp_harness.py` — model, and `train_sweep`
    over (depth, hidden dim, eps for FAGCN) with 3 seeds and val-based early
    stopping.
  - `train_spatial.py` — entry point. Set `model = 'gcn' | 'fagcn' | 'mlp'`,
    runs the sweep, writes per-config val curves and a `summary.csv` to
    `spatial/runs/<model>_<timestamp>/`.
  - `runs/collate.py` — aggregates the run directories listed at the top
    of the file into `runs/best_spatial.csv` (best mean config per
    dataset × model).
- `analysis/` — spectral and homophily diagnostics.
  - `spectrum_subgraph.py` — compares the SLP of the full graph vs. a random
    subgraph; writes `slp_comparison.png`.
  - `homophily.py` — per-class edge homophily; writes `homophily_per_class.csv`.
  - `metrics.py` — for every dataset, computes homophily and the SLP sampled on
    a 50-point lambda grid; writes `metrics.json` and `metrics.png`.

## Usage

```bash
pip install -r requirements.txt

# Download datasets
cd spatial && python datasets.py

# Train a model across the dataset pool
cd spatial && python train_spatial.py        # edit `model =` at the top first

# After running gcn / fagcn / mlp, aggregate best configs
cd spatial/runs && python collate.py

# Spectral + homophily metrics for every dataset
cd analysis && python metrics.py              # -> metrics.json, metrics.png
python homophily.py                            # -> homophily_per_class.csv
python spectrum_subgraph.py                    # -> slp_comparison.png
```

Each script expects to be run from its own directory (paths are relative to
`../graph_data`).
