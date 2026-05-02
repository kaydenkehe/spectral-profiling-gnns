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

## Sparsification Tradeoff Analysis

The `sparsify_tradeoff.py` script measures how Laplacians.jl graph sparsification affects SLP-CDF accuracy and speed using Lanczos approximations.

### Prerequisites

1. **Install Julia** and Laplacians.jl:
   ```bash
   # Install Julia (macOS with Homebrew)
   brew install julia
   
   # Install Laplacians.jl
   julia -e 'using Pkg; Pkg.add("Laplacians")'
   ```

2. **Update PATH** (if Julia not found):
   ```bash
   # Add Julia to PATH (for juliaup installations)
   echo 'export PATH="$HOME/.juliaup/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

### Usage

```bash
cd analysis

# Test single dataset with specific epsilon and Lanczos steps
python sparsify_tradeoff.py --datasets Photo --eps 0.5 --step-values 32

# Test multiple datasets with step sweep
python sparsify_tradeoff.py --datasets Cora PubMed --eps 0.3 0.6 --step-values 16 32 64

# Test all available datasets
python sparsify_tradeoff.py --eps 0.5 --step-values 64
```

**Outputs** (saved to `analysis/approximate_slp/`):
- `sparsify_tradeoff_YYYYMMDD-HHMMSS.csv` - Results table
- `sparsify_tradeoff_YYYYMMDD-HHMMSS.png` - Tradeoff plot  
- `sparsify_tradeoff_YYYYMMDD-HHMMSS_cdf.png` - CDF comparison plot

**Arguments**:
- `--datasets` - Dataset names (default: all with ground truth)
- `--eps` - Sparsification epsilon values (default: 0.3 0.6 0.9)
- `--step-values` - Lanczos step counts (single or multiple, default: 16 32 64)
- `--julia-bin` - Julia executable path (default: "julia")

## Approximate Metrics Generation

The `approx_metrics.py` script adds approximate SLP-CDF entries to `metrics.json` for larger datasets where exact eigen-decomposition is expensive.

### Usage

```bash
cd analysis

# Add approximate metrics for specific datasets
python approx_metrics.py --datasets PubMed WikiCS --steps 64

# Add sparsified approximate metrics  
python approx_metrics.py --datasets Questions Physics --epsilon 0.6 --steps 64

# Add missing datasets only (skip existing entries)
python approx_metrics.py --datasets AmazonRatings RomanEmpire --epsilon 0.6 --steps 64 --missing-only

# Replace entire metrics file
python approx_metrics.py --datasets all --steps 64 --replace
```

**Arguments**:
- `--datasets` - Dataset names (default: all available)
- `--epsilon` - Optional sparsification epsilon before SLP
- `--steps` - Lanczos steps (default: 64)
- `--grid-size` - Lambda grid points in output (default: 50)
- `--missing-only` - Skip datasets already in metrics.json
- `--replace` - Replace metrics.json instead of updating

Each script expects to be run from its own directory (paths are relative to
`../graph_data`).
