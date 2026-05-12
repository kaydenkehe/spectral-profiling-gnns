# Architecture sweep (spatial masks)

- One folder per dataset with `summary.csv`, matching `spectral/train_spectral_massive.py` outputs.
- **CS** is merged from multiple Slurm runs (GPRGNN/Cheb K=4, then BernNet, JacobiConv, Cheb K=10 patches under `spectral_massive_spatial_masks_cs_missing_b10_*`).
- **Questions** is taken from the `retry_anygpu_b15` run (complete 1440 rows); the May 11 retry folder only had a partial Questions file.
- **Physics** is under `partial/Physics/` (OOM; incomplete).
- `summary_all.csv` concatenates all per-dataset summaries (excludes `partial/Physics`).

Git revision: `73f0911a8ccfd6d72bae965cafe020d29eb1e51d
`
