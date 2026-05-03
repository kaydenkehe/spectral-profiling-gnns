from pathlib import Path
import pandas as pd

run_dirs = [
    'gcn_20260502-193049',
    'fagcn_20260502-193533',
    'mlp_20260502-193306',
    'hfgcn_20260502-193406',
    'h2gcn_20260502-201718'
]

runs_root = Path(__file__).parent

# get results
frames = []
for d in run_dirs:
    frames.append(pd.read_csv(runs_root / d / 'summary.csv'))
df = pd.concat(frames, ignore_index=True)

# aggregate results
group_cols = ['dataset', 'model', 'depth', 'hidden', 'eps', 'dropout', 'wd', 'relu']
agg = (
    df.groupby(group_cols, dropna=False)['test_acc']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)

# find best runs for each dataset and model
best = agg.loc[agg.groupby(['dataset', 'model'])['mean'].idxmax()]
best = best.sort_values(['dataset', 'model']).reset_index(drop=True)

best.to_csv(runs_root / 'best_spatial.csv', index=False)

# find best (model, config) per dataset — winner per task
best_per_task = agg.loc[agg.groupby('dataset')['mean'].idxmax()]
best_per_task = best_per_task.sort_values('dataset').reset_index(drop=True)

best_per_task.to_csv(runs_root / 'best_per_task.csv', index=False)

