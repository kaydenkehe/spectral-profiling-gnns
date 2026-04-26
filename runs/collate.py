from pathlib import Path
import pandas as pd

run_dirs = [
    'gcn_20260426-071840',
    'fagcn_20260426-203611'
]

runs_root = Path(__file__).parent

# get results
frames = []
for d in run_dirs:
    frames.append(pd.read_csv(runs_root / d / 'summary.csv'))
df = pd.concat(frames, ignore_index=True)

# aggregate results
group_cols = ['dataset', 'model', 'depth', 'hidden', 'eps']
agg = (
    df.groupby(group_cols)['test_acc']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)

# find best runs for each dataset and model
best = agg.loc[agg.groupby(['dataset', 'model'])['mean'].idxmax()]
best = best.sort_values(['dataset', 'model']).reset_index(drop=True)

best.to_csv(runs_root / 'best_spatial.csv', index=False)

