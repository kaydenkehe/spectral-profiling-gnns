import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fagcn_harness import train_sweep as train_fagcn
from gcn_harness import train_sweep as train_gcn
from h2gcn_harness import train_sweep as train_h2gcn
from hfgcn_harness import train_sweep as train_hfgcn
from mlp_harness import train_sweep as train_mlp


sbm_dir = Path('/n/home06/drooryck/spectral-profling-gnns/graph_data/sbms')
out_csv = Path('/n/home06/drooryck/spectral-profling-gnns/spatial/sbms/sbm_model_results.csv')
out_table = Path('/n/home06/drooryck/spectral-profling-gnns/spatial/sbms/sbm_model_table.md')

Dataset = type('Dataset', (list,), {})
datasets = []
for path in sorted(sbm_dir.glob('*.pt')):
    graph = torch.load(path, weights_only=False)
    dataset = Dataset([graph])
    dataset.num_classes = int(graph.num_classes)
    datasets.append((f'SBM_{path.stem}', dataset))

if not datasets:
    raise FileNotFoundError('Run spatial/sbms/generate_sbms.py first.')

runs = [
    ('mlp', train_mlp, {'depths': (2,), 'hidden_dims': (32,), 'n_runs': 3}),
    ('gcn', train_gcn, {'depths': (2,), 'hidden_dims': (32,), 'n_runs': 3}),
    ('hfgcn', train_hfgcn, {'depths': (2,), 'hidden_dims': (32,), 'n_runs': 3}),
    ('fagcn', train_fagcn, {
        'depths': (2,),
        'hidden_dims': (32,),
        'eps_values': (0.3,),
        'n_runs': 3,
    }),
    ('h2gcn', train_h2gcn, {
        'depths': (2,),
        'hidden_dims': (64,),
        'dropouts': (0.5,),
        'weight_decays': (5e-4,),
        'use_relu': (True,),
        'n_runs': 3,
    }),
]

rows = []
for model_name, train_sweep, kwargs in runs:
    print(f'training {model_name}', flush=True)
    results = train_sweep(datasets, **kwargs)
    for dataset_name, configs in results.items():
        family = dataset_name.removeprefix('SBM_').rsplit('_seed', 1)[0]
        for config, (accs, _) in configs.items():
            for seed, acc in enumerate(accs):
                rows.append({
                    'dataset': dataset_name,
                    'family': family,
                    'model': model_name,
                    'config': str(config),
                    'seed': seed,
                    'test_acc': f'{acc:.4f}',
                })

with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['dataset', 'family', 'model', 'config', 'seed', 'test_acc'],
    )
    writer.writeheader()
    writer.writerows(rows)

accs_by_family_model = defaultdict(list)
for row in rows:
    accs_by_family_model[(row['family'], row['model'])].append(float(row['test_acc']))

families = ['homophilic', 'paired_heterophilic', 'er_no_alignment', 'mixed']
models = ['mlp', 'gcn', 'hfgcn', 'fagcn', 'h2gcn']
lines = [
    '| family | mlp | gcn | hfgcn | fagcn | h2gcn |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
]
for family in families:
    cells = [family]
    for model in models:
        accs = accs_by_family_model[(family, model)]
        cells.append(f'{np.mean(accs):.3f} +/- {np.std(accs):.3f}' if accs else 'n/a')
    lines.append('| ' + ' | '.join(cells) + ' |')

out_table.write_text('\n'.join(lines) + '\n')
print(f'wrote {out_csv}')
print(f'wrote {out_table}')
