import csv
import json
from datetime import datetime
from pathlib import Path
from datasets import build_datasets

# model selection

model = 'fagcn'
if model == 'gcn':
    from gcn_harness import train_sweep
elif model == 'fagcn':
    from fagcn_harness import train_sweep
elif model == 'mlp':
    from mlp_harness import train_sweep
elif model == 'hfgcn':
    from hfgcn_harness import train_sweep
elif model == 'h2gcn':
    from h2gcn_harness import train_sweep

# setup dir

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = Path('runs') / f"{model}_{timestamp}"
curves_dir = run_dir / 'curves'
curves_dir.mkdir(parents=True, exist_ok=True)

# train

datasets = build_datasets()
results = train_sweep(datasets)
rows = []

# process results

for dataset_name, configs in results.items():
    for config_key, (accs, val_histories) in configs.items():
        depth = config_key[0]
        hidden_dim = config_key[1]
        eps = config_key[2] if model == 'fagcn' else ''
        if model == 'h2gcn':
            dropout, wd, relu_flag = config_key[2], config_key[3], config_key[4]
        else:
            dropout, wd, relu_flag = '', '', ''

        for seed, (acc, history) in enumerate(zip(accs, val_histories)):
            eps_str = f"{eps}" if eps != '' else 'na'
            if model == 'h2gcn':
                extras = f"_d{dropout}_wd{wd}_relu{int(relu_flag)}"
            else:
                extras = ''
            row_id = f"{dataset_name}_L{depth}_H{hidden_dim}_eps{eps_str}{extras}_s{seed}"

            rows.append({
                'id': row_id,
                'dataset': dataset_name,
                'model': model,
                'depth': depth,
                'hidden': hidden_dim,
                'eps': eps_str,
                'dropout': dropout,
                'wd': wd,
                'relu': relu_flag,
                'seed': seed,
                'test_acc': f"{acc:.4f}"
            })

            # val history to csv
            with open (curves_dir / f"{row_id}.csv", 'w') as f:
                cw = csv.writer(f)
                cw.writerow(['epoch', 'val_acc'])
                for epoch, val_acc in enumerate(history):
                    cw.writerow([epoch, f"{val_acc:.4f}"])

# summary csv
fieldnames = ['id', 'dataset', 'model', 'depth', 'hidden', 'eps', 'dropout', 'wd', 'relu', 'seed', 'test_acc']
with open(run_dir / 'summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# manifest
with open(run_dir / 'config.json', 'w') as f:
    json.dump({'model': model, 'timestamp': timestamp, 'n_rows': len(rows)}, f, indent=2)

