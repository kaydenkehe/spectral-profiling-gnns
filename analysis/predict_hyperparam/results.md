# Hyperparameter Prediction Results

This experiment asks whether graph-level features can choose good spatial GNN
hyperparameters on a held-out dataset.

## Setup

- Features:
  - `homophily`: scalar class-balanced homophily.
  - `slp`: spectral label profile split into four frequency bands.
  - `homophily+slp`: both feature sets together.
- Baseline:
  - `default`: choose what works best on average across the training datasets.
- Evaluation:
  - Leave-one-dataset-out.
  - For each held-out dataset, choose hyperparameters using the other datasets.
  - Score by regret:

```text
regret = oracle accuracy - selected accuracy
```

Lower regret is better. A regret of `0` means oracle-level performance.

## Headline Results

### Full Config Selection

| Selector | Mean Regret | Median Regret | Exact Match |
|---|---:|---:|---:|
| default | 0.0156 | 0.0085 | 0.250 |
| homophily | 0.0200 | 0.0052 | 0.327 |
| homophily+slp | 0.0211 | 0.0073 | 0.327 |
| slp | 0.0211 | 0.0073 | 0.308 |

### Depth Selection

| Selector | Mean Regret | Median Regret | Exact Match |
|---|---:|---:|---:|
| homophily | 0.0055 | 0.0000 | 0.596 |
| homophily+slp | 0.0068 | 0.0000 | 0.615 |
| slp | 0.0069 | 0.0000 | 0.577 |
| default | 0.0079 | 0.0004 | 0.462 |

### FAGCN Epsilon Selection

| Selector | Mean Regret | Median Regret | Exact Match |
|---|---:|---:|---:|
| default | 0.0014 | 0.0000 | 0.538 |
| homophily+slp | 0.0015 | 0.0000 | 0.538 |
| slp | 0.0018 | 0.0011 | 0.462 |
| homophily | 0.0038 | 0.0016 | 0.385 |

## Interpretation

On the existing 13 real datasets, SLP does not outperform homophily as a
hyperparameter selector.

- For full config selection, the feature-free `default` selector has the lowest
  mean regret.
- For depth selection, homophily has the lowest mean regret, with SLP close but
  not better.
- For FAGCN `eps`, regret is tiny for all strong selectors, suggesting there is
  little room for improvement or many epsilon values perform similarly.

Overall, these results are best treated as a small negative or inconclusive
finding: the current SLP band features do not clearly add predictive value over
homophily or simple defaults for hyperparameter selection.

## Files

- `predict_hyperparam.py`: experiment script.
- `summary.csv`: aggregate results.
- `details.csv`: fold-by-fold results.
