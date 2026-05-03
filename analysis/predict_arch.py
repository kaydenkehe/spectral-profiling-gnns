import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# minimal answer to RQ1: does the slp shape predict the empirically best architecture
# better than scalar class-balanced homophily?
# we restrict the prescription pool to three fixed-filter models: gcn (low-pass),
# hfgcn (high-pass), mlp (no graph). fagcn is intentionally excluded because
# best_spatial.csv shows it >= gcn on every dataset, so any classifier can
# trivially prescribe fagcn and then slp does no work. 

KEEP = {'gcn', 'hfgcn', 'mlp'}

# load empirical winners from the spatial sweep
best = pd.read_csv('/n/home06/drooryck/spectral-profling-gnns/spatial/runs/best_spatial.csv')
best = best[best['model'].isin(KEEP)]
winners = best.loc[best.groupby('dataset')['mean'].idxmax(), ['dataset', 'model']]

# load train-subgraph slp + class-balanced homophily from spatial/metrics.json
with open('/n/home06/drooryck/spectral-profling-gnns/spatial/metrics.json') as f:
    metrics = json.load(f)

NAME_MAP = {
    'Cora': 'Cora', 'CiteSeer': 'CiteSeer',
    'Amazon Photo': 'Photo', 'Amazon Computers': 'Computers',
    'Coauthor CS': 'CS', 'Texas': 'Texas', 'Cornell': 'Cornell',
    'Wisconsin': 'Wisconsin', 'Chameleon': 'Chameleon', 'Squirrel': 'Squirrel',
    'Actor': 'Actor', 'Minesweeper': 'Minesweeper', 'Tolokers': 'Tolokers',
}

# slp at three quantile cutoffs: how much label energy lives below lambda in {0.5, 1.0, 1.5}.
GRID = np.linspace(0, 2, 50)

def slp_at(cdf, lam):
    return float(np.interp(lam, GRID, cdf))

rows = []
for display_name, m in metrics.items():
    rows.append({
        'dataset': NAME_MAP.get(display_name, display_name),
        'h_class': m.get('class_homophily', m.get('homophily')),
        'pi_05': slp_at(m['cdf'], 0.5),
        'pi_10': slp_at(m['cdf'], 1.0),
        'pi_15': slp_at(m['cdf'], 1.5),
    })
feat = pd.DataFrame(rows).merge(winners, on='dataset')

# leave-one-out cross-validation: with only 13 graphs each held-out has to
# count. for each fold we pull one dataset out, fit a logistic regression on
# the other 12 (standardized), predict the held-out one, and score whether
# the prediction matches the empirical winner. the reported number is the
# fraction of correct predictions across all 13 folds. granularity is 1/13
# so small differences are not meaningful.
def loocv(X, y):
    correct = 0
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        correct += int(clf.predict(sc.transform(X[te]))[0] == y[te][0])
    return correct / len(y)

y = feat['model'].to_numpy()

acc_h = loocv(feat[['h_class']].to_numpy(), y)
acc_slp = loocv(feat[['pi_05', 'pi_10', 'pi_15']].to_numpy(), y)

counts = pd.Series(y).value_counts()
print(f'class distribution: {dict(counts)}')
print(f'majority-class baseline:     {counts.max() / counts.sum():.3f}')
print(f'class-homophily baseline:    LOOCV acc = {acc_h:.3f}')
print(f'slp 3-quantile features:     LOOCV acc = {acc_slp:.3f}')
print()
out = feat[['dataset', 'h_class', 'pi_05', 'pi_10', 'pi_15', 'model']].copy()
print(out.sort_values('h_class', ascending=False).to_string(index=False))
