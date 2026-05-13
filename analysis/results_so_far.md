# Results So Far

This note collects the concrete results obtained so far in the spectral-GNN/SLP experiments. It separates early exploratory runs from the current 17-dataset runs.

## 1. Early paper-harness comparison

Run context: `train_spectral.py --harness paper --k 4 8 10 --hidden 64 --runs 3 --device cuda`. This was the first 13-dataset paper-faithful comparison we discussed.

| Dataset | Winner | K | Mean test acc | Std |
| --- | --- | ---: | ---: | ---: |
| Actor | BernNet | 4 | 0.3605 | 0.0083 |
| CS | BernNet | 4 | 0.9561 | 0.0006 |
| Chameleon | GPRGNN | 10 | 0.6272 | 0.0496 |
| CiteSeer | GPRGNN | 4 | 0.7653 | 0.0090 |
| Computers | BernNet | 8 | 0.9128 | 0.0038 |
| Cora | GPRGNN | 4 | 0.8864 | 0.0087 |
| Cornell | BernNet | 8 | 0.8158 | 0.0263 |
| Minesweeper | ChebGNN | 8 | 0.8502 | 0.0085 |
| Photo | BernNet | 4 | 0.9529 | 0.0036 |
| Squirrel | GPRGNN | 10 | 0.4729 | 0.0070 |
| Texas | BernNet | 8 | 0.7719 | 0.0548 |
| Tolokers | ChebGNN | 4 | 0.7960 | 0.0033 |
| Wisconsin | BernNet | 4 | 0.8497 | 0.0300 |

Win counts: BernNet 7, GPRGNN 4, ChebGNN 2. JacobiConv won none in this early comparison.

## 2. Early Jacobi `(a,b)` sweep

The first 13-dataset Jacobi sweep with 10 seeds showed most winners at low K.

| Dataset | K | Best a | Best b | Mean test acc |
| --- | ---: | ---: | ---: | ---: |
| Actor | 4 | -0.6 | -0.7 | 0.3436 |
| CS | 4 | -0.5 | -0.8 | 0.9576 |
| Chameleon | 4 | -0.9 | -0.9 | 0.3686 |
| CiteSeer | 4 | 1.3 | -0.4 | 0.6887 |
| Computers | 4 | 2.0 | 0.0 | 0.9148 |
| Cora | 4 | 1.9 | -0.2 | 0.8038 |
| Cornell | 4 | -0.7 | -0.9 | 0.7054 |
| Minesweeper | 10 | 1.9 | 0.1 | 0.8542 |
| Photo | 4 | 1.0 | -0.7 | 0.9561 |
| Squirrel | 4 | 1.4 | -0.9 | 0.3637 |
| Texas | 4 | -0.9 | -0.4 | 0.8000 |
| Tolokers | 10 | 0.2 | -0.9 | 0.7859 |
| Wisconsin | 4 | -0.9 | -0.7 | 0.7824 |

K win counts: K=4 won 11 datasets; K=10 won 2.

## 3. Paper-massive targets on 17 datasets

Current target source: `feature_aware_slp_results/20260503-174925/target_table.csv`. Targets are selected by best mean test accuracy among complete 10-seed paper-massive configs.

Model win counts: BernNet 10, ChebGNN 5, GPRGNN 2. K targets: K=4 for 9 datasets, K=10 for 8 datasets.

| Dataset | Best model | K | Hidden | LR | WD | Mean test acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Actor | BernNet | 10 | 64 | 0.0100 | 0.0000 | 0.3677 |
| AmazonRatings | ChebGNN | 4 | 128 | 0.0050 | 0.0000 | 0.5035 |
| Chameleon | BernNet | 10 | 64 | 0.0010 | 0.0001 | 0.4242 |
| CiteSeer | GPRGNN | 4 | 64 | 0.0010 | 0.0005 | 0.7729 |
| Computers | BernNet | 10 | 128 | 0.0050 | 0.0000 | 0.9225 |
| Cora | GPRGNN | 4 | 64 | 0.0010 | 0.0005 | 0.8880 |
| Cornell | BernNet | 4 | 128 | 0.0100 | 0.0005 | 0.8184 |
| Minesweeper | ChebGNN | 4 | 128 | 0.0100 | 0.0000 | 0.8698 |
| Photo | BernNet | 10 | 128 | 0.0050 | 0.0001 | 0.9546 |
| PubMed | BernNet | 4 | 128 | 0.0100 | 0.0000 | 0.8970 |
| Questions | ChebGNN | 4 | 128 | 0.0100 | 0.0001 | 0.9718 |
| RomanEmpire | ChebGNN | 4 | 128 | 0.0100 | 0.0005 | 0.8211 |
| Squirrel | BernNet | 10 | 128 | 0.0050 | 0.0005 | 0.3695 |
| Texas | BernNet | 4 | 128 | 0.0100 | 0.0001 | 0.8342 |
| Tolokers | ChebGNN | 10 | 128 | 0.0100 | 0.0000 | 0.8103 |
| WikiCS | BernNet | 10 | 128 | 0.0100 | 0.0001 | 0.8537 |
| Wisconsin | BernNet | 10 | 128 | 0.0100 | 0.0005 | 0.8843 |

## 4. Current Jacobi targets on 17 datasets

Current target source: `feature_aware_slp_results/20260503-174925/target_table.csv`. Targets are selected by best mean validation accuracy among Jacobi `(a,b,K)` configs. K=4 wins 12 datasets; K=10 wins 5.

| Dataset | Best a | Best b | K | Mean val acc | Mean test acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor | 0.31 | -0.89 | 4 | 0.3646 | 0.3450 |
| AmazonRatings | 0.31 | 3.81 | 10 | 0.4391 | 0.4337 |
| Chameleon | 2.01 | -0.79 | 4 | 0.4094 | 0.3402 |
| CiteSeer | 0.11 | -0.69 | 4 | 0.7078 | 0.6877 |
| Computers | -0.99 | 2.81 | 4 | 0.9253 | 0.9174 |
| Cora | 1.21 | -0.49 | 10 | 0.8070 | 0.8011 |
| Cornell | -0.99 | 0.01 | 4 | 0.7627 | 0.6757 |
| Minesweeper | 3.91 | 2.61 | 4 | 0.8337 | 0.8325 |
| Photo | -0.99 | -0.69 | 4 | 0.9618 | 0.9544 |
| PubMed | 0.71 | -0.89 | 10 | 0.8270 | 0.7976 |
| Questions | 1.71 | 3.41 | 10 | 0.9714 | 0.9713 |
| RomanEmpire | 1.81 | 3.61 | 4 | 0.7180 | 0.7142 |
| Squirrel | 1.31 | -0.89 | 4 | 0.3834 | 0.3861 |
| Texas | -0.99 | -0.59 | 4 | 0.8356 | 0.7838 |
| Tolokers | 1.81 | 3.81 | 10 | 0.7923 | 0.7878 |
| WikiCS | 3.51 | -0.19 | 4 | 0.8096 | 0.7907 |
| Wisconsin | -0.89 | -0.99 | 4 | 0.8163 | 0.7941 |

## 5. Older ad-hoc SLP prediction fits

These were interactive, read-only fits from the earlier paper-massive and Jacobi analyses. They were not written to the current `feature_aware_slp_results` summaries, so treat them as exploratory transcript-backed results rather than final artifacts.

### Paper-massive K prediction

This was the surprisingly strong result. The target was the winning paper-massive K, restricted to `K=4` vs `K=10`, with exhaustive held-out-2 evaluation over 16 datasets. The target was balanced: 8 datasets preferred `K=4`, and 8 preferred `K=10`.

| Features | Bins | Held-out-2 acc | Baseline acc |
| --- | ---: | ---: | ---: |
| homophily-only | any | 0.267 | 0.267 |
| SLP-only | 1 | 0.567 | 0.267 |
| SLP-only | 2 | 0.404 | 0.267 |
| SLP-only | 3 | 0.546 | 0.267 |
| SLP-only | 4 | 0.658 | 0.267 |
| SLP-only | 5 | 0.863 | 0.267 |
| SLP-only | 6 | 0.867 | 0.267 |
| SLP-only | 7 | 0.900 | 0.267 |
| SLP-only | 8 | 0.912 | 0.267 |
| SLP-only | 9 | 0.921 | 0.267 |
| SLP-only | 10 | 0.883 | 0.267 |
| SLP+homophily | 9 | 0.917 | 0.267 |

Interpretation at the time: promising but probably optimistic. The strong shape was not a single-bin spike, since SLP-only rose sharply from 5 to 9 bins, while homophily-only was flat at baseline. The overfitting caveat is real: `n=16` is tiny, the bin choice was post hoc, and the SLP used full labels rather than train-label-only SLP.

### Paper-massive model prediction

Predicting the winning architecture was much weaker.

| Features | Best bins | Held-out-2 acc | Majority baseline |
| --- | ---: | ---: | ---: |
| SLP-only | 6 | 0.662 | 0.625 |
| SLP+homophily | 6 | 0.662 | 0.625 |

The model target counts were BernNet 10, ChebGNN 4, GPRGNN 2. This is only a small improvement over the BernNet majority baseline.

### Jacobi `(a,b)` prediction

Raw Jacobi parameter regression was disappointing. These targets were val-selected best `(a,b)` values, tested with leave-one-dataset-out regression.

| Features | K | Bins | Train R2 a/b | LOO R2 a/b | LOO MAE a/b | Baseline MAE a/b |
| --- | ---: | ---: | --- | --- | --- | --- |
| homophily | 4 | 1 | 0.000 / 0.000 | -0.129 / -0.129 | 1.473 / 1.747 | 1.473 / 1.747 |
| homophily | 10 | 1 | 0.177 / 0.006 | -0.231 / -0.199 | 1.197 / 1.377 | 1.200 / 1.356 |
| SLP | 4 | 2 | 0.470 / 0.248 | 0.109 / -0.032 | 1.237 / 1.654 | 1.473 / 1.747 |
| SLP | 10 | 3 | 0.000 / 0.000 | -0.160 / -0.160 | 1.200 / 1.356 | 1.200 / 1.356 |
| SLP+homophily | 4 | 2 | 0.584 / 0.280 | 0.334 / -0.085 | 1.080 / 1.569 | 1.473 / 1.747 |
| SLP+homophily | 10 | 2 | 0.487 / 0.310 | 0.153 / 0.008 | 0.905 / 1.215 | 1.200 / 1.356 |

The positive read is that SLP helped somewhat for `a`, especially with homophily. The negative read is stronger: `b` was hard, exact `(a,b)` argmax targets were noisy, and the models were not reliable enough to choose Jacobi configs.

Coarse Jacobi region classification was also mixed:

| Target | Features | K | Best bins | Held-out-2 acc |
| --- | --- | ---: | ---: | ---: |
| a region | SLP+homophily | 4 | 4 | 0.592 |
| a region | SLP | 4 | 4 | 0.574 |
| a region | SLP | 10 | 4 | 0.566 |
| a region | SLP+homophily | 10 | 6 | 0.549 |

`b` region classification was not convincing; simple baselines often beat it.

Jacobi preferred-K prediction was negative:

| Features | Acc | Majority baseline |
| --- | ---: | ---: |
| SLP+homophily | 0.687 | 0.714 |
| SLP-only | 0.621 | 0.714 |

Candidate validation-accuracy prediction had pointwise signal but failed as a selector: SLP with 10 bins got pointwise R2 = 0.342 and RMSE = 0.167, but predicted-selection regret was 0.038 versus 0.023 for the global-config baseline.

### Older 13-dataset Jacobi SLP regression

Before the 17-dataset massive sweep, the 13-dataset SLP-to-`(a,b)` regression also looked weak. `K=4` had train R2 values roughly in the 0.1 to 0.25 range with `a` MAE around 1.18 to 1.37 on the old sweep range. `K=10` had one apparently stronger row at 4 bins with train `R2_a = 0.70`, but held-out `a` MAE was still about 0.65. That was not enough to claim reliable Jacobi parameter prediction.

For earlier hyperparameter-prediction work in `analysis/predict_hyperparam/summary.csv`, depth prediction had small regret. The best aggregate depth selector was homophily: mean regret 0.0055, exact match 0.596. Homophily+SLP had exact match 0.615 but slightly higher regret 0.0068. This was useful background, but it is separate from the current spectral architecture/K/Jacobi target runs.

## 6. Feature-aware SLP / feature spectral density experiments

Run source: `feature_aware_slp_results/20260503-174925`. Protocol: nested 2-validation / 2-test CV over all 17 datasets, 14,280 folds per task. Candidate families in that run: `feature_density` and `label_slp_feature_density`, bins 1 through 10, with candidate-score logging disabled.

| Task | Main metric | Baseline | Selected feature mode | Selected bin mode | Result |
| --- | ---: | ---: | --- | ---: | --- |
| Architecture | 0.4641 accuracy | 0.5882 accuracy | feature_density | 2 | Worse than majority baseline |
| Jacobi `(a,b)` | MAE a=1.7086, b=2.2219 | MAE a=1.3483, b=1.9353 | label_slp_feature_density | 10 | Worse than mean baseline |
| Paper K | 0.5631 accuracy | 0.3353 accuracy | feature_density | 1 | Strongest positive result |

For K prediction, feature-only spectral density is the clear useful signal. Validation accuracy by dataset was especially high for Computers 0.9875, Photo 0.9821, CiteSeer 0.8857, Cora 0.8732, Tolokers 0.8560, Squirrel 0.8256, and Chameleon 0.7845. It was poor for Minesweeper 0.0167, Questions 0.0292, and PubMed 0.1190.

For architecture prediction, the overall result is negative. The majority baseline is BernNet, since the target distribution is BernNet 10, ChebGNN 5, GPRGNN 2. However, selected folds using `label_slp_feature_density` averaged 0.599 validation accuracy, slightly above the 0.588 majority baseline; the problem is that the selection procedure chose `feature_density` more often, and those folds performed worse.

For Jacobi `(a,b)` prediction, the aggregate result is negative. Some validation datasets improved over baseline, notably Wisconsin, Texas, Cornell, and Minesweeper, but large errors on WikiCS, CiteSeer, AmazonRatings, RomanEmpire, and Cora dominate.

### Train-label Lanczos SLP run

Run source: `feature_aware_slp_results/20260503-215333`. Protocol: nested 2-validation / 2-test CV over all 17 datasets, 14,280 folds per task. Candidate families: `label_slp`, `feature_density`, and `homophily`; bins 1 through 10. Label-dependent SLP was computed from train labels only, using the paper-random splits for seeds 0 through 9 and averaging profiles over those seeds. Spectral profiles used Lanczos with 64 steps.

| Task | Accuracy | Baseline accuracy | Selected feature mode | Selected bin mode |
| --- | ---: | ---: | --- | ---: |
| Architecture | 0.479937 | 0.588235 | label_slp | 1 |
| Paper K | 0.509944 | 0.335294 | label_slp | 3 |

This is the cleanest train-label SLP result so far for K prediction. It is positive but much weaker than the old full-label exploratory result: 0.509944 versus a 0.335294 nested baseline. Architecture prediction remains below the BernNet majority baseline.

For the selected 3-bin train-label SLP profile, class means are:

| Target K | Low mass [0, 0.67] | Mid mass [0.67, 1.33] | High mass [1.33, 2] | Homophily |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.329661 | 0.425124 | 0.245222 | 0.441 |
| 10 | 0.288817 | 0.590705 | 0.120480 | 0.442 |

Homophily is essentially identical across the two target classes. The label-SLP signal is mainly that `K=10` targets tend to have more mid-frequency mass and less high-frequency mass.

Per-dataset validation prediction stability for paper K:

| Dataset | Target K | CV accuracy | Baseline accuracy | Predicted K=10 rate | SLP low | SLP mid | SLP high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WikiCS | 10 | 0.920238 | 0.150000 | 0.920238 | 0.369044 | 0.586278 | 0.044677 |
| Squirrel | 10 | 0.789881 | 0.150000 | 0.789881 | 0.140643 | 0.766835 | 0.092523 |
| Photo | 10 | 0.761905 | 0.150000 | 0.761905 | 0.491464 | 0.471752 | 0.036786 |
| RomanEmpire | 4 | 0.756548 | 0.500000 | 0.243452 | 0.293282 | 0.300730 | 0.405989 |
| Texas | 4 | 0.745238 | 0.500000 | 0.254762 | 0.283075 | 0.292600 | 0.424325 |
| Computers | 10 | 0.641071 | 0.150000 | 0.641071 | 0.471982 | 0.500386 | 0.027649 |
| Chameleon | 10 | 0.637500 | 0.150000 | 0.637500 | 0.209366 | 0.653670 | 0.136964 |
| Tolokers | 10 | 0.632143 | 0.150000 | 0.632143 | 0.121138 | 0.853894 | 0.024965 |
| PubMed | 4 | 0.594643 | 0.500000 | 0.405357 | 0.412209 | 0.441038 | 0.146748 |
| CiteSeer | 4 | 0.486905 | 0.500000 | 0.513095 | 0.568201 | 0.205386 | 0.226417 |
| Actor | 10 | 0.470238 | 0.150000 | 0.470238 | 0.211794 | 0.576619 | 0.211581 |
| Cornell | 4 | 0.366667 | 0.500000 | 0.633333 | 0.242674 | 0.391074 | 0.366252 |
| Cora | 4 | 0.328571 | 0.500000 | 0.671429 | 0.595442 | 0.225434 | 0.179126 |
| Questions | 4 | 0.164286 | 0.500000 | 0.835714 | 0.161209 | 0.692436 | 0.146426 |
| Minesweeper | 4 | 0.128571 | 0.500000 | 0.871429 | 0.182150 | 0.672142 | 0.145708 |
| Wisconsin | 10 | 0.123214 | 0.150000 | 0.123214 | 0.295104 | 0.316202 | 0.388694 |
| AmazonRatings | 4 | 0.121429 | 0.500000 | 0.878571 | 0.228703 | 0.605279 | 0.166009 |

The easiest `K=10` datasets have low high-frequency mass and substantial mid-frequency mass. The cleanest `K=4` successes, RomanEmpire and Texas, have large high-frequency mass. The failures are informative: Questions, Minesweeper, and AmazonRatings are `K=4` targets with `K=10`-like mid-frequency profiles, while Wisconsin is a `K=10` target with a `K=4`-like high-frequency tail.

## 7. Implementation and metric-computation notes

The feature-aware script recomputes metrics from `graph_data`; it does not use `analysis/metrics.json` or spatial outputs. The script supports exact normalized-Laplacian eigendecomposition for small graphs and Chebyshev spectral projectors for large graphs. It also includes feature-only spectral density, defined as centered feature spectral energy mass divided by spectral bin width, and `label_slp_feature_density`, which concatenates label SLP mass with feature spectral density.

In the current feature-density run, all 17 datasets were included for all tasks. The command requested MPS, but metric computation ran on CPU because the script fell back from MPS where sparse or linear algebra support was unavailable.

## 8. Current interpretation

The strongest exploratory result was the older full-label SLP K predictor: SLP-only with 9 bins reached 0.921 held-out-2 accuracy on a balanced 16-dataset `K=4` vs `K=10` target. That is promising, but it is not leakage-controlled because the SLP used full labels and the bin choice was post hoc.

The strongest stricter result so far is feature-only spectral density for paper-massive K prediction under the 17-dataset nested protocol. It beats the class-balance/majority baseline by a meaningful margin: 0.563 vs 0.335. The cleanest train-label SLP result is weaker but still positive: 0.510 vs 0.335, with selected 3-bin label SLP features. Both are much weaker than the 0.921 exploratory full-label number.

Architecture prediction is not solved by these metrics as currently evaluated. Jacobi `(a,b)` prediction is also not solved; the target may be too noisy, too dataset-specific, or too sensitive to optimization/sweep resolution for 17 dataset-level points.

BernNet remains surprisingly strong in the paper-massive architecture targets, winning 10 of 17 datasets. Jacobi does not currently look competitive as a default architecture winner, even though its parameter landscape varies substantially across datasets.
