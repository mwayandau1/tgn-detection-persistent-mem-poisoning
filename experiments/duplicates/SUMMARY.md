# R1 duplicates study: results

Generated 2026-10-23 10:19 by `python -m src.experiments.duplicates_study summarize` from `summary.json`.
Test partition: 500 sequences (200 poisoned), sequence level (window max). AUROC is the mean over runs (SD across runs); the 95% interval is a bootstrap over sequences of the run-averaged AUROC; differences are run-averaged and paired over the same test sequences.

## The two benchmarks

* Same sequence ids: True; same split: True; same cluster sizes: True (poisoned range [5, 20]).
* Poisoned sequences with identical injection slots: 800/800.
* Clusters containing duplicate entries: with duplicates 1571/2000, without 170/2000.
* Background (non-cluster) records identical: 518/374567.

## Full detector, four cells

| Trained on / tested on | AUROC mean (SD) | AP | 95% CI | Runs |
|---|---|---|---|---|
| train dup / test dup | 0.8026 (0.0113) | 0.8126 | [0.7597, 0.8438] | 5 |
| train dup / test nodup | 0.6832 (0.0172) | 0.6546 | [0.6388, 0.7303] | 5 |
| train nodup / test nodup | 0.6614 (0.0144) | 0.6115 | [0.6166, 0.7054] | 5 |
| train nodup / test dup | 0.6852 (0.0062) | 0.6439 | [0.6375, 0.7330] | 5 |
| reported checkpoint (seed 42, trained dup) / test dup | 0.8194 | 0.8142 | [0.7794, 0.8596] | 1 |
| reported checkpoint (seed 42, trained dup) / test nodup | 0.6897 | 0.5999 | [0.6443, 0.7345] | 1 |

| Same test set | AUROC difference [95% CI] | AP difference [95% CI] |
|---|---|---|
| test dup: trained dup minus trained nodup | +0.1174 [+0.0782, +0.1542] | +0.1687 [+0.1214, +0.2151] |
| test nodup: trained dup minus trained nodup | +0.0219 [-0.0252, +0.0719] | +0.0430 [-0.0118, +0.0995] |

## Temporal gap (detector minus static GAT, both trained and tested on the same benchmark)

| Benchmark | Static GAT AUROC (SD) | Detector minus static, AUROC [95% CI] | AP [95% CI] |
|---|---|---|---|
| dup | 0.5533 (0.0063), 5 runs | +0.2493 [+0.2087, +0.2892] | +0.3536 [+0.3002, +0.4002] |
| nodup | 0.5408 (0.0051), 5 runs | +0.1206 [+0.0762, +0.1626] | +0.1357 [+0.0805, +0.1818] |

## Logistic regressions with a duplicate feature

Duplicate features: records in the final snapshot with a partner at cosine >= 0.99, the largest growth in that count within any 10-snapshot window, and the largest increase between two consecutive snapshots.

| Benchmark | Features | AUROC | AP | 95% CI | Detector minus this, AUROC | AP |
|---|---|---|---|---|---|---|
| dup | duplicate features | 0.7635 | 0.7138 | [0.7205, 0.8062] | +0.0391 [-0.0158, +0.0911] | +0.0988 [+0.0356, +0.1603] |
| dup | 13 features | 0.8030 | 0.7203 | [0.7644, 0.8400] | -0.0004 [-0.0532, +0.0518] | +0.0923 [+0.0304, +0.1529] |
| dup | 13 + duplicate features | 0.8495 | 0.7814 | [0.8168, 0.8835] | -0.0470 [-0.0944, +0.0023] | +0.0312 [-0.0305, +0.0906] |
| nodup | duplicate features | 0.5049 | 0.4154 | [0.4546, 0.5575] | +0.1564 [+0.0830, +0.2277] | +0.1961 [+0.1198, +0.2648] |
| nodup | 13 features | 0.6864 | 0.6035 | [0.6349, 0.7333] | -0.0250 [-0.0879, +0.0364] | +0.0081 [-0.0685, +0.0804] |
| nodup | 13 + duplicate features | 0.6889 | 0.6068 | [0.6379, 0.7354] | -0.0275 [-0.0894, +0.0347] | +0.0047 [-0.0716, +0.0792] |
