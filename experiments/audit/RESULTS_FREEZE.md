# Results freeze (brief v3)

Frozen 23 October 2026. This is the single source for the numbers in the thesis and the
paper. Every value below was written by a script in this round and read back from the file
named beside it; every summary file was produced twice from fresh processes and compared byte
for byte (identical in every case; see section 9). "Mean (SD)" is across training seeds;
intervals are 95% bootstrap over test sequences; differences are seed-averaged and paired over
the same test sequences. Sequence level means window-max aggregation over the 11 windows of a
sequence.

Abbreviations for sources: `S` = `experiments/seeds_nodup/`, `R` = `experiments/rebuild/`,
`D` = `experiments/duplicates/`, `O` = `experiments/seeds/` (original benchmark).

---

## 1. Main benchmark

**`ap_burst2k_nodup`**, chosen after R1 (section 6) because it is the only version that tests
the thesis's claim about the rate of arrival without a duplicate-record shortcut.

Generation command (from the repository root):

```
python -m src.data_generation.agentpoison_sequences --output data/raw/ap_burst2k_nodup \
    --num-sequences 2000 --matched-control 1.0 --benign-spread 120 \
    --attack-spread-mult 1.0 --seed 42 --no-duplicates
```

| Property | Value |
|---|---|
| Sequences | 2,000 (1,200 clean, 800 poisoned); 200 interactions and 20 snapshots each |
| Windows | 22,000 (11 per sequence, 10 snapshots each) |
| Split | Seeded shuffle, not stratified: train 1,200 / val 300 / test 500 sequences; 40.0% poisoned in every partition |
| Windows per partition (positive) | train 13,200 (4,098); val 3,300 (1,034); test 5,500 (1,769) |
| Record prevalence (window-final snapshots) | train 2.49%, val 2.79%, test 2.54% |
| Cluster size | 5 to 20 records, attack and matched control alike |
| Arrival | Attack: one record per consecutive interaction (k interactions). Matched clean cluster: random positions within up to 120 interactions (median first-to-last span 75) |
| Duplicates | None from sampling: each cluster is drawn without replacement from the max(8, k) nearest neighbours of one entry. 170 of 2,000 clusters still contain one identical pair because the StrategyQA knowledge base holds some identical entries |
| Relation to ap_burst2k | Same ids, split, sizes and injection positions (800/800 identical); background records differ (518 of 374,567 identical) because the generator's shared random stream advances differently |

Source: `D/benchmark_comparison.json`, `data/raw/ap_burst2k_nodup/{metadata,splits}.json`.

## 2. Training record actually used

Read from each run's `run_config.json` and `training_history.json` in `S`.

| Item | Value |
|---|---|
| Phases | Supervised phase only; the contrastive pre-training phase was not run |
| Epochs | 12 for every one of the 35 runs; early stopping (patience 10 on validation AUROC) never triggered |
| Checkpoint rule | Highest validation window-level AUROC (scored with per-window memory) |
| Objective | Binary cross-entropy (graph) weight 1.0 + focal loss (node) weight 1.5; focal gamma 3.0, class weight 0.85 |
| Optimiser | Adam, learning rate 0.0001, weight decay 0.00001, cosine annealing over 12 epochs, gradient clip 1.0 |
| Batch | 4 windows, shuffled; per-window memory slots (`memory_slots: per_window`) |
| Augmentation | Gaussian noise on the 384 embedding columns, probability 0.5, sigma drawn from U(0, 0.15) |
| Node features | 391 = 384 embedding + 3 behavioural + 4 structural (387 for the no-structural-features ablation) |
| Seeds | 1, 2, 3, 4, 5 for every configuration |
| Hardware and software | Quadro P2000 (4 GB), torch 2.5.1+cu121 |
| Code | Commit 600f647 plus uncommitted working-tree changes (`src_dirty: true` in every run_config); committing this round's code fixes the provenance |

Note: `run_config.json` records `node_meta_dim: null` because it is written before the trainer
sets it; the model actually used 7 (3 for the no-structural-features ablation), as the
parameter counts confirm. The full detector's best epochs were 12, 12, 10, 9 and 11, so it may
still have been improving at the end of the fixed 12-epoch budget.

## 3. Chapter 4 tables (main benchmark, five seeds unless marked)

### Table 4.1 Detection performance (test partition)
Source `S/seed_tables.json` (`table_4_1.test`) and `S/summary.json` (interval). Seeds 1 to 5.

| Metric | Window level | Sequence level |
|---|---|---|
| Graph AUROC | 0.7044 (0.0081) | 0.6614 (0.0144) [0.6166, 0.7054] |
| Graph AP | 0.5649 (0.0168) | 0.6115 (0.0150) |
| F1 at 0.5 | 0.4833 (0.0202) | n.a. |
| FPR at 90% TPR | 0.7299 (0.0213) | n.a. |
| Node AUROC (all records) | 0.7572 (0.0095) | n.a. |
| Node AP | 0.1819 (0.0129) | n.a. |

The best-F1 threshold is not reported: it is not stable across scoring configurations.

### Table 4.2 Detection latency (test, 200 poisoned sequences)
Source `S/seed_tables.json` (`latency`; detector seeds 1 to 5) and `S/latency_nodup.json`
(nine-feature regression, which does not depend on the detector seed). Thresholds calibrated
on benign validation windows.

| Quantity, 5% target | Detector (mean over seeds) | Nine-feature regression |
|---|---|---|
| Realised false-alarm rate on test | 4.79% | 4.48% |
| Attacks detected | 81.0 of 200 (40.5%); seeds 82, 79, 81, 79, 84 | 64 (32.0%) |
| Median latency, detected only | 4 (seed medians 5, 4, 4, 4, 4) | 7 |
| Mean latency, detected only | 4.75 | 6.73 |
| Pre-onset false alarms per sequence | 0.064 | 0.06 |

| Caught within | Detector | Regression | Difference [95% CI] |
|---|---|---|---|
| 0 snapshots | 2.3% | 1.5% | +0.008 [-0.014, 0.029] |
| 2 snapshots | 6.7% | 6.0% | +0.007 [-0.031, 0.046] |
| 5 snapshots | 28.4% | 14.0% | +0.144 [0.079, 0.211] |
| 10 snapshots | 39.1% | 25.5% | +0.136 [0.059, 0.217] |

At a 10% target: detector realises 9.40% and detects 105.0 of 200 (seeds 101, 100, 111, 102,
111), median 3; the regression realises 9.60% and detects 119, median 7. Differences: +0.032
[0.008, 0.059], +0.084 [0.035, 0.135], +0.176 [0.093, 0.255], +0.050 [-0.033, 0.132] at 0, 2,
5 and 10 snapshots.

### Table 4.3 Attack size, size-matched (sequence level)
Source `S/seed_tables.json` (`size_matched`). Each band's poisoned sequences against clean
sequences whose matched benign cluster falls in the same band.

| Injected records | Poisoned | Clean | AUROC mean (SD) |
|---|---|---|---|
| 5 to 8 | 45 | 76 | 0.618 (0.031) |
| 9 to 12 | 59 | 76 | 0.608 (0.012) |
| 13 to 16 | 43 | 68 | 0.649 (0.023) |
| 17 to 20 | 53 | 80 | 0.761 (0.009) |

Pooled against all 300 clean sequences the same bands score 0.623, 0.620, 0.655 and 0.745
(`size_curve`).

### Table 4.4 Benchmark methods (sequence level unless stated)
Sources: `R/table_4_4_unsupervised_nodup.json` (deterministic, seeded), `S/summary.json`,
`R/table_4_4_static_nodes.json`, `R/table_4_4_record_level.json`.

| Method | Sequence AUROC [95% CI] | AP | Record-level AUROC (AP) |
|---|---|---|---|
| Isolation Forest | 0.5748 [0.5219, 0.6267] | 0.4870 | 0.5108 (0.0273) |
| One-Class SVM | 0.5210 [0.4674, 0.5715] | 0.4747 | 0.5000 (0.0257) |
| Static graph attention network (5 seeds) | 0.5408 (0.0051) [0.4916, 0.5909] | 0.4759 | 0.7445 (0.1376) |
| Proposed detector (5 seeds) | 0.6614 (0.0144) [0.6166, 0.7054] | 0.6115 | 0.7572 (0.1819) |

Record-level AUROC split by what the injected records are compared against
(`R/table_4_4_record_level.json`, five seeds each):

| Comparison | Detector | Static GAT |
|---|---|---|
| Against all other records | 0.7572 (0.0095) | 0.7445 (0.0041) |
| Against ordinary background records | 0.7596 (0.0097) | 0.7496 (0.0043) |
| Against the records of matched benign clusters | **0.6611 (0.0060)** | **0.5461 (0.0030)** |

The all-records figure mostly measures membership of a dense cluster, which one snapshot
shows; against the matched clusters the static model is at chance and the gap equals the
sequence-level gap.

### Table 4.5 Ablations (sequence level)
Source `S/summary.json` (difference = detector minus configuration, seed-matched).

| Configuration | AUROC mean (SD) | AP | Detector minus this [95% CI] | Parameters |
|---|---|---|---|---|
| Full detector | 0.6614 (0.0144) | 0.6115 | reference | 670,915 |
| Temporal aggregator removed (node memory kept) | 0.6515 (0.0147) | 0.5856 | +0.0098 [-0.0130, 0.0326] | 571,843 |
| Node memory removed (temporal aggregator kept) | 0.5841 (0.0125) | 0.5323 | +0.0773 [0.0368, 0.1178] | 405,315 |
| Both removed (static GAT, attention readout) | 0.5408 (0.0051) | 0.4759 | +0.1206 [0.0762, 0.1626] | 306,243 |
| Both removed, widened to the detector's size | 0.5410 (0.0081) | 0.4787 | +0.1203 [0.0740, 0.1627] | 670,233 |
| Attention pooling replaced by mean pooling | 0.6187 (0.0077) | 0.5534 | +0.0427 [-0.0043, 0.0897] | 662,594 |
| Structural node features removed | 0.5396 (0.0076) | 0.4227 | +0.1218 [0.0619, 0.1792] | 667,331 |
| Gated readout | **not available on the main benchmark**: single seed, original benchmark, shared memory slots (0.8016 there) | | | 671,686 |

Noise floor (detector seed SD): 0.0144.

### Table 4.6 Attribution probes
Source `R/table_4_6_probes.json` (probes: train to test, logistic regression; detector row:
seeds 1 to 5).

| Probe | Information | AUROC | AP |
|---|---|---|---|
| Structural (nine features) | Final-snapshot graph and edge-weight statistics | 0.6341 | 0.4218 |
| Content (mean-pooled) | Pooled node embeddings | 0.4967 | 0.3058 |
| Combined | Both | 0.6309 | 0.4205 |
| Content (per record) | One record's embedding | 0.5143 | 0.0267 (base rate 0.0256) |
| Full detector, window level | Graph, attributes, temporal order | 0.7044 (0.0081) | 0.5649 |

### Tables 4.7 to 4.10 Robustness (full test partition, 5,500 windows)
Source `R/tables_4_7_to_4_10_sweeps.json`. Five seeds x three draws per stochastic
condition (15 values); the threshold sweep and the zero conditions are deterministic (5
values). Per-window memory; perturbations as `GraphNoiseTransform` defines them (embedding
noise changes only the 384 embedding columns, redrawn per snapshot; edges are not rebuilt).

| Condition | Window AUROC (SD) | Window AP | F1 at 0.5 | FPR at 90% TPR | Node AUROC | Sequence AUROC |
|---|---|---|---|---|---|---|
| Noise 0.00 | 0.7044 (0.0081) | 0.5649 | 0.4833 | 0.7299 | 0.7572 | 0.6614 |
| Noise 0.05 | 0.6989 (0.0066) | 0.5588 | 0.4689 | 0.7344 | 0.7547 | 0.6692 |
| Noise 0.10 | 0.6831 (0.0063) | 0.5398 | 0.4414 | 0.7565 | 0.7473 | 0.6701 |
| Noise 0.15 | 0.6607 (0.0071) | 0.5110 | 0.4219 | 0.7850 | 0.7345 | 0.6629 |
| Noise 0.20 | 0.6376 (0.0089) | 0.4800 | 0.4142 | 0.8098 | 0.7188 | 0.6518 |
| Noise 0.25 | 0.6168 (0.0104) | 0.4507 | 0.4138 | 0.8265 | 0.7030 | 0.6354 |
| Noise 0.30 | 0.5989 (0.0117) | 0.4261 | 0.4153 | 0.8412 | 0.6886 | 0.6189 |
| Threshold 0.4 | 0.6737 (0.0109) | 0.4868 | 0.5282 | 0.7609 | 0.7691 | 0.6503 |
| Threshold 0.5 | 0.7017 (0.0094) | 0.5575 | 0.5171 | 0.7348 | 0.7602 | 0.6630 |
| Threshold 0.6 | 0.7044 (0.0081) | 0.5649 | 0.4833 | 0.7299 | 0.7572 | 0.6614 |
| Threshold 0.7 | 0.6687 (0.0092) | 0.5103 | 0.3577 | 0.7619 | 0.7294 | 0.6278 |
| Threshold 0.8 | 0.6222 (0.0068) | 0.4578 | 0.2202 | 0.8276 | 0.7073 | 0.5954 |
| Metadata 0.05 | 0.7006 (0.0073) | 0.5602 | 0.4688 | 0.7326 | 0.7548 | 0.6592 |
| Metadata 0.10 | 0.6971 (0.0066) | 0.5550 | 0.4532 | 0.7362 | 0.7526 | 0.6573 |
| Metadata 0.20 | 0.6894 (0.0059) | 0.5445 | 0.4259 | 0.7426 | 0.7472 | 0.6540 |
| Metadata 0.30 | 0.6789 (0.0049) | 0.5296 | 0.3987 | 0.7541 | 0.7412 | 0.6454 |
| Edge loss 0.05 | 0.7016 (0.0077) | 0.5632 | 0.4564 | 0.7347 | 0.7569 | 0.6602 |
| Edge loss 0.10 | 0.6978 (0.0079) | 0.5601 | 0.4246 | 0.7365 | 0.7565 | 0.6570 |
| Edge loss 0.20 | 0.6898 (0.0068) | 0.5517 | 0.3603 | 0.7439 | 0.7558 | 0.6502 |
| Edge loss 0.30 | 0.6789 (0.0050) | 0.5363 | 0.2712 | 0.7552 | 0.7547 | 0.6492 |

### Table 4.11 Inference cost
Source `R/table_4_11_profile_nodup.json` (checkpoint `S/tgn_s1/phase2_best.pt`, 670,915
parameters, 2.571 MiB; 200 seeded test windows, forward pass only, per-window memory; timings
measured once, since they cannot be reproduced to the digit).

| Measurement | Batch 1 | Batch 4 |
|---|---|---|
| GPU mean latency (ms) | 89.11 | 131.63 |
| GPU 95th percentile (ms) | 104.62 | 147.45 |
| GPU throughput (windows per second) | 11.22 | 30.39 |
| CPU mean latency (ms) | 56.15 | 103.42 |
| CPU throughput (windows per second) | 17.81 | 38.68 |
| Peak GPU memory (MiB) | 22.75 | 32.29 |

### Table 3.7 Benchmark validation controls (main benchmark)
Sources: `R/probes_raw.json`, `R/table_3_7_gate_nodup.json` (test split, 500 sequences),
`data/raw/ap_burst2k_nodup/sequence_eval_baselines.json`.

| Control | Result | Verdict |
|---|---|---|
| Per-record content probe | 0.5143 (AP 0.0267, base 0.0256) | Chance |
| Pooled content probe (window) | 0.4967 | Chance |
| Position alone (window, test split) | 0.5950 | Reported explicitly |
| Final-state features (sequence) | 0.5514 [0.5043, 0.6006] | Near chance; the interval just excludes 0.5 |
| Node count (sequence) | 0.5000 | Inert by construction |
| Temporal features (sequence) | 0.6594 [0.6084, 0.7086] | Signal present |

## 4. Oracle comparison (main benchmark)
Sources: `S/oracle_comparison_5seeds.json` (detector seeds 1 to 5; gradient boosting tuned on
validation from a 12-point grid) and `D/summary.json` (duplicate features).

| Oracle | AUROC | AP | Detector minus, AUROC [CI] | Detector minus, AP [CI] |
|---|---|---|---|---|
| LR, final-state (6) | 0.5514 | 0.4369 | +0.110 [0.047, 0.171] | +0.175 [0.105, 0.237] |
| GB, final-state (6) | 0.5320 | 0.4364 | +0.129 [0.060, 0.203] | +0.175 [0.099, 0.245] |
| LR, temporal (7) | 0.6594 | 0.5732 | +0.002 [-0.060, 0.070] | +0.038 [-0.033, 0.115] |
| GB, temporal (7) | 0.6561 | 0.5589 | +0.005 [-0.055, 0.071] | +0.053 [-0.019, 0.125] |
| LR, all (13) | 0.6864 | 0.6035 | -0.025 [-0.089, 0.041] | +0.008 [-0.068, 0.083] |
| GB, all (13) | 0.6643 | 0.5617 | -0.003 [-0.066, 0.063] | +0.050 [-0.024, 0.124] |
| LR, nine features, final snapshot | 0.5353 | 0.4549 | +0.126 [0.053, 0.198] | +0.157 [0.081, 0.229] |
| LR, nine features, max over windows | 0.5403 | 0.4678 | +0.121 [0.061, 0.183] | +0.144 [0.079, 0.204] |
| LR, duplicate features | 0.5049 | 0.4154 | +0.156 [0.083, 0.228] | +0.196 [0.120, 0.265] |
| LR, 13 + duplicate features | 0.6889 | 0.6068 | -0.028 [-0.089, 0.035] | +0.005 [-0.072, 0.079] |

The detector beats every model restricted to the final state and ties every model given the
hand-built temporal summaries, on AUROC and on AP.

## 5. Validation against test, like for like
Source `S/seed_tables.json` (`table_4_1`), five seeds, as specified.

| Level | Validation | Test | Acceptance (0.85) |
|---|---|---|---|
| Window | 0.7001 (0.0051) | 0.7044 (0.0081) | Not met on either |
| Sequence | 0.6538 (0.0144) | 0.6614 (0.0144) | Not met on either |

There is no validation-to-test drop on this benchmark.

## 6. Duplicates finding (R1)
Source `D/SUMMARY.md` and `D/summary.json`; all cells five seeds (the reported checkpoint
listed separately).

In plain words: on the original benchmark the attacker draws its cluster from eight
neighbours with replacement, so clusters above eight records repeat entries, and the matched
clean clusters do too. A burst of repeated entries is then the easiest thing to detect. Three
duplicate counts alone reach 0.764 there, against the detector's 0.803, and together with the
13 features they reach 0.850, above the detector. Without duplicates those counts sit at
chance (0.505), the detector falls to 0.661, and its advantage over a static model halves from
0.249 to 0.121 but stays significant.

| Trained on / tested on | AUROC mean (SD) | AP |
|---|---|---|
| With duplicates / with duplicates | 0.8026 (0.0113) | 0.8126 |
| With duplicates / without | 0.6832 (0.0172) | 0.6546 |
| Without / without (main) | 0.6614 (0.0144) | 0.6115 |
| Without / with duplicates | 0.6852 (0.0062) | 0.6439 |
| Reported checkpoint (seed 42) / with, without | 0.8194, 0.6897 | |

| Comparison on one test set | AUROC difference [95% CI] |
|---|---|
| Test with duplicates: trained with minus trained without | +0.1174 [0.0782, 0.1542] |
| Test without: trained with minus trained without | +0.0219 [-0.0252, 0.0719] |
| Temporal gap, with duplicates (static 0.5533) | +0.2493 [0.2087, 0.2892] |
| Temporal gap, without (static 0.5408) | +0.1206 [0.0762, 0.1626] |
| Duplicate features alone, with duplicates | 0.7635 [0.7205, 0.8062]; detector minus +0.039 [-0.016, 0.091] |
| 13 + duplicate features, with duplicates | 0.8495 [0.8168, 0.8835]; detector minus -0.047 [-0.094, 0.002] |

Training on duplicates does not hurt on duplicate-free data; only models trained on duplicates
exceed about 0.69 on the original test set, and they do it by reading the duplicate bursts.

## 7. Figures
Source `R/figures_4_1_to_4_3.json` and the JSON beside each PNG.

| Figure | File | Seed | Test sequence | Source data |
|---|---|---|---|---|
| 4.1 Node score distribution | `R/figures/fig_4_1_node_score_distribution.png` | 1 (median of 0.6401, 0.6582, 0.6595, 0.6721, 0.6770) | poisoned_placement_0756 | `.json` beside it |
| 4.2 Node graph overlay | `R/figures/fig_4_2_node_graph_overlay.png` | 1 | poisoned_placement_0756 | `.json` beside it |
| 4.3 Detection score across a sequence | `R/figures/fig_4_3_single_sequence_detection_score.png` | 1 | poisoned_placement_0756 | `.json` beside it |
| Missed example (limitations) | `R/figures_missed_example/` (same three) | 1 | poisoned_placement_0463 | `.json` beside each |
| 4.4 Embedding noise | `R/figures/fig_4_4_embedding_noise.png` | 1 to 5, 3 draws | full test partition | `.json` beside it |
| 4.5 Edge threshold | `R/figures/fig_4_5_edge_threshold.png` | 1 to 5 | full test partition | `.json` beside it |
| 4.6 Metadata corruption and edge loss | `R/figures/fig_4_6_metadata_and_edge_loss.png` | 1 to 5, 3 draws | full test partition | `.json` beside it |
| 4.7 Four degradations together | `R/figures/fig_4_7_degradation_dashboard.png` | 1 to 5, 3 draws | full test partition | `.json` beside it |

Sequence rule (fixed in `rebuild.pick_sequence` before plotting): median attack size (12),
onset at snapshot 12 to 16, detected at the median latency of all detected test attacks (5),
ties to the lowest id; 2 of the 6 candidates were detected. Thresholds at a 5% benign
false-alarm rate on the whole validation partition: node 0.440, graph 0.944. On 0756 the
detector flags 11 of 12 injected records and 26 benign ones, and alerts at snapshot 17 after
onset at 12. The thesis's earlier example, 0463, is missed entirely by the median seed.

Figures 4.4 to 4.7 (added 23 October at Moses's request, `rebuild.py sweep-figures`) plot the
Tables 4.7 to 4.10 data as mean +/- SD over seeds and draws, with a sequence-AUROC line added
to the original three (window AUROC, window AP, node AUROC).

## 8. Not done, and why
* **Gated readout:** single seed on the original benchmark only (brief: stays single seed); it
  cannot be compared with main-benchmark rows.
* **Second threat model (E5), attack success (E6), training speed fix (R4):** not approved for
  this round.
* **Longer training:** the detector's best epoch was at or near the end of the 12-epoch budget;
  a longer budget was not tried because it would change the recipe across configurations.
* **Inference timings:** measured once; timings are not reproducible to the digit.
* **Code provenance:** all runs used uncommitted code on top of 600f647; the commit awaits
  Moses's confirmation.

## 9. Reproducibility checks (each run twice from a fresh process, outputs compared)

Identical in every case: `S/summary.json`, `S/seed_tables.json`,
`S/oracle_comparison_5seeds.json`, `D/summary.json`, `R/table_3_7_gate_nodup.json`,
`R/table_4_4_unsupervised_nodup.json`, `R/table_4_4_static_nodes.json`,
`R/table_4_4_record_level.json`, `R/table_4_6_probes.json`,
`R/tables_4_7_to_4_10_sweeps.json`. Logs: `S/PLAN.md` and `R/freeze_extras.log`.
