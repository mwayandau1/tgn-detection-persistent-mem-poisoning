# Results freeze 2 (step 5 configuration)

Frozen 3 November 2026. This is the single source for the numbers of the improved detector, in
the thesis and the paper. It follows `RESULTS_FREEZE.md` (23 October 2026, "freeze 1") section
by section; freeze 1 stays the record of the original design and of everything that was not
redone (section 9). How this configuration was reached, every attempt on the way, the decision
rules fixed before each run and one protocol breach are in `experiments/IMPROVEMENT.md`
(steps 1 to 6).

Every value below was written by a script and read back from the file named beside it. Every
summary file was produced twice from fresh processes and compared byte for byte (section 11).
"Mean (SD)" is across training seeds 1 to 5; intervals are 95% bootstrap over the 500 test
sequences (2,000 resamples, seed 0); differences are seed-averaged and paired over the same
test sequences. Test figures were read once, after all runs of a phase had finished (headline
30 October, everything else 3 November); nothing was chosen after reading them.

Abbreviations for sources: `S2` = `experiments/seeds_coherence/`, `R2` =
`experiments/rebuild_coherence/`, `S` = `experiments/seeds_nodup/` and `R` = `experiments/rebuild/`
(freeze 1).

---

## 0. What changed from freeze 1

| Item | Freeze 1 | Freeze 2 |
|---|---|---|
| Unit scored | 11 windows of 10 snapshots per sequence; sequence score = maximum over windows | One window of all 20 snapshots; sequence score = that window's score |
| Edges | Cosine similarity at or above 0.6 (edge feature: weight) | The same, plus arrival edges between records that arrived within 10 interactions of each other (edge features: weight, arrival flag) |
| Node features | 391 = 384 embedding + 3 behavioural + 4 structural | 401 = the same + 10 arrival-coherence columns (`dataset.with_coherence_features`) |
| Epochs | 12, fixed | Up to 66, early stopping with patience 10 |
| Window target | Attack present | Attack present (unchanged) |
| Configuration chosen by | n.a. | The step 5 rule, on validation only (IMPROVEMENT.md step 5): detector + static GAT, rank mean |

The ten coherence columns, per record and per snapshot, from the embeddings and arrival order
only: for runs of 5, 10, 15 and 20 consecutive records containing the record, the highest mean
pairwise cosine, and that minus its median over the snapshot; and the mean cosine to the
records that arrived within 5 and within 10 interactions. They target the attack's form (a
consecutive burst of near-identical records). They never see a label.

## 1. Main benchmark

Unchanged: `ap_burst2k_nodup`, as in freeze 1 section 1 (2,000 sequences; train 1,200, val
300, test 500; 40.0% poisoned in every partition). Online scoring (section 7) reads each
sequence after snapshots 9 to 19, which are the end points of freeze 1's 11 windows with the
same labels (5,500 on test).

## 2. Training record actually used

Read from each run's `run_config.json`, `training_history.json` and `variant.json` in `S2`.

| Item | Value |
|---|---|
| Phases | Supervised phase only; the contrastive pre-training phase was not run |
| Epochs | Up to 66 with early stopping (patience 10 on validation AUROC). Full detector: stopped at 59, 35, 44, 35 and 60 epochs, best epochs 49, 25, 34, 25 and 50. Static GAT: 66, 66, 57, 66, 66 (best 59, 58, 47, 62, 66) |
| Checkpoint rule | Highest validation AUROC of the one window, which is the validation sequence AUROC (per-window memory) |
| Objective | Binary cross-entropy (graph) weight 1.0 + focal loss (node) weight 1.5; focal gamma 3.0, class weight 0.85 |
| Optimiser | Adam, learning rate 0.0001, weight decay 0.00001, cosine annealing over 66 epochs, gradient clip 1.0 |
| Batch | 4 windows (sequences), shuffled; per-window memory slots |
| Augmentation | Gaussian noise on the 384 embedding columns, probability 0.5, sigma drawn from U(0, 0.15); the coherence columns are computed from the clean embeddings |
| Node features | 401 (397 without structural features; 391 without coherence features) |
| Edge features | 2: similarity weight (0 on arrival edges) and arrival flag |
| Seeds | 1, 2, 3, 4, 5 for every configuration. Seed 1 of the detector and static GAT are the step 5 pilots, trained with validation scoring only and scored on test in phase A; seed 1 without coherence features is the step 4 pilot (`experiments/option3_sequence/tgn_s1`) |
| Hardware and software | Quadro P2000 (4 GB), torch 2.5.1+cu121 |
| Training time | 1.0 to 6.2 h per run. Exception: `meanpool_s3` records 14.8 h because the laptop slept for about 11.5 h during its second epoch (IMPROVEMENT.md, "Interruption"); its result is unaffected |
| Code | Commit 2dbcdbc plus uncommitted working-tree changes (`src_dirty: true` in every run_config); committing this round's code fixes the provenance |

## 3. Headline: the configuration chosen by the step 5 rule

Source `S2/ensemble_headline.json` (`coherence_plan ensemble`). Detector seed s paired with
static GAT seed s; each model's sequence scores converted to ranks over the 500 test sequences
and averaged. The deployable form replaces each score by its position among the same model's
validation scores, so it needs no other test sequence.

| Configuration | Sequence AUROC mean (SD) [95% CI] | AP |
|---|---|---|
| **Detector + static GAT, rank mean** | **0.9221 (0.0089) [0.8975, 0.9424]** | 0.9080 |
| The same, deployable form | 0.9213 (0.0088) [0.8964, 0.9419] | 0.9072 |
| Detector alone | 0.9088 (0.0032) [0.8819, 0.9319] | 0.8936 |
| Static GAT alone | 0.9156 (0.0120) [0.8916, 0.9367] | 0.8995 |
| Freeze 1 detector | 0.6614 (0.0144) [0.6166, 0.7054] | 0.6115 |

Per pair: 0.9202, 0.9317, 0.9149, 0.9126, 0.9311.

| Paired difference (test) | AUROC [95% CI] | AP [95% CI] |
|---|---|---|
| Ensemble minus detector | +0.0133 [0.0063, 0.0206] | +0.0145 [0.0064, 0.0233] |
| Ensemble minus static GAT | +0.0065 [0.0001, 0.0134] | +0.0086 [0.0007, 0.0170] |
| Detector minus static GAT | -0.0068 [-0.0195, 0.0061] | -0.0059 [-0.0208, 0.0097] |

Validation (300 sequences), for the record of the decision: ensemble 0.9260 (0.0070), detector
0.9138 (0.0032), static GAT 0.9110 (0.0109). The rule chose the ensemble on seed 1 because the
detector fell 0.0122 short of it, outside the 0.01 margin within which a single model was to be
preferred. The interval of that difference included zero on validation (seed 1:
[-0.0035, 0.0292]; five pairs: [-0.0001, 0.0254]) and excludes it on test.

The ensemble is reported at sequence level only. Every table below describes the detector (the
temporal model) unless it says otherwise.

## 4. Chapter 4 tables (main benchmark, five seeds unless marked)

### Table 4.1 Detection performance (test partition)
Source `S2/seed_tables.json` (`table_4_1.test`) and `S2/summary.json` (interval).

| Metric | Online level (snapshots 9 to 19) | Sequence level |
|---|---|---|
| Graph AUROC | 0.8248 (0.0046) | 0.9088 (0.0032) [0.8819, 0.9319] |
| Graph AP | 0.7698 (0.0050) | 0.8936 (0.0027) |
| F1 at 0.5 | 0.6311 (0.0092) | n.a. |
| FPR at 90% TPR | 0.5653 (0.0161) | n.a. |
| Node AUROC (all records, at each end point's snapshot) | 0.8846 (0.0068) | n.a. |
| Node AP | 0.5272 (0.0610) | n.a. |

Online level replaces freeze 1's window level: the same 5,500 end points and labels, read from
a model that has seen the sequence from its start (section 7 explains why it is lower than the
sequence level). Node AUROC on the final snapshot only is 0.9043 (0.0063) (`R2/tables_4_7_to_4_10_sweeps.json`, zero condition).

### Table 4.2 Detection latency (test, 200 poisoned sequences)
Source `S2/seed_tables.json` (`latency`, `latency_per_snapshot`; detector seeds 1 to 5) and the
nine-feature regression reused from freeze 1 (`S/latency_window_scores_nodup.npz`,
`S/latency_nodup.json`: same data and windows, independent of the detector). Thresholds
calibrated on benign validation online scores.

Primary (fixed in advance): one threshold for all end points.

| Quantity, 5% target | Detector (mean over seeds) | Nine-feature regression |
|---|---|---|
| Realised false-alarm rate on test | 3.67% | 4.48% |
| Attacks detected | 129.8 of 200 (64.9%); seeds 132, 126, 128, 134, 129 | 64 (32.0%) |
| Median latency, detected only | 2 (every seed 2) | 7 |
| Mean latency, detected only | 2.20 | 6.73 |
| Pre-onset false alarms per sequence | 0.100 | 0.06 |

| Caught within | Detector | Regression | Difference [95% CI] |
|---|---|---|---|
| 0 snapshots | 5.8% | 1.5% | +0.043 [0.010, 0.078] |
| 2 snapshots | 44.8% | 6.0% | +0.388 [0.320, 0.458] |
| 5 snapshots | 62.7% | 14.0% | +0.487 [0.412, 0.558] |
| 10 snapshots | 64.8% | 25.5% | +0.393 [0.313, 0.471] |

At a 10% target: the detector realises 9.75% and detects 150.4 of 200 (seeds 152, 154, 148,
148, 150), median 1 to 2 (seed medians 1.5, 1, 2, 2, 1), pre-onset alarms 0.361 per sequence;
the regression realises 9.60% and detects 119, median 7. Differences: +0.118 [0.072, 0.167],
+0.457 [0.385, 0.525], +0.483 [0.405, 0.558], +0.282 [0.202, 0.359] at 0, 2, 5 and 10
snapshots.

Secondary (added before any test figure, IMPROVEMENT.md step 6): one threshold per end
snapshot. At a 5% target the detector realises 5.50% and detects 148.4 of 200 (seeds 146, 150,
156, 145, 145), median 2, mean 2.48, pre-onset 0.084; caught within 0, 2, 5 and 10 snapshots
7.7%, 49.1%, 68.6%, 72.9% (differences from the regression +0.062 [0.028, 0.097], +0.431
[0.361, 0.500], +0.546 [0.476, 0.616], +0.474 [0.396, 0.550]). At 10%: realises 10.91%,
detects 165.2 (seeds 167, 163, 165, 168, 163).

### Table 4.3 Attack size, size-matched (sequence level)
Source `S2/seed_tables.json` (`size_matched`).

| Injected records | Poisoned | Clean | AUROC mean (SD) |
|---|---|---|---|
| 5 to 8 | 45 | 76 | 0.828 (0.004) |
| 9 to 12 | 59 | 76 | 0.874 (0.007) |
| 13 to 16 | 43 | 68 | 0.925 (0.005) |
| 17 to 20 | 53 | 80 | 0.973 (0.004) |

Pooled against all 300 clean sequences the same bands score 0.873, 0.887, 0.919 and 0.955
(`size_curve`).

### Table 4.4 Benchmark methods (sequence level unless stated)
Sources: `S2/summary.json`, `S2/ensemble_headline.json`, `S2/oracle_comparison_5seeds.json`,
`R2/table_4_4_static_nodes.json`, `R2/table_4_4_record_level.json`; the unsupervised rows are
freeze 1's (`R/table_4_4_unsupervised_nodup.json`) and do not see the coherence columns.

| Method | Sequence AUROC [95% CI] | AP | Record-level AUROC (AP) |
|---|---|---|---|
| Isolation Forest (freeze 1) | 0.5748 [0.5219, 0.6267] | 0.4870 | 0.5108 (0.0273) |
| One-Class SVM (freeze 1) | 0.5210 [0.4674, 0.5715] | 0.4747 | 0.5000 (0.0257) |
| Coherence features alone, logistic regression (40 aggregates) | 0.8351 | 0.7891 | n.a. |
| Static graph attention network (5 seeds) | 0.9156 (0.0120) [0.8916, 0.9367] | 0.8995 | 0.9162 (0.5499) |
| Proposed detector (5 seeds) | 0.9088 (0.0032) [0.8819, 0.9319] | 0.8936 | 0.8846 (0.5272) |
| Detector + static GAT, rank mean (5 pairs) | 0.9221 (0.0089) [0.8975, 0.9424] | 0.9080 | n.a. |

Record-level AUROC split by what the injected records are compared against
(`R2/table_4_4_record_level.json`, five seeds each, all end points):

| Comparison | Detector | Static GAT |
|---|---|---|
| Against all other records | 0.8846 (0.0068) | 0.9162 (0.0027) |
| Against ordinary background records | 0.8833 (0.0069) | 0.9151 (0.0029) |
| Against the records of matched benign clusters | **0.9352 (0.0044)** | **0.9570 (0.0033)** |

In freeze 1 the static model was at chance against the matched clusters (0.5461); with the
coherence columns one snapshot separates them, and the static model ranks records better than
the detector.

### Table 4.5 Ablations (sequence level)
Source `S2/summary.json` (difference = detector minus configuration, seed-matched); parameters
from each run's `variant.json`.

| Configuration | AUROC mean (SD) | AP | Detector minus this [95% CI] | Parameters |
|---|---|---|---|---|
| Full detector | 0.9088 (0.0032) | 0.8936 | reference | 680,515 |
| Temporal aggregator removed (node memory kept) | 0.8988 (0.0027) | 0.8830 | +0.0100 [0.0011, 0.0195] | 581,443 |
| Node memory removed (temporal aggregator kept) | 0.9148 (0.0014) | 0.8972 | -0.0060 [-0.0180, 0.0058] | 411,075 |
| Both removed (static GAT, attention readout) | 0.9156 (0.0120) | 0.8995 | -0.0068 [-0.0195, 0.0061] | 312,003 |
| Both removed, widened to the detector's size | **0.9332 (0.0023)** | 0.9157 | **-0.0244 [-0.0414, -0.0097]** | 680,943 |
| Attention pooling replaced by mean pooling | 0.9172 (0.0070) | 0.8984 | -0.0084 [-0.0228, 0.0055] | 672,194 |
| Structural node features removed | 0.9069 (0.0051) | 0.8932 | +0.0019 [-0.0093, 0.0128] | 676,931 |
| Coherence features removed (new) | 0.6680 (0.0583) | 0.6075 | **+0.2408 [0.2036, 0.2767]** | 671,555 |
| Gated readout | not available (freeze 1 section 8) | | | |

Noise floor (detector seed SD): 0.0032. Without coherence features the seeds score 0.6551,
0.5738, 0.6857, 0.7235 and 0.7019 (validation 0.7206, 0.6625, 0.7167, 0.7275, 0.7316), the
level of freeze 1. The widened static GAT is matched to 680,515 parameters
(`component_ablation_models.matched_static_hidden`, hidden 238; per seed 0.9334, 0.9338,
0.9291, 0.9348, 0.9347).

### Table 4.6 Attribution probes
Source `R2/table_4_6_probes.json` (probes: train to test, logistic regression; they read the
data only and equal freeze 1's; detector row: online level, seeds 1 to 5).

| Probe | Information | AUROC | AP |
|---|---|---|---|
| Structural (nine features) | Final-snapshot graph and edge-weight statistics | 0.6341 | 0.4218 |
| Content (mean-pooled) | Pooled node embeddings | 0.4967 | 0.3058 |
| Combined | Both | 0.6309 | 0.4205 |
| Content (per record) | One record's embedding | 0.5143 | 0.0267 (base rate 0.0256) |
| Coherence (40 aggregates, sequence level) | Arrival-coherence columns of the final snapshot | 0.8351 | 0.7891 |
| Full detector, online level | Graph, attributes, coherence, temporal order | 0.8248 (0.0046) | 0.7698 |

The coherence row is the oracle comparison's "LR, coherence (40)" (`S2/oracle_comparison_5seeds.json`),
at sequence level, not window level.

### Tables 4.7 to 4.10 Robustness (full test partition, 500 sequences)
Source `R2/tables_4_7_to_4_10_sweeps.json` (per seed: `R2/sweeps/tgn_s{1..5}.json`). Five
seeds x three draws per stochastic condition (15 values); the threshold sweep and the zero
conditions are deterministic (5 values). One window per sequence, so every column is sequence
level; node columns are the final snapshot. Rule as in freeze 1 (amended before any sweep ran,
IMPROVEMENT.md step 6): everything derived from the embeddings when a snapshot is built
(similarity edges and weights, structural columns, arrival edges, coherence columns) comes from
the clean embeddings; noise changes the 384 embedding columns only; the threshold and edge loss
change the similarity edges only (arrival edges are never dropped); corruption changes the
behavioural columns. Online scoring under perturbation was not run.

| Condition | Sequence AUROC (SD) | AP | F1 at 0.5 | FPR at 90% TPR | Node AUROC | Node AP |
|---|---|---|---|---|---|---|
| Noise 0.00 | 0.9088 (0.0032) | 0.8936 | 0.7652 | 0.3020 | 0.9043 | 0.5552 |
| Noise 0.05 | 0.9096 (0.0041) | 0.8944 | 0.7699 | 0.3016 | 0.9041 | 0.5491 |
| Noise 0.10 | 0.9054 (0.0051) | 0.8892 | 0.7678 | 0.3180 | 0.8991 | 0.5307 |
| Noise 0.15 | 0.8952 (0.0066) | 0.8768 | 0.7532 | 0.3393 | 0.8886 | 0.4988 |
| Noise 0.20 | 0.8779 (0.0074) | 0.8558 | 0.7285 | 0.3778 | 0.8740 | 0.4573 |
| Noise 0.25 | 0.8532 (0.0089) | 0.8254 | 0.7059 | 0.4551 | 0.8563 | 0.4081 |
| Noise 0.30 | 0.8237 (0.0102) | 0.7885 | 0.6748 | 0.5207 | 0.8348 | 0.3494 |
| Threshold 0.4 | 0.9118 (0.0044) | 0.8942 | 0.6912 | 0.2900 | 0.8781 | 0.5185 |
| Threshold 0.5 | 0.9115 (0.0030) | 0.8955 | 0.7538 | 0.3000 | 0.8974 | 0.5544 |
| Threshold 0.6 | 0.9088 (0.0032) | 0.8936 | 0.7652 | 0.3020 | 0.9043 | 0.5552 |
| Threshold 0.7 | 0.9038 (0.0030) | 0.8865 | 0.7720 | 0.3400 | 0.9110 | 0.5467 |
| Threshold 0.8 | 0.9040 (0.0023) | 0.8869 | 0.7720 | 0.2907 | 0.9202 | 0.5383 |
| Metadata 0.05 | 0.9093 (0.0031) | 0.8939 | 0.7598 | 0.3011 | 0.9000 | 0.5503 |
| Metadata 0.10 | 0.9090 (0.0031) | 0.8935 | 0.7544 | 0.2967 | 0.8956 | 0.5460 |
| Metadata 0.20 | 0.9072 (0.0029) | 0.8917 | 0.7356 | 0.3011 | 0.8862 | 0.5318 |
| Metadata 0.30 | 0.9059 (0.0032) | 0.8890 | 0.6990 | 0.3038 | 0.8787 | 0.5147 |
| Edge loss 0.05 | 0.9094 (0.0029) | 0.8942 | 0.7700 | 0.2980 | 0.9070 | 0.5567 |
| Edge loss 0.10 | 0.9103 (0.0029) | 0.8952 | 0.7742 | 0.2938 | 0.9100 | 0.5591 |
| Edge loss 0.20 | 0.9115 (0.0028) | 0.8962 | 0.7848 | 0.2931 | 0.9152 | 0.5610 |
| Edge loss 0.30 | 0.9134 (0.0026) | 0.8978 | 0.7888 | 0.2858 | 0.9197 | 0.5600 |

Extra sweep (not in freeze 1), `embedding_noise_full`: the embeddings themselves are noisy, so
the similarity edges are rebuilt from them at 0.6 (refreshing the structural columns) and the
arrival-edge weights and coherence columns are computed from them too.

| Condition | Sequence AUROC (SD) | AP | F1 at 0.5 | FPR at 90% TPR | Node AUROC | Node AP |
|---|---|---|---|---|---|---|
| Full noise 0.00 | 0.9088 (0.0032) | 0.8936 | 0.7652 | 0.3020 | 0.9043 | 0.5552 |
| Full noise 0.05 | 0.8112 (0.0146) | 0.7506 | 0.0177 | 0.4880 | 0.7566 | 0.1048 |
| Full noise 0.10 | 0.6106 (0.0150) | 0.5003 | 0.0000 | 0.8062 | 0.6148 | 0.0316 |
| Full noise 0.15 | 0.5444 (0.0150) | 0.4395 | 0.0000 | 0.8838 | 0.5784 | 0.0276 |
| Full noise 0.20 | 0.5220 (0.0152) | 0.4221 | 0.0000 | 0.9007 | 0.5646 | 0.0267 |
| Full noise 0.25 | 0.5106 (0.0138) | 0.4136 | 0.0000 | 0.8980 | 0.5574 | 0.0264 |
| Full noise 0.30 | 0.5021 (0.0116) | 0.4067 | 0.0000 | 0.8978 | 0.5524 | 0.0265 |

Under the freeze 1 rule the detector degrades smoothly (0.9088 to 0.8237 at 0.30). When the
noise reaches the coherence columns it is near chance from 0.15 and every score falls below
0.5 from 0.10 (F1 0): the features depend on precise cosines between records, and the model
never saw noisy coherence columns in training.

### Table 4.11 Inference cost
Source `R2/table_4_11_profile.json` (checkpoint `S2/tgn_s1/phase2_best.pt`, 680,515 parameters,
2.607 MiB; 200 seeded test sequences of 20 snapshots, forward pass only, per-window memory;
timings measured once). A window now holds 20 snapshots against freeze 1's 10, so per-window
figures are not comparable with freeze 1's.

| Measurement | Batch 1 | Batch 4 |
|---|---|---|
| GPU mean latency (ms) | 209.13 | 259.75 |
| GPU 95th percentile (ms) | 260.31 | 304.62 |
| GPU throughput (windows per second) | 4.78 | 15.40 |
| CPU mean latency (ms) | 199.73 | 593.01 |
| CPU throughput (windows per second) | 5.01 | 6.75 |
| Peak GPU memory (MiB) | 36.55 | 104.78 |

Input preparation (arrival edges and coherence columns, CPU, per snapshot, 1,000 snapshots):
mean 5.89 ms, 95th percentile 9.13 ms.

### Table 3.7 Benchmark validation controls
Not redone: every control reads the data only, which is unchanged; freeze 1's values stand.

## 5. Oracle comparison (main benchmark)
Source `S2/oracle_comparison_5seeds.json` (detector seeds 1 to 5; gradient boosting tuned on
validation from a 12-point grid). The coherence rows use the 40 sequence-level aggregates of
the ten columns on the final snapshot (max, mean, mean of the top five, max minus median),
fitted on train.

| Oracle | AUROC | AP | Detector minus, AUROC [CI] | Detector minus, AP [CI] |
|---|---|---|---|---|
| LR, final-state (6) | 0.5514 | 0.4369 | +0.357 [0.302, 0.411] | +0.457 [0.389, 0.512] |
| GB, final-state (6) | 0.5320 | 0.4364 | +0.377 [0.321, 0.432] | +0.457 [0.390, 0.511] |
| LR, temporal (7) | 0.6594 | 0.5732 | +0.249 [0.195, 0.301] | +0.320 [0.250, 0.383] |
| GB, temporal (7) | 0.6561 | 0.5589 | +0.253 [0.202, 0.306] | +0.335 [0.261, 0.397] |
| LR, all (13) | 0.6864 | 0.6035 | +0.222 [0.171, 0.276] | +0.290 [0.220, 0.355] |
| GB, all (13) | 0.6643 | 0.5617 | +0.245 [0.195, 0.297] | +0.332 [0.259, 0.396] |
| LR, nine features, final snapshot | 0.5353 | 0.4549 | +0.373 [0.313, 0.432] | +0.439 [0.368, 0.501] |
| LR, nine features, max over windows | 0.5403 | 0.4678 | +0.369 [0.310, 0.423] | +0.426 [0.354, 0.486] |
| LR, coherence (40) | 0.8351 | 0.7891 | +0.074 [0.035, 0.111] | +0.104 [0.056, 0.153] |
| GB, coherence (40) | 0.8034 | 0.7773 | +0.105 [0.064, 0.144] | +0.116 [0.072, 0.160] |
| LR, all (13) + coherence (40) | 0.8815 | 0.8416 | +0.027 [-0.007, 0.059] | +0.052 [0.010, 0.092] |

For reference, not paired with this detector: the step 1 arrival-run scan with logistic
regression, 0.8298 [0.793, 0.866] (`experiments/ceiling/ceiling_check.json`).

The detector beats every oracle on AP, and every oracle on AUROC except the regression given
both the thirteen temporal summaries and the coherence aggregates, which it does not separate
from on AUROC (+0.027, interval spanning zero).

## 6. Validation against test, like for like
Source `S2/seed_tables.json` (`table_4_1`), five seeds.

| Level | Validation | Test | Acceptance (0.85) |
|---|---|---|---|
| Online (freeze 1's windows) | 0.8387 (0.0029) | 0.8248 (0.0046) | Not met on either |
| Sequence | 0.9138 (0.0032) | 0.9088 (0.0032) | Met on both |

The validation-to-test drop at sequence level is 0.005, well inside the test interval
[0.8819, 0.9319], so the five rounds of selection on validation (IMPROVEMENT.md) flattered the
validation figure very little.

## 7. Online scores and sequence length (new)
Source `S2/online_drift_test.json` (`online_drift`, five seeds, mean over seeds). The detector
was trained on whole 20-snapshot sequences only; read after snapshot e it has seen snapshots 0
to e.

| Read after snapshot | AUROC, poisoned vs clean sequences | AUROC, attack present by then | Clean, median score | Clean, 95th percentile | Poisoned, median score |
|---|---|---|---|---|---|
| 9 | 0.633 | 0.733 | 0.352 | 0.990 | 0.786 |
| 11 | 0.717 | 0.783 | 0.143 | 0.991 | 0.840 |
| 13 | 0.789 | 0.823 | 0.055 | 0.989 | 0.955 |
| 15 | 0.844 | 0.851 | 0.017 | 0.979 | 0.980 |
| 17 | 0.891 | 0.891 | 0.008 | 0.972 | 0.993 |
| 19 (whole sequence) | 0.909 | 0.909 | 0.005 | 0.880 | 0.996 |

Pooled over end points 9 to 19 the online AUROC is 0.8248 (Table 4.1). Clean sequences score
high early and fall as more is seen, so a single threshold (Table 4.2, primary) is set by the
early scores: 0.968 to 0.999 across seeds at the 5% target; per-snapshot thresholds (Table 4.2, secondary) recover part of the
difference. Training on prefixes of random length would address it and was not run.

## 8. Duplicates benchmark (cross-scoring only)
Source `S2/cross_duplicates.json` (`coherence_plan cross-duplicates`; the five detector seeds
scored on `ap_burst2k`'s test split, 500 sequences, with the freeze 1 bootstrap).

| Trained on / tested on | AUROC mean (SD) [95% CI] | AP |
|---|---|---|
| Without / with duplicates (freeze 2 detector) | 0.9481 (0.0018) [0.9285, 0.9647] | 0.9392 |
| Without / with duplicates (freeze 1 detector) | 0.6852 (0.0062) | 0.6439 |

Training on the duplicates benchmark was not redone; freeze 1 section 6 remains the record of
why the benchmark changed.

## 9. Figures
Source `R2/figures_4_1_to_4_3.json` and the JSON beside each PNG.

| Figure | File | Seed | Test sequence | Source data |
|---|---|---|---|---|
| 4.1 Node score distribution | `R2/figures/fig_4_1_node_score_distribution.png` | 5 (median of 0.9033, 0.9089, 0.9093, 0.9110, 0.9113) | poisoned_placement_0047 | `.json` beside it |
| 4.2 Node graph overlay | `R2/figures/fig_4_2_node_graph_overlay.png` | 5 | poisoned_placement_0047 | `.json` beside it |
| 4.3 Detection score across a sequence | `R2/figures/fig_4_3_single_sequence_detection_score.png` | 5 | poisoned_placement_0047 | `.json` beside it |
| Missed example (limitations) | `R2/figures_missed_example/` (same three) | 5 | poisoned_placement_0463 | `.json` beside each |
| 4.4 Embedding noise | `R2/figures/fig_4_4_embedding_noise.png` | 1 to 5, 3 draws | full test partition | `.json` beside it |
| 4.4b Embedding noise, features recomputed (new) | `R2/figures/fig_4_4b_embedding_noise_full.png` | 1 to 5, 3 draws | full test partition | `.json` beside it |
| 4.5 Edge threshold | `R2/figures/fig_4_5_edge_threshold.png` | 1 to 5 | full test partition | `.json` beside it |
| 4.6 Metadata corruption and edge loss | `R2/figures/fig_4_6_metadata_and_edge_loss.png` | 1 to 5, 3 draws | full test partition | `.json` beside it |
| 4.7 Four degradations together | `R2/figures/fig_4_7_degradation_dashboard.png` | 1 to 5, 3 draws | full test partition | `.json` beside it |

Sequence rule as in freeze 1 (`rebuild.pick_sequence`): median attack size (12), onset at
snapshot 12 to 16, detected at the median latency of all detected test attacks (2), ties to the
lowest id; 4 of the 6 candidates were detected. Thresholds at a 5% benign false-alarm rate on
the whole validation partition: node 0.414, graph 0.9987 (online scores). On 0047 the detector
flags all 12 injected records and 12 benign ones, and alerts at snapshot 15 after onset at 13.
On 0463 (onset 16) it flags all 12 injected records and 8 benign ones but raises no sequence
alert: its online score peaks at 0.9982, below the 0.9987 threshold set by the early clean
scores (section 7).

Figures 4.4 to 4.7 plot the Tables 4.7 to 4.10 data as mean +/- SD over seeds and draws; with
one window per sequence they show sequence AUROC, sequence AP and node AUROC (the window lines
of freeze 1 coincide with the sequence lines here). Figure 4.4b plots the extra sweep.

## 10. Not done, and why
* **The ensemble beyond sequence level:** rank-normalised scores have no online or perturbed
  counterpart without new definitions; online, latency, sweeps, figures and ablations describe
  the detector.
* **Online scoring under perturbation:** about eight times the sweep cost.
* **Prefix training** (the remedy for section 7): a new configuration, not run.
* **Unsupervised baselines, Table 3.7, probes, training on the duplicates benchmark:** they read
  the data only, or belong to freeze 1's record of the benchmark change.
* **Gated readout, second threat model (E5), attack success (E6):** as freeze 1.
* **An attacker who spreads the injection out:** the coherence columns target consecutive
  bursts; no benchmark with spread-out injections was run, so robustness to that attacker is
  unmeasured.
* **Inference timings:** measured once.
* **Code provenance:** all runs used uncommitted code on top of 2dbcdbc; the commit awaits
  Moses's confirmation.

## 11. Reproducibility checks (each run twice from a fresh process, outputs compared)

Identical in every case: `S2/summary_headline.json`, `S2/seed_tables.json`,
`S2/ensemble_headline.json`, `S2/summary.json`, `S2/oracle_comparison_5seeds.json`,
`S2/cross_duplicates.json`, `S2/online_drift_test.json`.

R2 (3 November): every `rebuild` step after `online` was run a second time into a separate
folder from the saved scores, and its JSON compared with `R2`'s. Identical:
`table_4_4_static_nodes.json`, `table_4_4_record_level.json`, `table_4_6_probes.json`,
`figures_4_1_to_4_3.json`, `figures_missed_example_4_1_to_4_3.json`, the six figure JSONs of
Figures 4.1 to 4.3 and the missed example, and `tables_4_7_to_4_10_sweeps.json`. The five sweep
figure JSONs differ only in their `source` field, which names the output folder. The second
`sweeps` pass re-aggregated the per-seed sweep files; the sweeps themselves (about 16 GPU-h)
were run once.

Logs: `S2/PLAN.md`, `S2/plan.log`, `S2/phase_*.log`, `R2/STATUS.md`.
