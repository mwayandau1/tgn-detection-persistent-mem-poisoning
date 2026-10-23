# Audit of the reported figures (brief v2, items E10 and E7)

Audited 14 October 2026 against `MosesWinboodAyandauTGNN-3.docx` (13 Oct 2026) and the
repository at commit `600f647` (branch `benchmark-hardening`). Every figure below was traced to
an artefact on disk and, where possible, recomputed from that artefact. Where no artefact
existed, the figure was re-measured or is marked as unsupported.

Scripts added for the audit: `src/experiments/score_windows.py`,
`src/experiments/oracle_comparison.py`, `src/data_generation/cached_dataset.py`. Outputs are
in this directory.

---

## 1. Findings that change what the thesis can claim

### 1.1 The "seed 1" run was trained from seed 42

`src/models/train.py` called `set_seed(config["dataset"]["seed"] or --seed)`, and
`configs/default.yaml` sets `dataset.seed: 42`, so `--seed` was ignored. Evidence that
`run_burst2k_base_s1` used seed 42: the flattened weights of both checkpoints correlate 0.790
with a freshly built seed-42 initialisation and -0.0002 with a seed-1 initialisation; the two
checkpoints correlate 0.982 with each other; their training curves agree epoch by epoch
(epoch-1 loss 0.5934 vs 0.5946, best epoch 9 in both).

Consequence: the thesis rests on **one** seed. The 0.0018 gap between "seed 42" and
"seed 1" measures GPU non-determinism in training, not seed variance, so the 0.02 noise floor
(Sections 3.14, 4.3, 4.7, 5.7) has no measured basis. The static and mean-pool baselines call
`set_seed(args.seed)` directly and are unaffected. Fixed in `train.py` (the CLI seed now
governs and is written to `run_config.json`).

### 1.2 Batched windows share node-memory slots

The dataset sets `node_ids = arange(n)` per snapshot, and PyG's `Batch.from_data_list` offsets
only attributes whose name contains "index", so with batch size > 1 node *i* of every window
in a batch reads and writes the same memory slot (330 shared slots in the first test batch).
With `shuffle=False` those windows are consecutive windows of one sequence.

Every training run used batch size 4, so the memory module was trained with cross-window
state. Every evaluation script defaulted to the training batch size, so the reported figures
were scored that way too. Section 3.3 states the opposite ("no figure reported in this thesis
derives from that configuration").

Measured effect at evaluation, same checkpoint (`run_burst2k_base/phase2_best.pt`), test split:

| Quantity | Batch 4, shared slots (as reported) | Batch 1 (as specified) |
|---|---|---|
| Sequence AUROC (window max) | 0.8144 [0.7715, 0.8565] | 0.8194 [0.7794, 0.8596] |
| Sequence AP | 0.8216 | 0.8142 |
| Paired difference, batch 4 minus batch 1 | -0.0051 [-0.0219, +0.0092] | |
| Window AUROC / AP | 0.8262 / 0.7776 | 0.8249 / 0.7693 |
| F1 at 0.5 / best F1 | 0.6684 / 0.6738 | 0.6778 / 0.6815 |
| Best-F1 threshold | 0.2561 (reported 0.2129) | 0.6717 |
| FPR at 90% TPR | 0.5792 (reported 0.5757) | 0.6068 |
| Node AUROC / AP | 0.9088 / 0.5660 | 0.9137 / 0.5794 |
| Correlation of per-window scores | 0.901 (mean abs. difference 0.062, max 0.985) | |

Ranking metrics move by at most 0.005 and no difference is significant. Individual window
scores move substantially, so every thresholded quantity (Table 4.2 latency, the F1 columns,
the figure operating points) differs between what was reported and the model as specified.
The suspected look-ahead did not inflate anything: pre-onset windows of poisoned sequences
score lower at batch 4 (mean 0.077) than at batch 1 (0.144).

Determinism: as specified, evaluation is deterministic. Batch-1 scoring (with and without the
cache) and five per-window batch-4 repeats all returned window AUROC 0.8248917939455518.
Under shared slots, three scorings gave 0.8260, 0.8261 and 0.8262. So the fourth-decimal
variation Section 4.3 attributes to scatter in attention pooling comes from colliding
memory writes, which also resolves brief item 6 (Section 3.3 vs 4.3).

Fix: `TGNDetector.memory_slots = "per_window"` offsets each window's ids by its batch
position. At batch 1 it is bit-identical to the old code; at batch 4 it reproduced the batch-1
test scores exactly (window AUROC 0.8248917939455518 both ways) at 3.7x the speed. The
default stays `"shared"` so the existing figures reproduce. **The effect of the defect on
training cannot be measured without retraining.**

### 1.3 The detector ties every temporal oracle, and gradient boosting is numerically ahead

`oracle_comparison_b4.json` (detector scores as reported):

| Model (500 test sequences) | AUROC [95% CI] | AP | Detector minus model, AUROC | Detector minus model, AP |
|---|---|---|---|---|
| LR, final-state (6) | 0.4650 [0.4146, 0.5190] | 0.3872 | +0.348 [+0.292, +0.404] | +0.426 [+0.372, +0.481] |
| LR, temporal (7) | 0.7934 [0.7526, 0.8311] | 0.7108 | +0.020 [-0.029, +0.073] | +0.107 [+0.046, +0.171] |
| LR, all (13) | 0.8030 [0.7630, 0.8397] | 0.7203 | +0.011 [-0.042, +0.065] | +0.097 [+0.038, +0.163] |
| GB, final-state (6) | 0.5436 [0.4889, 0.5970] | 0.4514 | +0.270 [+0.203, +0.338] | +0.362 [+0.287, +0.433] |
| GB, temporal (7) | 0.8108 [0.7707, 0.8468] | 0.7473 | +0.003 [-0.047, +0.055] | +0.072 [+0.014, +0.131] |
| GB, all (13) | 0.8204 [0.7832, 0.8542] | 0.7631 | -0.007 [-0.057, +0.045] | +0.056 [-0.000, +0.110] |
| LR, nine features, final snapshot | 0.4951 [0.4449, 0.5439] | 0.4005 | +0.318 [+0.260, +0.376] | +0.413 [+0.353, +0.468] |
| LR, nine features, max over windows | 0.5648 [0.5159, 0.6173] | 0.5221 | +0.248 [+0.197, +0.298] | +0.295 [+0.235, +0.357] |
| Detector (window max) | 0.8133 [0.7726, 0.8541] | 0.8190 | | |

GB = scikit-learn HistGradientBoosting; depth, learning rate and iterations selected on
validation AUROC from a 12-point grid, fitted on train only. The detector's AP lead over the
linear oracle is significant; over gradient boosting it is +0.056 with an interval touching
zero. All rows are single-seed. With the detector scored as specified (batch 1,
`oracle_comparison_b1.json`) the picture is the same: 0.8194 against gradient boosting's
0.8204, AUROC difference -0.0004 [-0.051, +0.048], AP difference +0.052 [-0.006, +0.109],
and against the linear oracle +0.017 [-0.034, +0.070] and +0.093 [+0.030, +0.162].

### 1.4 Detection tracks duplicated records at least as closely as attack size

The attacker samples from a pool of 8 neighbours (`--poison-pool 8`), without replacement when
k <= 8 and **with replacement** when k > 8. Every cluster of 9 or more records therefore holds
exact copies (cosine 1.0) and no cluster of 8 or fewer does. The matched clean clusters follow
the same rule. Measured on the test partition at batch 1 (`attack_size_by_k_b1.json`,
`cluster_composition_test.json`):

* Sequence AUROC by attack size is 0.55, 0.48, 0.45, 0.56 for k = 5, 6, 7, 8 and 0.78, 0.80,
  0.85, 0.91 for k = 9, 10, 11, 12, rising to 0.97 to 0.99 for k >= 17. The step sits exactly
  at the duplication threshold. The 5 to 8 band is at chance at sequence level (0.497).
* Clusters with duplicates carry 42.3 edges above threshold on average against 6.6 without.
* Within poisoned sequences, the detector's score correlates 0.644 with the number of
  duplicates and 0.622 with k.

The decoupling grid (`src/experiments/attack_size_grid.py`, `attack_size_grid.json`) crosses
duplication, size and burst length on the fixed reported detector: 50 poisoned sequences per
cell against 100 matched clean controls of the same size and duplication rule; sequence AUROC.

| k | dup, burst k | dup, 40 | dup, 80 | no dup, burst k | no dup, 40 | no dup, 80 |
|---|---|---|---|---|---|---|
| 5 | 0.521 | 0.599 | 0.577 | 0.673 | 0.613 | 0.558 |
| 8 | 0.624 | 0.525 | 0.484 | 0.571 | 0.532 | 0.505 |
| 9 | 0.794 | 0.604 | 0.499 | 0.690 | 0.532 | 0.448 |
| 12 | 0.830 | 0.696 | 0.509 | 0.767 | 0.567 | 0.517 |
| 16 | 0.963 | 0.736 | 0.588 | 0.697 | 0.650 | 0.494 |
| 20 | 0.958 | 0.877 | 0.622 | 0.634 | 0.615 | 0.474 |

Intervals are about +/-0.1 per cell (k <= 8 is the same process under both rules, so those
pairs show the cell-to-cell noise). Three readings:

1. **Burst length matters**: at every size the same records spread over 80 interactions fall
   to chance. The detector does respond to arrival rate, as the thesis claims.
2. **The high end is duplication**: with duplicates, tight bursts of 16 to 20 records reach
   0.96; without them the same sizes reach only 0.63 to 0.70. Without duplicates the size
   curve rises to about 0.77 at k = 12 and then flattens.
3. Caveat: the detector was trained on clusters that contain duplicates, so the no-duplicate
   columns measure transfer, not what a model trained without duplicates could learn.

### 1.5 Benchmark facts that differ from the text

| Thesis says | Ground truth | Evidence |
|---|---|---|
| Cluster size 5 to 24 (3.6, Table 3.3, 4.5) | 5 to 20 | `metadata.json`; generator default `--injection-range 5 20` |
| Bands 1-8 ... 17-24 (Table 4.3) | 5-8, 9-12, 13-16, 17-20 | the labels are bin edges hard-coded in `generate_eval_results.py` |
| Clean cluster arrives "gradually across 120 interactions" (3.6, Table 3.3) | First-to-last span 7 to 120 interactions, median 75; 18% within 40 | RNG replay of the generator, validated against stored labels (0 mismatches) |
| Attack is "a burst within a short span" | Exactly k consecutive interactions (`--attack-spread-mult 1.0`) | same replay |
| Partition "stratified by class" (3.7, 3.16, 6.2, 6.3) | Unstratified seeded shuffle that happens to give exactly 40.0% in each partition | current generator reproduces `splits.json` exactly |
| Edge where similarity "met or exceeded" 0.6 (3.5, Table 3.2) | Strictly greater than 0.6 | `agentpoison_sequences.py`, `sim > threshold` |
| Retrieval frequency normalised by the trajectory maximum (3.5, Table 3.1) | Divided by a fixed 50 | `agentpoison_sequences.py` line 238 |
| Domain "customer support" (3.7) | StrategyQA knowledge entries | config value left over from the LLM generator |
| Node prevalence "fewer than 2%" (3.9) | 2.49% train, 2.79% val, 2.54% test (window-final snapshots) | `_seqcache` labels |

### 1.6 Five-seed results (E2, E3), 16 to 18 October 2026

25 runs (`src.experiments.seed_queue --models core`, seeds 1 to 5, per-window memory, cached
loader; logs and `run_config.json` in each `experiments/seeds/<model>_s<seed>/`). Summaries:
`experiments/seeds/summary.json`, `seed_tables.json`, `size_matched_auroc.json`,
`oracle_comparison_5seeds.json`. Sequence level, test partition:

| Configuration | AUROC mean (SD) | Detector minus this [95% CI] |
|---|---|---|
| Detector | 0.8026 (0.0113) [0.7597, 0.8438] | |
| Temporal aggregator removed | 0.7094 (0.0089) | +0.0932 [0.0680, 0.1177] |
| Node memory removed | 0.6520 (0.0153) | +0.1506 [0.1203, 0.1810] |
| Static GAT | 0.5533 (0.0063) | +0.2493 [0.2087, 0.2892] |
| Static GAT at 670,233 parameters | 0.5512 (0.0066) | +0.2514 [0.2117, 0.2914] |

Noise floor (detector seed SD) 0.0113. Oracle tie holds (LR-13: -0.000 [-0.050, +0.055];
GB-13: -0.018 [-0.071, +0.033]). Latency improves over the reported run (140 of 200 detected
at 5%, median 1.0). Size-matched attack-size AUROC: 0.520, 0.891, 0.913, 0.968 for 5-8, 9-12,
13-16, 17-20. Thesis text for all of this: `THESIS_CHANGES.md` group C.

---

## 2. Training record (E7)

| Item | Thesis | Logs | Source |
|---|---|---|---|
| Phase one (contrastive) | Run for 5 epochs (3.9, Table 3.6, 3.16, 4.2, 6.2) | **Not run** (`--skip-phase1`); no `phase1_best.pt` in any run directory | notebook training command; run directories |
| Phase two epochs | 10 (3.9, Table 3.6) | 12 (`--epochs-supervised 12`) | `training_history.json` (12 entries) |
| Checkpoint rule | Lowest validation loss (3.9) | Highest validation window AUROC, epoch 9 (0.8979); lowest loss was epoch 7 (0.4491) | `training_history.json` `selection` |
| Focal class weight | 0.92 (3.9, Table 3.6) | 0.85 | `configs/default.yaml`, no override in the command |
| Objective weights | 1.0 graph, 2.5 node | 1.0, 1.5 | same |
| Cosine schedule length | not stated | T_max = 12 epochs | `train.py` |
| Early stopping | patience 10 | patience 10; never triggered in 12 epochs | config |
| Seed | 42, plus a second run under seed 1 | 42 twice (see 1.1) | weight correlation |
| Acceptance criterion | AUROC >= 0.85 on validation (3.12); "specified at window level" (5.3) | `evaluation.target_auroc: 0.85` with no level stated. The only validation AUROC recorded is the in-training window-level figure, scored at batch 4 with shared slots | config; `train.py` `_evaluate` |
| Validation vs test, like for like | val 0.8979 (window) vs test 0.8122 (sequence) | see table below | `base_val_*.npz`, `base_test_*.npz` |

Acceptance criterion like for like (`acceptance_like_for_like.json`; checkpoint
`run_burst2k_base/phase2_best.pt`; threshold 0.85):

| Level | Configuration | Validation | Test | Drop |
|---|---|---|---|---|
| Window | as reported (batch 4, shared slots) | 0.8982 (training log: 0.8979) | 0.8262 | 0.072 |
| Window | as specified (batch 1) | 0.8845 | 0.8249 | 0.060 |
| Sequence | as reported | 0.8556 [0.8085, 0.8991] | 0.8144 [0.7715, 0.8565] | 0.041 |
| Sequence | as specified | 0.8525 [0.8044, 0.8975] | 0.8194 [0.7794, 0.8596] | 0.033 |

The criterion is met on validation at both levels and on test at neither. The drop is
compositional, not a failure to generalise. Restricted to attacks of nine or more records,
sequence AUROC is 0.9028 on validation and 0.9131 on test. Attacks of eight or fewer are at
chance in both (0.5851 and 0.4966). The unstratified split put 15.8% of validation attacks and
22.5% of test attacks in that range.

---

## 3. Provenance of every reported figure (E10)

Status codes. **OK**: reproduced from the artefact. **OK (b4)**: reproduced, but scored with
shared memory slots at batch 4 (see 1.2). **Mismatch**: the artefact says something else.
**No artefact**: nothing on disk supports it.

| Thesis location | Figure(s) | Artefact | Produced by | Status |
|---|---|---|---|---|
| Table 3.4, 3.7 | 2,000 / 1,200 / 300 / 500 sequences, 40.0% poisoned, 13,200 / 3,300 / 5,500 windows, 1,769 positive test windows | `data/raw/ap_burst2k/{metadata,splits}.json` | `agentpoison_sequences --num-sequences 2000 --matched-control 1.0 --benign-spread 120 --attack-spread-mult 1.0 --seed 42` | OK; "stratified" is wrong |
| Table 3.5 | 200,064 / 334,464 / 8,321 / 99,072 / 8,321 / 20,673 = 670,915 | model instantiation | `TGNDetector`, `node_meta_dim = 7` | OK |
| Table 3.5, 3.8, 4.10 | Checkpoint 2.56 MB | `phase2_best.pt` is 2,695,493 bytes (2.57 MiB, 2.70 MB) | | Mismatch (2.557 came from the old 667,331-parameter profile) |
| 3.8, 4.7 | 667,331 without structural features; 662,594 mean-pool; 306,243 static | `run_burst2k_*` metadata, instantiation | | OK |
| Table 4.5 | Gated "Not recorded" | 671,686 | instantiation | Now known |
| Table 4.5 | Static mean pool "Not recorded" | 297,922 | `run_burst2k_static_mean/static_baseline.json` | Now known |
| Table 3.6 | Configuration | see section 2 | | Mismatch (phase one, epochs, focal weight, loss weights) |
| Table 3.7 | Per-record probe 0.5300, AP 0.0312, base rate 0.0256 | `ap_burst2k/separability_probe.json` | `separability_probe` (train -> test, about one window per sequence) | OK |
| Table 3.7, 4.6 | Pooled content 0.5166 / 0.3213; structural 0.6818 / 0.4594; combined 0.6893 / 0.4695 | same | same (cluster k = 15, unscaled) | OK |
| Table 3.7 | Position alone 0.6086 (window) | `ap_burst2k/benchmark_gate.json` | `benchmark_gate --split all --max-sequences 800`: the first 800 sequences of train+val+test, i.e. **800 training sequences** | OK, but not a test-partition figure |
| Table 3.7 | Node count 0.5000 | none needed: every sequence ends with 200 records | | OK by construction |
| Table 3.7, 5.5 | Final-state 0.4650 [0.4146, 0.5190]; temporal 0.7934; all 13 0.8030 | `ap_burst2k/sequence_eval_baselines.json` | `sequence_eval --skip-model` | OK |
| 3.10 | Final-state inflated "from 0.465 to approximately 0.70"; strongest single feature | recomputed: 0.4650 -> 0.7015; `snapshots_at_peak_maxdeg` 0.7190 | | OK |
| 4.2 | Losses, validation AUROC by epoch, epoch 9 selected | `run_burst2k_base/training_history.json` | `train` | OK |
| Table 4.1, 4.6 | 0.8261 / 0.7769 / 0.6691 / 0.6750 @ 0.2129 / 0.5757; node 0.9087 / 0.5662 | `run_burst2k_base/eval_results.json` | `generate_eval_results` (batch 4) | OK (b4); batch-1 values in 1.2 |
| Table 4.1, Abstract, 4.3, 4.11, 5.3, 6.2, 6.3, 6.6 | Sequence 0.8122 [0.7707, 0.8532], AP 0.8185 | `run_burst2k_base/sequence_eval.json` | `sequence_eval` (batch 4) | OK (b4); batch 1: 0.8194 [0.7794, 0.8596], AP 0.8142 |
| 4.3, Table 4.5 | Seed 1: 0.8140 / 0.8204, +0.0018 | `run_burst2k_base_s1/sequence_eval.json` | `train --seed 1` (seed ignored) | Mislabelled: it is seed 42 |
| 4.3 | Five evaluations, 0.8259 to 0.8264, SD 0.0002 | **none** | | No artefact |
| 3.12, Table 4.2 | 3.19% / 5.71% / 8.74% / 11.63%; 135 vs 60; 150 vs 98; medians; caught-within deltas and CIs | `run_burst2k_base/detection_latency.json` | `detection_latency` (batch 4) | OK (b4), reproduced exactly offline. As specified (`detection_latency_perwindow.json`): 3.50%, 133 vs 60; 7.80%, 143 vs 98; deltas +0.075 / +0.199 / +0.335 / +0.365, all intervals above zero |
| (not in thesis) | `latency_matched_fpr.json`, frozen into `reported_results.json` | | **no script writes it**; reproduced offline as thresholds matched on test benign windows (142 vs 56) | Orphan |
| Figures 4.1 to 4.3 text | node threshold 0.377; graph operating point; 188/12/19 records | `notebooks/ap_burst2k_figures.ipynb` outputs | batch 1, thresholds from 60 of 300 validation sequences | OK, re-verified from `base_test_b1.npz` (188 benign, 12 injected, 19 flagged of which 11 injected; trace below 0.08 before onset, then 0.900, 0.987, 0.995). The graph operating point (0.9784) is not Table 4.2's (0.9549) |
| Table 4.3, 4.5, 5.3, 5.7, 6.2, 6.6 | 0.5735 / 0.8220 / 0.9297 / 0.9554; node 0.6894 to 0.9864; 45 / 59 / 43 / 53 | `eval_results.json` `by_attack_size` | `generate_eval_results` (window level, batch 4) | OK (b4); band labels wrong |
| Table 4.4 | IF 0.6538 [0.6040, 0.7004] AP 0.5449, record 0.5029 / 0.0275; OCSVM 0.5719 [0.5183, 0.6242] AP 0.5044, record 0.5531 / 0.0284 | `ap_burst2k/unsupervised_baselines.json` | `unsupervised_baselines` | OK |
| Table 4.4, 4.5 | Static GAT 0.5602 [0.5085, 0.6106] AP 0.4684; paired 0.2517 [0.2049, 0.2995]; static mean 0.5424 | `run_burst2k_static_*` | `train_static_baseline` (no memory module, so batch-independent) | OK |
| Table 4.4 | Static GAT record-level AUROC "Not measured" | `static_attn_test.npz` | `score_windows --model static_attn --node-scores` | Now measured: 0.8832, node AP 0.3310 (window AUROC 0.6880) |
| Table 4.5 | Gated 0.8016 / 0.8095; mean-pool 0.6207 / 0.5836; no-struct 0.5078 / 0.4429 | `run_burst2k_{gated,meanpool,nostruct}` | `train` / `train_ablation_meanpool` (batch 4) | OK (b4) |
| 4.7 | "All three effects are significant" | recomputed paired CIs: mean-pool -0.1917 [0.1456, 0.2377]; no-struct -0.3049 [0.2424, 0.3645]; gated -0.0104 [-0.0033, 0.0243] | `sequence_eval.paired_boot` | OK |
| 3.11, 4.9, Tables 4.7 to 4.10 | Sweep figures | `run_burst2k_base/ood_results_burst2k.json` | `ood_evaluator` (batch 4, shared slots, one random draw) | OK (b4). Sample size, sequence count, seed and batch size are not recorded in the JSON. The five no-op conditions disagree on FPR at 90% TPR (0.5868 to 0.6218), more than several FPR "effects" in the tables |
| 4.10, Table 4.11 | 51.75 ms, 55.79, 55.90, 19.32 / 71.55 per second, 116.88 MB, CPU 53.70 / 134.28 | `experiments/computational_profile.json` (25 Aug) | `compute_profile.py`: **source deleted**, only `.pyc` remains | Mismatch: profiles a 667,331-parameter model on the earlier benchmark. Re-measured on the reported checkpoint in `compute_profile.json` (new `src/experiments/compute_profile.py`): GPU 87.35 ms / 11.45 per second at batch 1, 30.41 per second at batch 4; CPU 57.37 ms; peak 22.78 / 31.81 MiB |

### Other unreproducible or orphaned items

* Source deleted, bytecode only: `attack_type_breakdown`, `compute_profile`,
  `eval_holdout_checkpoint`, `shortcut_baseline`, `static_gnn_graph_ap`,
  `static_gnn_partial_visibility_probe`, `static_gnn_with_node_loss`, `structural_leak_probe`.
  None of their outputs is quoted in the current thesis except Table 4.11.
* `README.md` and `REPRODUCE.md` describe the removed synthetic benchmark.
* `notebooks/ap_burst2k_reproduction.ipynb` prints a hard-coded "Recorded in the thesis" latency
  table (142/200) that the current thesis does not contain.
* Two different "nine-feature" regressions exist: the probe (cluster k = 15, unscaled, window
  AUROC 0.6818) and the latency baseline (k = 12, standardised, 0.6925). Sections 3.10 and 3.11
  describe them as one.
* No run directory holds the configuration that produced it, although Section 3.3 says each
  does. `train.py` now writes `run_config.json`.
* Section 4.9 explains the embedding-noise curve by noise "perturbing the similarity matrix
  from which edges are constructed". `GraphNoiseTransform.add_embedding_noise` changes only
  the 384 embedding columns, redrawn per snapshot; edges, weights and structural columns are
  untouched. Edge loss drops directed entries independently, so it can leave a pair connected
  in one direction only. `src/experiments/robustness_sweep.py` re-runs all four sweeps on the
  full test partition with per-window memory and three draws (E8), keeping the perturbations
  as defined so the curves stay comparable.
* Section 6.5 says the simulator "constructs" a target query per injection. The ap_burst2k
  generator constructs none; that logic is in the superseded `llm_generator.py`. E6 needs a
  new design.
* `freeze_reference.py` copies the orphan `latency_matched_fpr.json` into
  `reported_results.json` under `latency_matched_fpr`; the thesis's Table 4.2 is
  `detection_latency.json`, which the frozen reference does not contain.
