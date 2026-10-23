# T-GNN Audit Handover

Audit of 14 to 18 October 2026 for the thesis *Detection of Persistent Memory Poisoning in
Large Language Model Agents Using Temporal Graph Neural Networks* (Moses W. Ayandau,
PG3996225, KNUST). Web version: https://claude.ai/artifact/SU89wF92w2g4AT3X8gD6h2 (private to
your claude.ai account).

Details behind every figure: `AUDIT.md`. Every thesis edit: `THESIS_CHANGES.md`. Venues:
`manuscript/VENUES.md`.

## Summary

Brief v2 asked for ground truth first (E10, E7), then the oracle comparison (E1), more seeds
(E2), a component split and capacity control (E3) and an attack-size analysis (E4). All five
are done. The temporal result survives every check and is stronger; several numbers and
descriptions in the text were wrong and have exact replacements in the change list.

| Key figure | Now (five seeds) | Thesis |
|---|---|---|
| Sequence AUROC | 0.8026 (SD 0.0113) [0.7597, 0.8438] | 0.8122, one seed |
| Detector over static GAT | +0.249 [0.209, 0.289] | +0.2517 |
| Noise floor | 0.011, measured | 0.02, assumed |
| AUROC versus 13-feature oracle | -0.000, a tie (AP lead +0.092, significant) | 5.5 only |

## What happened

1. **14 Oct, afternoon.** Brief v2 received; thesis docx extracted and read; every reported
   number traced to files in `experiments/` (E10).
2. **14 Oct.** Training logs checked against Sections 3.9 and 4.2 (E7). Found that the
   "seed 1" run was seed 42, and that batched windows share node-memory slots; both
   confirmed by measurement.
3. **14 Oct.** Built a verified per-sequence data cache (bit-identical tensors, about 3x
   faster scoring), fixed the memory slots behind a switch, re-scored the reported checkpoint
   as the thesis describes it.
4. **14 Oct.** Oracle comparison with gradient boosting (E1), validation versus test like for
   like (E7), compute profile re-measured, repeat evaluations, latency re-derived, citations
   checked on the web, venue table compiled.
5. **14 Oct, evening.** Attack-size grid (E4), 36 cells. The laptop slept for 75 minutes
   mid-run; the session then ended.
6. **16 Oct, 14:58.** Decisions: keep ap_burst2k; five seeds of the five core configurations;
   train with the memory fix; keep the laptop awake only while jobs run. 25-run queue started.
7. **16 to 18 Oct.** Queue ran without interruption (about 45 GPU-hours on the Quadro P2000);
   progress in `experiments/seeds/STATUS.md`.
8. **18 Oct, 17:39.** All 25 runs finished; summaries computed and written into the change
   list (group C).

## Defects found

* **Fixed. The "seed 1" run was trained from seed 42.** `train.py` let `configs/default.yaml`
  override `--seed`. Both checkpoints correlate 0.790 with a fresh seed-42 initialisation and
  -0.0002 with seed 1, so the thesis rested on one seed. The seed is now honoured and recorded
  in `run_config.json`.
* **Fixed. Batched windows shared node-memory slots.** Node ids were not offset per window, so
  at batch size 4 node *i* of every window used one slot (330 collisions in one batch). All
  training and every reported evaluation ran this way; Section 3.3 says none did. Ranking
  barely changes when corrected (sequence AUROC 0.8144 vs 0.8194, not significant), but window
  scores move (correlation 0.90), so thresholded figures shift. Fix: `memory_slots:
  per_window` (old behaviour remains the default so old figures reproduce). Correct evaluation
  is deterministic to every digit.
* **Text. Training record.** Phase one (contrastive) never ran (`--skip-phase1`); 12 epochs,
  not 10; checkpoint chosen by highest validation AUROC (epoch 9), not lowest loss; focal
  class weight 0.85, not 0.92; objective weights 1.0 and 1.5, not 2.5; node prevalence about
  2.5%, not "fewer than 2%".
* **Text. Benchmark facts.** Cluster sizes 5 to 20, not 5 to 24 (Table 4.3 labels 1-8 and
  17-24 were wrong); clean clusters arrive over 7 to 120 interactions (median 75), not
  "across 120"; the split is a seeded shuffle, not stratified; edges use similarity > 0.6,
  not >=; retrieval frequency is divided by 50, not the trajectory maximum; clusters above 8
  records always contain duplicate entries.
* **Text. Figures without an artefact or with the wrong source.** Table 4.11 profiled an
  older 667,331-parameter model (re-measured); the "five evaluations 0.8259 to 0.8264" claim
  had no record; `latency_matched_fpr.json` had no producing script; Section 4.9 explains
  embedding noise by edge changes the code never makes; the "nine-feature" regression exists
  in two versions.
* **Text. Citations.** Xiang et al. (2024) is BadChain, an attack (Sections 1.2, 2.4);
  Shafran et al. (2025) is a denial-of-service attack (Section 2.8); MINJA (Dong et al., 2025)
  was evaluated on RAP, EHRAgent and a QA agent, not MemGPT (Section 1.6), and is NeurIPS
  2025, vol. 38, pp. 52119-52153; Li et al. (2026) needs volume and DOI from ScienceDirect;
  19 page ranges use en dashes.

## Five-seed results (E2, E3)

Seeds 1 to 5, memory fix on, test partition of 500 sequences (200 poisoned), sequence level.

| Configuration | AUROC mean (SD) | AP | Detector minus this [95% CI] | Parameters |
|---|---|---|---|---|
| Full detector | 0.8026 (0.0113) | 0.8126 | reference | 670,915 |
| Temporal aggregator removed (memory kept) | 0.7094 (0.0089) | 0.6638 | +0.093 [0.068, 0.118] | 571,843 |
| Node memory removed (aggregator kept) | 0.6520 (0.0153) | 0.6303 | +0.151 [0.120, 0.181] | 405,315 |
| Static GAT (both removed) | 0.5533 (0.0063) | 0.4590 | +0.249 [0.209, 0.289] | 306,243 |
| Static GAT widened to detector size | 0.5512 (0.0066) | 0.4591 | +0.251 [0.212, 0.291] | 670,233 |

The temporal advantage replicates at every seed; capacity does not explain it; both temporal
components are needed.

Table 4.1 replacement (window level unless stated, mean of five seeds; thesis in brackets):
graph AUROC 0.8413 (0.8261); graph AP 0.8013 (0.7769); F1 at 0.5 0.6854 (0.6691); FPR at 90%
TPR 0.5727 (0.5757); node AUROC 0.9034 (0.9087); node AP 0.5956 (0.5662); sequence AUROC
validation 0.8760, test 0.8026 (0.8122).

Acceptance criterion 0.85: met on validation (window 0.9093, sequence 0.8760), not on test at
either level. On the reported checkpoint the drop was compositional: for attacks of 9+ records
test (0.9131) was above validation (0.9028).

## Other results

**Oracles (E1, against all five seeds)**

| Oracle | AUROC | Detector minus, AUROC | Detector minus, AP |
|---|---|---|---|
| Logistic regression, 13 features | 0.8030 | -0.000 [-0.050, 0.055] | +0.092 [0.034, 0.157] |
| Gradient boosting, 13 features | 0.8204 | -0.018 [-0.071, 0.033] | +0.050 [-0.003, 0.102] |
| Logistic regression, 7 temporal | 0.7934 | +0.009 [-0.040, 0.063] | +0.102 [0.042, 0.168] |
| Logistic regression, 6 final-state | 0.4650 | +0.338 [0.283, 0.391] | +0.425 [0.367, 0.473] |

**Latency at a 5% false-alarm target (five seeds):** detected 140 of 200 on average (thesis
135), median 1.0 snapshot (thesis 2.0). Caught within 0 / 2 / 5 / 10 snapshots: detector 19.9 /
53.4 / 67.8 / 70.2%, regression 1.5 / 21.5 / 29.5 / 30.0%; differences +0.184 [0.138, 0.233],
+0.319 [0.258, 0.382], +0.383 [0.319, 0.452], +0.402 [0.339, 0.469]. At 10%: 155 of 200
against 98.

**Attack size, size-matched (five seeds):** 5-8: 0.520; 9-12: 0.891; 13-16: 0.913; 17-20:
0.968 (each band against clean controls with a benign cluster of the same size; pooled against
all clean sequences, small attacks fall below chance).

**E4 grid (reported checkpoint; sequence AUROC):**

| k | Dup, burst k | Dup, 40 | Dup, 80 | No dup, burst k | No dup, 40 | No dup, 80 |
|---|---|---|---|---|---|---|
| 5 | 0.521 | 0.599 | 0.577 | 0.673 | 0.613 | 0.558 |
| 8 | 0.624 | 0.525 | 0.484 | 0.571 | 0.532 | 0.505 |
| 9 | 0.794 | 0.604 | 0.499 | 0.690 | 0.532 | 0.448 |
| 12 | 0.830 | 0.696 | 0.509 | 0.767 | 0.567 | 0.517 |
| 16 | 0.963 | 0.736 | 0.588 | 0.697 | 0.650 | 0.494 |
| 20 | 0.958 | 0.877 | 0.622 | 0.634 | 0.615 | 0.474 |

Burst length matters at every size; the highest scores need duplicated entries. About +/-0.1
per cell; the detector was trained on data with duplicates.

**Inference cost (Table 4.11 replacement):** GPU 87.35 ms per window at batch 1 (11.45 per
second), 30.41 per second at batch 4; CPU 57.37 ms; peak GPU memory 22.8 to 31.8 MiB. The
static GAT ranks records at 0.8832 (Table 4.4 said "not measured").

## Thesis changes

All in `THESIS_CHANGES.md` (location, current text, proposed text, reason):

* **Group A, apply now (28 items):** objectives wording, 387 vs 391, three threat models,
  Section 3.9 training record, cluster sizes, arrival pattern and duplicates, stratification,
  ethics text, citations, references, Table 4.11, contributions count.
* **Group B, wording settled:** oracle tie in Chapter 4 and the Abstract, removing "outperforms
  every implemented alternative", Section 5.3 acceptance like for like, attack-size caution.
* **Group C, five-seed numbers:** headline, Table 4.1, noise floor, Table 4.5 with three new
  rows, Table 4.2, size-matched Table 4.3, oracle figures, acceptance.
* Front matter, title-page date and updating Word fields last (select all, F9) are your
  decisions (A28).

## Code changes (branch `benchmark-hardening`, not committed)

* `src/models/train.py`: seed honoured; `--memory-slots`, `--cached`; writes `run_config.json`.
* `src/models/tgn_detector.py`, `src/experiments/ablation_meanpool_model.py`: per-window memory
  slots behind a switch.
* `src/data_generation/cached_dataset.py` (new): verified per-sequence cache.
* `src/data_generation/agentpoison_sequences.py`: `--no-duplicates`, `--attack-spread` (off by
  default; ap_burst2k regenerates unchanged).
* `sequence_eval.py`, `detection_latency.py`, `generate_eval_results.py`: memory-slot,
  batch-size and cache options.
* New in `src/experiments/`: `score_windows`, `oracle_comparison`, `component_ablation_models`,
  `train_variant`, `seed_queue`, `seed_summary`, `seed_tables`, `attack_size_grid`,
  `compute_profile`, `robustness_sweep`.

## Files

* `experiments/audit/HANDOVER.md`: this report.
* `experiments/audit/AUDIT.md`: every figure traced to its source, with status.
* `experiments/audit/THESIS_CHANGES.md`: the change list.
* `manuscript/VENUES.md`: venues, deadlines, suggested route.
* `experiments/seeds/STATUS.md`, `summary.json`, `seed_tables.json`: five-seed run list and
  summaries.
* `experiments/seeds/<model>_s<seed>/`: each run's checkpoint, log, config and scores (933 MB).
* `experiments/audit/*.json`, `*.npz`: every audit measurement.

## Still open

* **E8 robustness sweeps** on the full test set with 3 draws: script ready, about 3 to 4
  GPU-hours.
* **Single-seed ablations** (mean pooling, gated readout, no structural features): about 2.8 h
  per run for five seeds.
* **E5, a second threat model** (slow or direct-write injection): the biggest lever for a
  stronger paper; without it the text should say "burst detection".
* **E6, attack success:** the generator builds no target query, so this needs a new design.
* **Venues:** WISTP 2027 (9 Dec 2026 or 2 Jan 2027, sources disagree), AsiaCCS 2027 round 2
  (11 Dec 2026), ICLR 2027 workshops (1 Feb 2027), IEEE Communications Magazine special issue
  on memory poisoning (15 Mar 2027).
* **Speed:** training waits on the CPU (GPU about 30% busy); removing a per-batch cache flush
  and sync points could give 1.5 to 2x without changing results.

## Continue in Claude on the web

Claude on the web cannot see your laptop, so bring the work to it in one of three ways.

**A. Open the web page.** Sign in at claude.ai and open
https://claude.ai/artifact/SU89wF92w2g4AT3X8gD6h2. It is private until you share it, and it
appears in your artifacts list.

**B. A claude.ai Project (recommended for the thesis text).**
1. On claude.ai choose Projects, then Create project; name it "T-GNN thesis".
2. Add to its knowledge: `experiments/audit/HANDOVER.md`, `experiments/audit/THESIS_CHANGES.md`,
   `experiments/audit/AUDIT.md`, `manuscript/VENUES.md`, the thesis docx
   (`thesis-pdf/MosesWinboodAyandauTGNN-3.docx`) and brief v2.
3. Paste these project instructions:

```
Thesis: Detection of Persistent Memory Poisoning in LLM Agents Using Temporal Graph Neural Networks (KNUST MSc).
Rules for any thesis text: British English; APA 7; no em or en dashes in prose; no equations in Chapter 3;
hardware is a Quadro P2000. Give change lists (location, current text, proposed text, reason), not rewritten
chapters, and never paraphrase passages (the text was revised to reduce Turnitin similarity).
Every number must come from HANDOVER.md, AUDIT.md or THESIS_CHANGES.md; do not invent figures or citations.
```

4. Start a chat with:

```
Read HANDOVER.md first. The audit and five-seed runs are finished; THESIS_CHANGES.md lists every edit.
Help me apply group A to the docx section by section, starting with Chapter 3, and flag anything that
conflicts with group C.
```

**C. Claude Code on the web (for the code; needs a push to GitHub).**
1. On the laptop, commit and push the branch, leaving out the large run files:

```
git add src manuscript experiments/audit/*.md experiments/audit/*.json experiments/seeds/STATUS.md experiments/seeds/*.json
git commit -m "Audit of the reported figures, memory-slot and seed fixes, five-seed results"
git push origin benchmark-hardening
```

2. Open claude.ai/code, connect GitHub if asked, choose
   `mwayandau1/tgn-detection-persistent-mem-poisoning` and the `benchmark-hardening` branch.
3. Give it the starter prompt above, pointing at `experiments/audit/HANDOVER.md`.

The 6.7 GB dataset, the checkpoints and the GPU stay on the laptop: a web session can read and
edit code and documents but cannot run the training.
