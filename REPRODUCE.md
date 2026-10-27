# Reproducing the results freeze

`experiments/audit/RESULTS_FREEZE.md` is built in four stages. Times are for one NVIDIA Quadro
P2000 (4 GB); about 140 GPU-hours in all. Every long command is resumable: re-run it after an
interruption and it skips what is finished. Commands assume the setup in `README.md` and are run
from the repository root.

**What reproduces exactly and what does not.** Every summary script reads saved per-window scores
and is deterministic: run on the same scores, it writes the same file byte for byte (each was
checked by running it twice). Training on a GPU is not bit-reproducible, so retrained models give
slightly different scores; expect seed means within about one seed standard deviation (0.014 on
the main benchmark).

## Stage 0. Knowledge base

```
mkdir -p data/raw/agentpoison
curl -L -o data/raw/agentpoison/strategyqa_train_filtered.json https://raw.githubusercontent.com/AI-secure/AgentPoison/master/ReAct/database/strategyqa_train_filtered.json
```

## Stage 1. The earlier benchmark, ap_burst2k (needed for the duplicates comparison)

```
python -m src.data_generation.agentpoison_sequences --output data/raw/ap_burst2k --num-sequences 2000 --matched-control 1.0 --benign-spread 120 --attack-spread-mult 1.0 --seed 42
python -m src.data_generation.cached_dataset build --data data/raw/ap_burst2k
python -m src.experiments.sequence_eval --data data/raw/ap_burst2k --skip-model

# 25 runs: detector, static GAT, memory only, aggregator only, size-matched static; seeds 1-5 (~45 h)
python -m src.experiments.seed_queue --data data/raw/ap_burst2k --root experiments/seeds --models core --keep-awake

# the single checkpoint the pre-audit thesis reported, trained the legacy way (~3.5 h)
python -m src.models.train --data data/raw/ap_burst2k --output experiments/run_burst2k_base --structural-features --skip-phase1 --epochs-supervised 12 --seed 42 --memory-slots shared --cached
python -m src.experiments.score_windows --data data/raw/ap_burst2k --run-dir experiments/run_burst2k_base --cached --split test --batch-size 1 --node-scores --output experiments/audit/base_test_b1.npz

# summaries for this benchmark
python -m src.experiments.seed_summary --root experiments/seeds --output experiments/seeds/summary.json
```

`experiments/run_burst2k_base` is committed with its scores, so this last block can be skipped
unless you want to retrain it.

## Stage 2. R1, the duplicates study (~15 h)

Builds `ap_burst2k_nodup`, its cache and features, trains the detector and the static GAT on it
(seeds 1-3), scores every model on both benchmarks and writes `experiments/duplicates/SUMMARY.md`.

```
python -m src.experiments.duplicates_study run --seeds 3
```

## Stage 3. Everything on the main benchmark (~80 h)

Seeds 4-5 of the detector and static GAT, then the R2 rebuild (record-level metrics, probes,
Figures 4.1 to 4.3, robustness sweeps), then five seeds of the five ablations, then the latency
baseline and the summaries, each run twice and compared. Progress: `experiments/seeds_nodup/PLAN.md`.

```
python -m src.experiments.main_benchmark_plan
```

## Stage 4. Remaining freeze items (~40 min)

```
# main-benchmark seeds 4 and 5 scored on the earlier benchmark, then the R1 summary with five seeds
python -m src.experiments.score_windows --data data/raw/ap_burst2k --run-dir experiments/seeds_nodup/tgn_s4 --cached --split test --batch-size 4 --output experiments/duplicates/scores/tgn_s4_nodup_on_dup.npz
python -m src.experiments.score_windows --data data/raw/ap_burst2k --run-dir experiments/seeds_nodup/tgn_s5 --cached --split test --batch-size 4 --output experiments/duplicates/scores/tgn_s5_nodup_on_dup.npz
python -m src.experiments.duplicates_study summarize

# Table 3.7 position control, Table 4.4 unsupervised rows, Table 4.11 timings
python -m src.experiments.benchmark_gate --data data/raw/ap_burst2k_nodup --split test --max-sequences 500 --output experiments/rebuild/table_3_7_gate_nodup.json
python -m src.experiments.unsupervised_baselines --data data/raw/ap_burst2k_nodup --output experiments/rebuild/table_4_4_unsupervised_nodup.json
python -m src.experiments.compute_profile --data data/raw/ap_burst2k_nodup --run-dir experiments/seeds_nodup/tgn_s1 --output experiments/rebuild/table_4_11_profile_nodup.json

# Figures 4.4 to 4.7
python -m src.experiments.rebuild sweep-figures
```

## Where each table comes from

| Thesis item | File |
|---|---|
| Tables 4.1, 4.2, 4.3, validation against test | `experiments/seeds_nodup/seed_tables.json` |
| Table 4.5, headline intervals, noise floor | `experiments/seeds_nodup/summary.json` |
| Oracle comparison | `experiments/seeds_nodup/oracle_comparison_5seeds.json` |
| Table 4.4 | `experiments/rebuild/table_4_4_*.json` |
| Table 4.6 | `experiments/rebuild/table_4_6_probes.json` |
| Tables 4.7 to 4.10 | `experiments/rebuild/tables_4_7_to_4_10_sweeps.json` |
| Table 4.11 | `experiments/rebuild/table_4_11_profile_nodup.json` |
| Table 3.7 | `experiments/rebuild/table_3_7_gate_nodup.json`, `probes_raw.json`, `data/raw/ap_burst2k_nodup/sequence_eval_baselines.json` |
| Duplicates finding | `experiments/duplicates/summary.json` |
| Figures 4.1 to 4.7 | `experiments/rebuild/figures/` (each PNG has a JSON beside it) |

To re-check the summaries without retraining, re-run the summary commands above on the saved
scores and compare the output with the committed JSON; they should be identical.
