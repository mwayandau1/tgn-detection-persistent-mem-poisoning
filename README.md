# Temporal graph detection of persistent memory poisoning in LLM agents

Code for the MSc thesis *Detection of Persistent Memory Poisoning in Large Language Model Agents
Using Temporal Graph Neural Networks* (Moses W. Ayandau, KNUST).

An agent's long-term memory is represented as a sequence of graphs: one node per stored record
(its sentence embedding plus behavioural and structural attributes), edges between records whose
embeddings are similar. A temporal graph neural network reads ten consecutive snapshots and
decides whether the memory is being poisoned. Poisoning is simulated as *flood placement*: an
attacker writes a burst of genuine knowledge-base entries into one narrow region of the embedding
space, and every poisoned sequence is paired with a clean control carrying an equally large and
equally tight benign cluster that arrives gradually. The only thing separating the classes is the
rate at which the cluster arrives.

## Results

All reported numbers are in **[`experiments/audit/RESULTS_FREEZE.md`](experiments/audit/RESULTS_FREEZE.md)**,
each with the file it comes from. Main benchmark `ap_burst2k_nodup`, five training seeds, 500 test
sequences:

| | Sequence AUROC |
|---|---|
| Detector | 0.6614 (SD 0.0144) |
| Static graph model, same capacity | 0.5410 (detector leads by 0.12) |
| Without per-node memory | 0.5841 |
| Without temporal aggregator | 0.6515 |
| Logistic regression on 13 hand-built temporal features | 0.6864 (a tie) |

The temporal gain is real and lives in the per-node memory; the detector matches, but does not
beat, features hand-built from knowledge of the attack. An earlier version of the benchmark
(`ap_burst2k`) let the detector score 0.80 by reading bursts of duplicated records; that finding
and why the benchmark was rebuilt are in [`experiments/duplicates/SUMMARY.md`](experiments/duplicates/SUMMARY.md).

## Repository layout

```
configs/default.yaml          every setting the code reads (graph, model, training)
src/
  data_generation/
    agentpoison_sequences.py  builds a benchmark from the AgentPoison StrategyQA knowledge base
    dataset.py                windows of snapshots as PyTorch Geometric graphs (+ structural features)
    cached_dataset.py         the same windows from a per-sequence cache; bit-identical, much faster
    noise_transforms.py       the four input degradations used in the robustness sweeps
  models/
    tgn_detector.py           the detector: node memory, graph attention, attention pooling, GRU
    train.py                  Trainer (loss, optimiser, checkpoint rule) and a single-run CLI
  experiments/
    train_variant.py          train any model variant with the reported recipe, then score it
    seed_queue.py             many variants x seeds, resumable, with STATUS.md and keep-awake
    main_benchmark_plan.py    everything behind the results freeze, in order
    duplicates_study.py       R1: with versus without duplicated records
    rebuild.py                Tables 4.4, 4.6 to 4.10 and Figures 4.1 to 4.7 from the trained seeds
    score_windows.py          score a checkpoint on a split at a stated batch size
    seed_summary.py           mean, SD and paired differences across seeds (Tables 4.1, 4.5)
    seed_tables.py            latency (4.2), attack size (4.3), validation versus test
    oracle_comparison.py      the detector against engineered-feature models
    sequence_eval.py          the 13 sequence features and their regressions
    separability_probe.py     attribution probes (structural, content, per record)
    benchmark_gate.py         benchmark validation controls (position, single features)
    unsupervised_baselines.py Isolation Forest and One-Class SVM
    detection_latency.py      calibrated latency against the nine-feature regression
    complementarity.py        nine-feature window statistics (used by detection_latency)
    robustness_sweep.py       the four degradation sweeps on the full test partition
    compute_profile.py        inference time and memory (Table 4.11)
    attack_size_grid.py       E4: size, duplication and burst length varied apart (audit analysis)
    static_gnn_model.py, ablation_meanpool_model.py, component_ablation_models.py
                              the ablation architectures
experiments/
  audit/        RESULTS_FREEZE.md (the numbers), THESIS_CHANGES.md (edits for the thesis),
                AUDIT.md (what was wrong with the earlier results and why)
  seeds_nodup/  the 35 main-benchmark runs (7 configurations x 5 seeds) and their summaries
  seeds/        25 runs on the earlier benchmark (used for the duplicates comparison)
  duplicates/   R1 results
  rebuild/      Tables 4.4, 4.6 to 4.11, Figures 4.1 to 4.7, each with source JSON
  run_burst2k_base/  the single checkpoint the pre-audit thesis reported (R1 compares against it)
manuscript/VENUES.md          candidate venues and deadlines for the paper
archive/llm_benchmark/        the withdrawn LLM-generated benchmark (evidence for the content leak)
data/  (not in Git)           knowledge base, generated benchmarks and caches
```

Each run folder holds `run_config.json` (exact settings, seed, commit), `training_history.json`,
`variant.json` (parameters, test AUROC) and, outside Git, its checkpoint, log and scores.

## Setup

Python 3.12. Install PyTorch for your CUDA version first, then the rest:

```
python -m venv .venv
.venv\Scripts\activate                      (Linux/macOS: source .venv/bin/activate)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt` pins the versions that produced the results. Everything runs on a 4 GB GPU
(the reported runs used an NVIDIA Quadro P2000) and also on CPU, more slowly.

## Quick start

```
# 1. knowledge base (AgentPoison's StrategyQA base); its embeddings are computed on first use
mkdir -p data/raw/agentpoison
curl -L -o data/raw/agentpoison/strategyqa_train_filtered.json https://raw.githubusercontent.com/AI-secure/AgentPoison/master/ReAct/database/strategyqa_train_filtered.json

# 2. the main benchmark (about 30 min, CPU) and its cache
python -m src.data_generation.agentpoison_sequences --output data/raw/ap_burst2k_nodup --num-sequences 2000 --matched-control 1.0 --benign-spread 120 --attack-spread-mult 1.0 --seed 42 --no-duplicates
python -m src.data_generation.cached_dataset build --data data/raw/ap_burst2k_nodup

# 3. train and score one detector (about 3 h on a P2000)
python -m src.experiments.train_variant --data data/raw/ap_burst2k_nodup --model tgn --seed 1 --output experiments/my_run --cached
```

The last command writes the checkpoint, the per-window validation and test scores, and
`variant.json` with the test AUROC. To rebuild every reported number, follow
[`REPRODUCE.md`](REPRODUCE.md).

## Things a reader should know

* **Evaluate at sequence level.** Window labels turn positive once an injection exists, and
  window position alone predicts them (AUROC about 0.6). Every sequence has the same length, so
  sequence-level scores (window max) carry no such floor.
* **Memory slots.** Node memory is addressed per window (`memory_slots: per_window`, the default).
  The earlier default, `shared`, let windows in a batch overwrite each other's memory; it is kept
  only to re-score the pre-audit checkpoint. At batch size 1 the two are identical.
* **Seeds.** `--seed` sets the seed (an earlier bug let the config override it). Training on a GPU
  is not bit-reproducible; evaluation of a saved checkpoint is.
* **Structural features.** Graph attention averages over neighbours and cannot count them, so
  each node also gets its degree and incident-edge statistics (4 columns, 391 features in all).
  Without them the detector is at chance.
* **The cache.** `cached_dataset.py` stores each sequence once instead of re-reading 10 compressed
  files per window. It is verified tensor by tensor against the original loader.

## Names in code comments that refer to retired data

Comments explaining design choices sometimes cite measurements on data that is no longer used:

| Name | What it was |
|---|---|
| `synthetic_gap0125`, `synthetic_hard`, `synthetic_medium` | early synthetic benchmarks with controlled cluster separation, removed |
| `llm_generated`, `llm_hardened` | the LLM-generated benchmark, withdrawn for leaking its label through record text (`archive/llm_benchmark/`) |
| `ap_sequences`, `ap_matched_2k` | intermediate corpora from the same knowledge base, before the matched-control and burst settings were fixed |
| `ap_burst2k` | the benchmark of the pre-audit thesis; clusters repeat entries (see R1) |
| `ap_burst2k_nodup` | the main benchmark |

Everything removed from the repository in the October 2026 cleanup is preserved in Git under the
tag `pre-cleanup`.
