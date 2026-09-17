# Reproducing the reported results

Every number reported for the hardened benchmark comes from the three commands below.
All of them run on CPU or a single small GPU; none of them call an external API.

Seeds are fixed at 42 (`configs/default.yaml -> dataset.seed`, and `--seed` on both CLIs).
Dataset generation is fully deterministic. Training is deterministic on CPU; on GPU the
non-deterministic scatter kernels used by PyTorch Geometric mean you should expect results
that are close but not bit-identical.

## 1. Generate the benchmark dataset

```bash
python -m src.data_generation.synthetic_generator \
  --output data/raw/synthetic_gap0125 \
  --difficulty benchmark \
  --num-sequences 300 \
  --seed 42
```

`--difficulty benchmark` expands to `topic_similarity=0.59`, `coherence=0.715`,
`session_length=12`, `retrieval_bias=0.0`. The generator prints a separability report and
writes it to `generation_report.json`; the value to check is
`coherence_gap = 0.1248`.

Difficulty is controlled by the **coherence gap**: how much more similar the attacker's
planted records are to each other than ordinary same-topic records are to each other.
Raise `--coherence` to make the attack easier to spot, lower it to make it harder. Do not
calibrate on `density_gap` — it thresholds the same quantity and discards the magnitude
the detector actually reads.

## 2. Train

```bash
python -m src.models.train \
  --data data/raw/synthetic_gap0125 \
  --output experiments/run_gap0125_struct \
  --structural-features \
  --skip-phase1 \
  --select-by auroc \
  --epochs-supervised 40 \
  --seed 42
```

The four non-default flags all matter:

| flag | why |
|---|---|
| `--structural-features` | Appends per-node degree and edge-weight statistics. `GATConv` normalises attention with a softmax, so its output is a convex combination of neighbours and it cannot count them. Without this the same run reaches 0.559 instead of 1.000 on the easy dataset. |
| `--skip-phase1` | The contrastive phase does not learn at `batch_size: 4` (see the analysis document, §4c.4). Skipping it halves wall-clock time and changes no reported metric. |
| `--select-by auroc` | Validation loss and validation AUROC diverge once overfitting starts, so loss-based checkpoint selection can save the worst model by the metric being reported. |
| `--epochs-supervised 40` | The config default of 10 stops while validation AUROC is still climbing. On the easy dataset the transition happens at epochs 8-10 — right where the default cuts off. |

Roughly 25 minutes on a Quadro P2000. `training_history.json` records the per-epoch curves
and a `selection` block naming the epoch that was saved (epoch 33, val AUROC 0.904 in the
committed run).

## 3. Evaluate

```bash
python -m src.experiments.generate_eval_results \
  --data data/raw/synthetic_gap0125 \
  --run-dir experiments/run_gap0125_struct \
  --checkpoint phase2_best.pt \
  --structural-features \
  --output experiments/eval_gap0125_struct.json
```

`--structural-features` must match the training run or the node feature width will not line up.

## Optional: forecast difficulty before training

```bash
python -m src.experiments.separability_probe --data data/raw/synthetic_gap0125
```

Fits a logistic regression on hand-crafted structural and content features. Runs in seconds
and predicted 0.9977 against a measured 0.9967 on the easy dataset, so it is worth running
on any newly generated benchmark before committing to a training run. A high probe AUROC
means the dataset has a cheap shortcut and the GNN is not being tested.

## What the committed run produced

`experiments/eval_gap0125_struct.json`, on a test split of **30 sequences (14 poisoned)**:

| metric | value |
|---|---|
| graph AUROC | 0.889 |
| graph average precision | 0.875 |
| graph best F1 | 0.774 |
| FPR at TPR 0.9 | 0.369 |
| node AUROC | 0.942 |
| node average precision | 0.650 |
| mean detection latency | 3.25 snapshots |

Thirty test sequences is a small sample. Windows within a sequence overlap heavily, so the
effective sample size is the sequence count, not the window count, and the standard error on
graph AUROC is roughly ±0.07. Report the difficulty curve rather than this single point
wherever possible.
