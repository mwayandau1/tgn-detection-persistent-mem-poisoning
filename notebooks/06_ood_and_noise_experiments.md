> # SUPERSEDED
>
> This walkthrough documents a notebook that runs on `data/raw/llm_generated`.
>
> **That benchmark has been withdrawn.** A logistic regression given a single record's
> 384-dimensional embedding -- no graph, no edges, no temporal order -- scores **AUROC 1.0000**
> on it. The label was recoverable from one record's text, so no graph model was being tested.
> On the replacement benchmark `ap_burst2k` the same probe scores 0.5300.
>
> Where this document says the detector reaches "graph-level AUROC ~ 1.0", that number is real
> but is evidence the benchmark was defective, not evidence about the detector. The amended
> headline figure is **sequence-level AUROC 0.8122, AP 0.8185** on `ap_burst2k`.
>
> The reasoning here about leakage, shortcuts and evaluation artefacts is still sound -- it is
> the reasoning that eventually caught the defect. Only the numbers are withdrawn.
>
> Current results: `notebooks/ap_burst2k_reproduction.ipynb`,
> `notebooks/ap_burst2k_baselines.ipynb`, `notebooks/ap_burst2k_ablations.ipynb`.

# Notebook 06 — OOD & Noise Stress-Testing: A Full Walkthrough

This document explains **every cell, function, and variable** in
[`06_ood_and_noise_experiments.ipynb`](06_ood_and_noise_experiments.ipynb): what it does, why it
exists, and what the numbers/plots it produces actually mean.

---

## 1. Purpose of the notebook

Notebook 05 trains the main TGN detector and reports graph-level AUROC ≈ 1.0 on the clean,
in-distribution held-out test split. That number says nothing about what happens once the world
stops looking exactly like the training distribution. Notebook 06 is the **robustness audit**: it
takes the same trained checkpoint and asks four separate "what if the input degrades in a
specific, realistic way?" questions:

1. **Gaussian embedding noise** — what if the sentence-embedding model producing node features
   drifts, or its outputs pick up noise (a different encoder version, floating-point
   nondeterminism, noisy LLM-generated text)?
2. **Cosine-similarity edge threshold shift** — the memory graph's edges only exist because the
   data generator drew them at cosine similarity ≥ `0.6` (`semantic_edge_threshold` in
   `configs/default.yaml`). What if the *deployed* system uses a different threshold than the one
   used to build the training graphs?
3. **Metadata corruption** — the last 3 feature columns per node (recency, retrieval frequency,
   retrieval recency) come from operational logs, which can go missing or get corrupted. What
   happens if some fraction of them are replaced with noise?
4. **Random edge loss** — what if the memory index used to build the graph misses some
   relationships (sparse/incomplete indexing)?

Unlike notebook 07, **this notebook does not run any model inference itself.** It is purely a
**reporting/visualization layer**: it loads a pre-computed results file,
[`experiments/ood_results.json`](../experiments/ood_results.json), and turns it into tables and
publication-quality plots. The actual perturb-and-score work happens in a standalone script,
[`src/experiments/ood_evaluator.py`](../src/experiments/ood_evaluator.py), which must be run first
to (re)generate that JSON file. This split mirrors notebook 07's pattern of keeping expensive
compute out of notebook cells — here the compute isn't hours of training, but the notebook still
treats the JSON file as the single source of truth rather than recomputing it inline.

This notebook is the visual counterpart to the noise-robustness section of
[`docs/near_perfect_metrics_defense.md`](../docs/near_perfect_metrics_defense.md) (§3 row 8, §4.3)
— that document narrates *why* the numbers look the way they do (including a real bug that was
found and fixed in the noise-injection code itself); this file explains what each notebook cell
does with those numbers, and reports what's actually in the current `ood_results.json`.

---

## 2. Where the numbers actually come from: `src/experiments/ood_evaluator.py`

Before walking through the notebook cells, it's worth understanding the script that produced the
JSON they read, since the notebook's variable names (`sigmas`, `thetas`, `m_p`, `e_r`) are just
the JSON's keys converted back to floats.

### `calculate_fpr_at_tpr90(y_true, y_scores)`

Finds the False Positive Rate at the point on the ROC curve where True Positive Rate first reaches
≥ 90% (`sklearn.metrics.roc_curve` gives `(fpr, tpr, thresholds)` arrays; `np.where(tpr >= 0.90)[0]`
finds the first index meeting that bar, and the FPR at that same index is the answer). Returns
`0.0` if only one class is present (metric undefined) and `1.0` if TPR never reaches 90% at any
threshold (worst-case fallback). This complements AUROC with an operationally meaningful number:
"if I insist on catching 90% of poisoned trajectories, how many false alarms do I pay for that?"

### `preload_dataset_to_gpu(dataset, device, max_samples=100)`

Loads up to `max_samples` test-set windows once and moves every graph snapshot's tensors to
`device` (GPU) up front, returning them as a plain Python list of dicts. This exists purely for
speed: the same 100 preloaded items are reused across the *entire* sweep (baseline + 7 noise
levels + 5 thresholds + 5 corruption levels + 5 edge-drop rates = 23 evaluation passes), so paying
the CPU→GPU transfer cost once instead of 23 times matters when each pass otherwise involves
repeated perturbation and inference.

**Historical note:** this preload-once-reuse-many-times design is exactly what surfaced a real bug
in an earlier version of the noise-injection code — see §6 below.

### `evaluate_preloaded_data(model, preloaded_items, device, batch_size, embedding_sigma, threshold, meta_corrupt_prob, edge_drop_rate)`

The core evaluation loop, run once per sweep point:

- Iterates the preloaded items in batches of `batch_size` (4, matching training).
- For each item, calls `GraphNoiseTransform.perturb_trajectory(...)` (see §3) with whichever single
  perturbation parameter is non-default for this sweep point — every other parameter stays at its
  no-op default, so each sweep isolates one perturbation type at a time.
- Re-collates the perturbed graphs with `collate_trajectories` and runs the model forward exactly
  as in training/eval elsewhere in the project.
- Accumulates `graph_targets`/`graph_preds` (one per window) and `node_targets`/`node_preds` (from
  only the final snapshot of each window, same convention as notebook 07).
- Computes six metrics: `graph_auroc`, `graph_ap`, `graph_f1` (at the fixed 0.5 threshold),
  `fpr_at_tpr90`, `node_auroc`, `node_ap` — all rounded to 4 decimal places, matching exactly the
  columns seen in the notebook's `df_noise`/`df_thresh` tables.
- Metric fallbacks: `g_auroc`/`g_ap` default to `1.0` (not `0.0`, unlike notebook 07's convention)
  when only one graph label class is present in the batch pool; `n_auroc`/`n_ap` do the same for
  node labels. **This fallback choice matters for reading the tables below** — a `1.0` in this
  notebook's tables is not automatically "the model is perfect," it can also mean "this batch
  happened to have no negative examples."

### `run_all_ood_experiments(...)`

The orchestration function `python -m src.experiments.ood_evaluator` runs:
1. Loads config, model checkpoint, and preloads 100 test-set windows.
2. Runs the **clean baseline** once (all perturbation params at their no-op defaults) →
   `results["baseline_clean"]`.
3. **Embedding noise sweep**: `sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]`.
4. **Threshold sweep**: `thetas = [0.4, 0.5, 0.6, 0.7, 0.8]` (0.6 is the training-time value).
5. **Metadata corruption sweep**: `probs = [0.0, 0.05, 0.10, 0.20, 0.30]`.
6. **Edge-loss sweep**: `rates = [0.0, 0.05, 0.10, 0.20, 0.30]`.
7. Writes everything to `experiments/ood_results.json` (the notebook's only input).

---

## 3. The perturbations themselves: `src/data_generation/noise_transforms.py`

`GraphNoiseTransform` is a collection of static methods, one per perturbation type, all operating
on a single PyG `Data` snapshot and returning a **new** `Data` object (never mutating the input in
place — see §6 for why that specific guarantee was hard-won).

- **`add_embedding_noise(data, sigma, embedding_dim=384, device=None)`** — clones `data.x`, adds
  `N(0, sigma²)` Gaussian noise to only the first `embedding_dim` columns (the sentence-embedding
  portion), leaving the last 3 metadata columns untouched. No-ops if `sigma <= 0.0`.
- **`apply_edge_threshold_shift(data, new_theta, embedding_dim=384, device=None)`** — **recomputes
  the entire graph structure from scratch** at a new cosine-similarity threshold, rather than
  editing the existing edges: L2-normalizes the embedding columns, computes the full pairwise
  cosine-similarity matrix (`norm_emb @ norm_emb.T`), masks out self-loops, keeps only pairs with
  similarity ≥ `new_theta`, and rebuilds `edge_index`/`edge_attr` from that mask. This means a
  lower `new_theta` produces a **denser** graph (more pairs clear the bar) and a higher one
  produces a **sparser** graph — important for reading the threshold sweep's shape in §5.
- **`corrupt_metadata_jitter(data, corrupt_prob, embedding_dim=384, device=None)`** — for each node
  and each of the 3 metadata columns independently, with probability `corrupt_prob`, replaces that
  value with fresh `Uniform(0, 1)` noise (`meta_mask` is an independent Bernoulli draw per
  node×column, not per node — so a node can have 1, 2, 3, or 0 of its metadata fields corrupted).
- **`drop_random_edges(data, drop_rate, device=None)`** — keeps each edge independently with
  probability `1 - drop_rate` (`keep_mask = rand >= drop_rate`), then filters both `edge_index` and
  `edge_attr` by that mask. No-ops if the graph already has zero edges.
- **`perturb_trajectory(graphs, embedding_sigma, threshold, meta_corrupt_prob, edge_drop_rate, device)`**
  — applies whichever of the four transforms above have a non-default argument, **in a fixed
  order** (embedding noise → threshold shift → metadata corruption → edge drop), independently to
  every one of the `T` snapshots in a trajectory window. In every sweep the evaluator runs, only
  one of the four parameters is ever non-default at a time, so in practice this behaves as four
  separate single-factor experiments, not a combined one — the notebook's plots only ever describe
  one dimension moving at a time.

---

## 4. Notebook cells

### Cell 1 — Setup

```python
plt.style.use('seaborn-v0_8-whitegrid' if ... else 'default')
EXP_DIR = Path('../experiments')
IMG_DIR = EXP_DIR / 'eval_images'
IMG_DIR.mkdir(parents=True, exist_ok=True)
```

Standard plotting setup: picks a whitegrid style if the installed matplotlib/seaborn version has
it (falls back gracefully rather than failing on an older library version), fixes a consistent
font size, and ensures `experiments/eval_images/` exists so every `plt.savefig` call later in the
notebook has somewhere to write. `EXP_DIR`/`IMG_DIR` are reused by every later cell.

### Cell 3 — Load results, print clean baseline

```python
results_file = EXP_DIR / 'ood_results.json'
with open(results_file) as f: results = json.load(f)
```

Loads the whole precomputed JSON into `results` (a dict with 5 top-level keys: `baseline_clean`,
`embedding_noise_sweep`, `threshold_sensitivity_sweep`, `metadata_corruption_sweep`,
`edge_loss_sweep` — exactly the structure `run_all_ood_experiments` writes). Prints the clean
baseline metrics as a sanity check that the file loaded and matches expectations before doing
anything with the sweeps.

**Current values in `experiments/ood_results.json`:**

| metric | value |
|---|---|
| graph_auroc | 1.0 |
| graph_ap | 1.0 |
| graph_f1 | 0.9822 |
| fpr_at_tpr90 | 0.0 |
| node_auroc | 0.9737 |
| node_ap | 0.634 |

This matches the main evaluation numbers reported elsewhere in the project (notebook 05 /
`docs/near_perfect_metrics_defense.md`) — as it should, since `sigma=0`/`theta=0.6`/`p=0`/`rate=0`
here is just the unperturbed test set scored through the same pipeline.

### Cells 5 — Embedding noise plot and table

```python
noise_data = results['embedding_noise_sweep']
sigmas = [float(k) for k in noise_data.keys()]
g_auroc = [v['graph_auroc'] for v in noise_data.values()]
g_ap = [v['graph_ap'] for v in noise_data.values()]
n_auroc = [v['node_auroc'] for v in noise_data.values()]
```

Unpacks the sweep dict's string keys (`"0.0"`, `"0.05"`, ...) into a float list `sigmas` for
plotting, and pulls out three metric series in parallel arrays (relying on Python dict ordering
matching insertion order, which holds here since the JSON was written in sweep order). Plots all
three against `sigmas` on one axes, with a horizontal reference line at `y=0.85` — this is
`config["evaluation"]["target_auroc"]` from `configs/default.yaml`, hardcoded here rather than
read from config, marking the minimum bar the project set for itself. `df_noise` is just the same
dict reshaped into a pandas DataFrame (`pd.DataFrame(noise_data).T` transposes so sigma values
become rows) for a readable tabular `display()`.

**Current values:**

| σ | graph_auroc | graph_ap | graph_f1 | fpr_at_tpr90 | node_auroc | node_ap |
|---|---|---|---|---|---|---|
| 0.00 | 1.0000 | 1.0000 | 0.9822 | 0.0000 | 0.9737 | 0.6333 |
| 0.05 | 0.9996 | 0.9990 | 0.9762 | 0.0000 | 0.9722 | 0.6239 |
| 0.10 | 0.9981 | 0.9957 | 0.9822 | 0.0000 | 0.9712 | 0.6125 |
| 0.15 | 0.9932 | 0.9924 | 0.9497 | 0.0000 | 0.9687 | 0.5448 |
| 0.20 | 0.9298 | 0.8925 | 0.4311 | 0.1270 | 0.9296 | 0.4155 |
| 0.25 | 0.7283 | 0.5152 | 0.4135 | 0.4385 | 0.8580 | 0.3197 |
| 0.30 | 0.5635 | 0.2837 | 0.4135 | 0.6885 | 0.8228 | 0.2596 |

**Reading this:** AUROC (ranking quality) holds essentially flat through σ=0.15, then degrades
sharply — 0.93 at σ=0.20, 0.73 at σ=0.25, down to 0.56 (near chance) at σ=0.30. Note the
**`graph_f1` collapse happens earlier and more sharply than AUROC** — it drops from 0.95 to 0.43
between σ=0.15 and σ=0.20, a full step before AUROC shows serious damage. This is a *threshold
miscalibration* effect, not a ranking failure: F1 is computed at a fixed 0.5 cutoff, so once noise
shifts the score distribution enough, the fixed threshold stops separating classes well even
though the model can still rank poisoned windows above clean ones (AUROC 0.93 at that same point).
This is exactly why the sweep reports both metrics rather than F1 alone.

### Cells 7 — Threshold sensitivity plot and table

```python
thresh_data = results['threshold_sensitivity_sweep']
thetas = [float(k) for k in thresh_data.keys()]
```

Same pattern as the noise cell, plotted against `thetas` instead, with a vertical reference line
at `theta=0.6` marking the threshold actually used to build the training graphs
(`semantic_edge_threshold` in config) — every other θ value represents a *mismatch* between
training-time and evaluation-time graph construction.

**Current values:**

| θ | graph_auroc | graph_ap | graph_f1 | fpr_at_tpr90 | node_auroc | node_ap |
|---|---|---|---|---|---|---|
| 0.4 | 0.9985 | 0.9968 | 0.9576 | 0.0000 | 0.9563 | 0.3127 |
| 0.5 | 1.0000 | 1.0000 | 0.9701 | 0.0000 | 0.9676 | 0.5090 |
| 0.6 | 1.0000 | 1.0000 | 0.9822 | 0.0000 | 0.9737 | 0.6339 |
| 0.7 | 1.0000 | 1.0000 | 0.9383 | 0.0000 | 0.9733 | 0.6497 |
| 0.8 | 0.8996 | 0.8484 | 0.6870 | 0.4467 | 0.9981 | 0.9070 |

**Reading this — the non-obvious result in this notebook:** graph-level AUROC is essentially flat
at 1.0 across θ=0.4–0.7, then drops to 0.90 at θ=0.8 (a much sparser graph, since only near-duplicate
embeddings clear a 0.8 cosine bar). But **node-level performance moves in the opposite direction**
at that same point — node AUROC actually *rises* to 0.9981 (above every other value in the sweep,
including the θ=0.6 baseline's 0.9737) and node AP jumps to 0.9070 (vs. 0.6339 at baseline). A very
sparse graph apparently hurts the *graph-level pooling* mechanism (fewer edges means less
structural signal to detect a dense poisoned sub-clique — the exact mechanism
`docs/near_perfect_metrics_defense.md` §5 identifies as the model's core signal) while *helping*
per-node scoring (with fewer confounding edges, each node's own embedding content dominates its
score, and content alone is already a reasonably strong signal per notebook 07's content-probe
AUROC of 0.77). This is a genuine graph-vs-node trade-off surfaced only by this sweep, not visible
in any single-metric summary.

### Cell 9 — Metadata corruption & edge loss (side-by-side subplots)

```python
meta_data = results['metadata_corruption_sweep']
edge_data = results['edge_loss_sweep']
fig, axes = plt.subplots(1, 2, ...)
```

Two independent line plots sharing one figure: left panel is graph/node AUROC vs. metadata
corruption probability `m_p`, right panel is the same two metrics vs. edge-drop rate `e_r`. Both
use the same plotting pattern as the previous two cells but without a table (`display()` is not
called here) and without any reference line, since there's no analogous "training-time value" for
these two knobs (corruption probability and edge-drop rate are both 0 during training by
definition — there's no non-trivial default to mark).

**Current values — metadata corruption:**

| p | graph_auroc | node_auroc |
|---|---|---|
| 0.00 | 1.0000 | 0.9737 |
| 0.05 | 1.0000 | 0.9736 |
| 0.10 | 1.0000 | 0.9738 |
| 0.20 | 0.9998 | 0.9738 |
| 0.30 | 0.9965 | 0.9740 |

**Current values — edge loss:**

| drop rate | graph_auroc | node_auroc |
|---|---|---|
| 0.00 | 1.0000 | 0.9737 |
| 0.05 | 1.0000 | 0.9734 |
| 0.10 | 1.0000 | 0.9737 |
| 0.20 | 1.0000 | 0.9736 |
| 0.30 | 1.0000 | 0.9749 |

**Reading this:** both perturbations are essentially inert across their whole tested range —
neither corrupting up to 30% of metadata values nor dropping up to 30% of edges moves either
AUROC by more than ~0.004. This makes sense given the mechanism identified above: detection relies
on *embedding-content similarity between planted records forming a dense sub-clique*, and neither
metadata corruption (which never touches the embedding columns) nor moderate random edge loss
(which thins but doesn't destroy a dense sub-clique — the surviving edges are still enough to mark
it as denser than its surroundings) attacks that mechanism directly. This is the flattest, least
newsworthy pair of sweeps in the notebook, but the flatness itself is a genuine, useful robustness
result: two entire categories of realistic operational data-quality problems (missing telemetry,
incomplete indexing) simply don't move the needle.

### Cell 11 — Summary dashboard

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
```

Recombines all four sweeps' key series (already computed as `sigmas/g_auroc/n_auroc`,
`thetas/g_auroc_t/n_auroc_t`, `m_p`/`meta_data`, `e_r`/`edge_data` from the earlier cells — this
cell has a hidden dependency on cells 5, 7, and 9 having already run, since it reuses their
variables rather than recomputing anything) into one 2×2 panel figure, labeled (A)–(D), and saves
it as `experiments/eval_images/ood_summary_dashboard.png` — the single combined figure referenced
by `docs/near_perfect_metrics_defense.md` §7 as the thesis's main OOD figure. This is a
presentation cell only; it introduces no new computation.

---

## 5. Cross-reference: this notebook's role in the project's self-audit

`docs/near_perfect_metrics_defense.md` §3 row 8 cites exactly this notebook's data (row 8: "AUROC
holds at 1.0 through σ=0.10–0.15, degrades from σ=0.20, reaches chance level (0.51) only at
σ=0.30") as one of eight independent tests defending the headline graph AUROC ≈ 1.0 against the
suspicion that it's a methodology artifact. The current `ood_results.json` (σ=0.30 → 0.5635, not
quite 0.51, but in the same "near chance" range — small differences are expected between
re-generation runs since the perturbations are stochastic and unseeded) is the *corrected* run,
generated after the bug described next was fixed.

---

## 6. A bug this notebook's own design exists to guard against — already fixed

`docs/near_perfect_metrics_defense.md` §4.3 documents a serious bug that was found and fixed in
exactly the code this notebook's numbers depend on: `GraphNoiseTransform.add_embedding_noise`
originally did

```python
x = data.x.to(device) if device is not None else data.x.clone()
if x.device != data.x.device:
    x = x.clone()
x[:, :embedding_dim] += noise   # in-place
```

Since `preload_dataset_to_gpu` (§2 above) preloads the *entire* test set onto the GPU **once** and
reuses those same tensors across all 23 sweep points, `.to(device)` on a tensor already on that
device returns the **same object**, not a copy — so the device-equality clone check never fired,
and the in-place `+=` permanently and cumulatively corrupted the shared preloaded tensors across
the whole sweep sequence (embedding noise → threshold → metadata → edge-drop, in that order). Every
sweep run *after* the embedding-noise sweep — including the threshold, metadata, and edge-drop
sweeps, which are supposed to be completely independent of embedding noise — was silently operating
on progressively corrupted data. The current source (§3 above) always calls `.clone()`
unconditionally rather than relying on a device-equality check, and the `ood_results.json` values
tabulated in this document are from the corrected pipeline. If you ever see numbers from this
notebook that don't match the tables above — in particular, threshold/metadata/edge-drop sweeps
showing degraded performance rather than the flat curves in §4's cells 7 and 9 — that's the
signature of this bug reappearing (or a similar shared-mutable-state issue elsewhere), not a real
finding, and the fix is to check every perturbation function still clones defensively.

---

## 7. A cosmetic bug in the current notebook source: broken θ (theta) symbol in labels

Inspecting the raw notebook JSON reveals an encoding mistake in the **threshold-sweep markdown and
plot-label cells** (cells covering §4's "Cell 7" and "Cell 11" above). The `\sigma` symbol used
throughout the noise-sweep cells is correctly written in the notebook's JSON source as `\\sigma`
(a properly-escaped backslash, which JSON decodes back to the literal two characters `\sigma`,
rendering correctly as σ once matplotlib/LaTeX sees it). The `\theta` symbol, however, is written
as a **single** backslash — `\theta` — directly in the JSON string. `\t` is itself a valid JSON
escape sequence (tab character), so a standard-compliant JSON parser decodes `\theta` not as the
four characters `\theta` but as **a literal tab character followed by the letters `heta`**. This
means every cell title, axis label, and legend entry that was meant to show "θ" instead currently
reads as a tab character followed by "heta" (visible above as `$	heta$` when the raw source is
displayed) — e.g. the section header intended as "Dynamic Cosine Threshold Sensitivity (θ Sweep)"
and the legend entry intended as "Training Threshold (θ=0.6)".

This does not affect any computed number in the notebook — it is a rendering/labeling bug only,
confined to the cells that reference `theta`/`heta`. The fix is to re-save those cells with `\theta`
written as `\\theta` in the notebook's JSON source (matching how `\sigma` is already written
correctly), so the LaTeX renders as an actual θ symbol instead of a tab-mangled string.

---

## 8. Key takeaways to remember when citing this notebook's numbers

| Claim | Evidence in this notebook | Caveat |
|---|---|---|
| Robust to embedding noise up to realistic levels | AUROC flat through σ≤0.15, degrading only from σ=0.20 onward, chance-level only at σ=0.30 | `graph_f1` (fixed 0.5 threshold) degrades a full noise step earlier than AUROC — a calibration effect, not a ranking failure |
| Robust to edge-threshold mismatch across most of the tested range | Graph AUROC flat at 1.0 for θ=0.4–0.7 | At θ=0.8 (very sparse graph), graph AUROC drops to 0.90 **while node AUROC rises to 0.998** — a real graph-vs-node trade-off, not simple monotonic degradation |
| Robust to metadata corruption and edge loss | Both sweeps stay within ~0.004 of baseline across their full 0–30% range | Consistent with the mechanism (embedding-content similarity), since neither perturbation touches embedding content directly |
| This notebook only visualizes — it doesn't compute | All numbers come from `experiments/ood_results.json`, produced by `src/experiments/ood_evaluator.py` | Re-running the notebook without first re-running the evaluator script just re-plots the same (possibly stale) JSON |
| A previously-corrupting bug in the perturbation code is fixed | `add_embedding_noise` now always clones defensively (§6) | If numbers ever look implausibly degraded across *all* sweeps, suspect a shared-mutable-tensor bug re-appearing, not a real regression |
| Cosmetic-only bug currently present in the source | θ-related labels render as a tab + "heta" due to a missing double-backslash in the notebook's JSON (§7) | Does not affect any computed metric — labels only |
