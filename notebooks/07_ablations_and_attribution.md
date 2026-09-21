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

# Notebook 07 — Ablations and Attribution: A Full Walkthrough

This document explains **every cell, function, and variable** in
[`07_ablations_and_attribution.ipynb`](07_ablations_and_attribution.ipynb): what it does, why it
exists, and what the numbers it produces actually mean. It is meant to be read alongside the
notebook, or in place of it if you just need to understand the reasoning without re-running code.

---

## 1. Purpose of the notebook

The project's main result (from notebook 05 / `experiments/run_current_arch`) is a Temporal
Graph Network (TGN) detector that reaches **graph-level AUROC ≈ 1.0** at spotting memory-poisoning
attacks against an LLM agent's memory store. A number that clean is, by default, more likely to be
a methodology bug than a real result — leakage, a trivial shortcut, or an evaluation artifact are
all statistically more common than a genuinely unbeatable task. Notebook 07 is the **adversarial
self-audit** of that claim: it exists to answer *"what's wrong with your evaluation?"* before a
reader has to ask.

It does this by bundling together five otherwise-homeless pieces of that audit into one notebook:

1. **Attention-pooling ablation** — does the specific pooling mechanism matter, or would plain
   averaging do just as well?
2. **Per-attack-type breakdown** — is one of the three attack types trivially easy and dragging
   the pooled average up?
3. **Attribution probes** — is the near-perfect score explained by a simple shortcut (graph size,
   or averaged embedding content) rather than the relational/temporal mechanism the model was
   designed around?
4. **Computational profile** — what does this cost to run (parameters, latency, memory)?
5. **Holdout attack-type generalization** — did the model memorize the three fixed attack
   templates, or does it generalize to a pattern it never trained on?

A key design choice stated up front in the notebook: **everything here evaluates already-trained
checkpoints.** The two expensive trainings it depends on (the main detector and the mean-pool
ablation variant, each several hours on a 4GB GPU) are run as standalone background scripts
(`src/models/train.py`, `src/experiments/train_ablation_meanpool.py`), not notebook cells. Only
the cheap linear probes (logistic regression) are actually fit inside this notebook.

This notebook is the notebook-form counterpart to the investigation documented narratively in
[`docs/near_perfect_metrics_defense.md`](../docs/near_perfect_metrics_defense.md) — read that file
for the "why does this mechanism work" story; read this file for what the code that produces the
numbers actually does.

---

## 2. Setup cell (cell 1)

```python
import sys, json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
```

- **`sys.path.insert(0, str(Path("..").resolve()))`** — makes the repo root importable so
  `from src... import ...` works when the notebook's working directory is `notebooks/`.
- **`device`** — `cuda` if a GPU is visible, else `cpu`. Every model and tensor in the notebook is
  moved to this device.
- **`CONFIG`** — the parsed contents of `configs/default.yaml`, the single source of truth for
  architecture sizes and training hyperparameters used everywhere in the project. Relevant values
  used later in this notebook: `temporal_window: 10` (how many past graph snapshots form one
  detection window), `embedding_dim: 384` (sentence-embedding size), `node_feature_dim: 384`,
  `node_memory_dim: 128`, `gat_hidden_dim: 128`, `gat_heads: 4`, `gru_hidden_dim: 128`,
  `mlp_hidden_dim: 64`, `dropout: 0.3`.
- **`DATA_DIR`** — `../data/raw/llm_generated`, the LLM-generated synthetic dataset of memory
  trajectories (clean and poisoned) that every experiment in this notebook scores against.
- **`MAIN_RUN_DIR`** — `../experiments/run_current_arch`, the directory holding the main trained
  detector's checkpoint (`phase2_best.pt`) and training history.

---

## 3. Section 1 — Attention-Pooling Ablation (cells 2–4)

**Question:** the model pools per-node embeddings into one graph-level vector using a learned
*attention* mechanism (`AttentionPool`) instead of a plain average. Does that specific choice
matter, or is it just as good to average?

### Why this matters (markdown cell 2)

A poisoned trajectory plants roughly 10–20 anomalous records among several hundred benign ones.
Plain mean pooling divides evenly by node count, so a small anomalous cluster gets washed out
almost by construction — averaging 15 unusual vectors into a pool of 300+ nodes barely moves the
mean. `AttentionPool` instead learns per-node weights so the pooled vector can be dominated by the
few nodes that matter. This ablation isolates *only* that one design choice: `TGNDetectorMeanPool`
is stated to be identical to `TGNDetector` in every other respect (node memory GRU, GAT encoder,
temporal GRU, both heads) — a claim the code verifies by checking the exact parameter-count
difference, not just asserting it.

This is explicitly contrasted with a *different*, coarser ablation living in notebook 05 (the
"static-graph ablation"), which removes attention pooling **and** node memory **and** the temporal
aggregator all at once — so it can't isolate which of those three ingredients does the work. This
notebook's ablation changes exactly one thing.

### `TGNDetectorMeanPool` (cell 3)

```python
class TGNDetectorMeanPool(TGNDetector):
    def __init__(self, config: dict):
        super().__init__(config)
        del self.attn_pool
    def forward(self, graph_sequence): ...
```

- Subclasses the real `TGNDetector` (defined in
  [`src/models/tgn_detector.py`](../src/models/tgn_detector.py)) and immediately deletes the
  `attn_pool` submodule it doesn't use, so the parameter-count comparison later is honest (dead
  weights aren't silently still allocated).
- `forward` is a near-verbatim copy of `TGNDetector.forward` (see §9 below for what each line does
  in detail) with exactly one line swapped: `graph_emb = self.attn_pool(node_embeddings,
  batch_idx, num_graphs=num_graphs)` becomes `graph_emb = global_mean_pool(node_embeddings,
  batch_idx)` — PyTorch Geometric's built-in "average all nodes belonging to the same graph in the
  batch" pooling op.
- Everything downstream (temporal GRU, graph head, node head) is untouched.

### `evaluate_meanpool_ablation` (cell 4)

```python
def evaluate_meanpool_ablation(config, checkpoint_path, data_dir):
```

Loads the **already-trained** mean-pool checkpoint
(`experiments/run_ablation_meanpool/phase2_best.pt`) and scores it on the held-out test split.
Step by step:

- `T`, `embedding_dim` — pulled from config (`temporal_window`, `simulation.embedding_dim`).
- `model = TGNDetectorMeanPool(config).to(device)` then `load_state_dict(...)` — instantiate the
  ablation architecture and load the pretrained weights (`weights_only=True` is the modern, safer
  `torch.load` mode that refuses to unpickle arbitrary code).
- **Parameter-count sanity check** — computes `full_params` (a *fresh, untrained* `TGNDetector`'s
  parameter count) vs `meanpool_params` (the ablation's), and prints the difference. This is a
  correctness assertion disguised as a print statement: if the two architectures differed by more
  than `AttentionPool`'s own parameter count (8,321, computed as `128*64 + 64 + 64*1 + 1` for the
  two linear layers of `AttentionPool`), the "isolates pooling cleanly" claim would be false. The
  printed output confirms the difference is exactly 8,321.
- `test_dataset` / `test_loader` — the test split of `MemoryTrajectoryDataset` (see §8), batched
  with `collate_trajectories`, `batch_size=4` (same as training, chosen for the 4GB-VRAM GPU this
  project was developed on).
- The inference loop under `torch.no_grad()` — for each batch, run the model to get
  `(graph_scores, node_scores)`, then:
  - `all_graph_scores` / `all_graph_labels` accumulate one score and one label per trajectory
    window.
  - For node-level: only the **last graph in the sequence** (`graphs[-1]`) carries meaningful
    `node_labels` (a node's poisoned/clean status is evaluated at the point where the model is
    asked to make a call, i.e. the present moment of the window) — `min_len` guards against any
    off-by-one between the number of scored nodes and the number of labeled nodes in that batch.
- **`results` dict** — AUROC / AP / F1 for graph-level, AUROC / AP for node-level, but *only if*
  both classes (0 and 1) are present in that batch of labels (`len(np.unique(...)) > 1`) — AUROC
  is undefined with a single class, so this guards against a `sklearn` exception on a degenerate
  split (this same guard reappears in `Trainer._evaluate`, see §10, and its absence is exactly why
  the holdout run's `graph_auroc` shows as `0.0` in §7 below).

**Result:** graph AUROC 0.9944 / AP 0.9921 / F1 0.9576, node AUROC 0.9718 / AP 0.6161 — clearly
*worse* than the full attention-pooling model's graph AUROC of ~1.0, confirming attention pooling
is doing real work, not just adding harmless extra parameters.

---

## 4. Section 2 — Per-Attack-Type Breakdown (cells 5–6)

**Question:** the pooled test AUROC is ~1.0 — is that because *all three* attack types are equally
easy, or because one is trivial and inflates the average while another is actually weak?

### `evaluate_attack_type_breakdown` (cell 6)

```python
def evaluate_attack_type_breakdown(config, checkpoint_path, data_dir):
```

- Loads `metadata.json` into a dict keyed by `sequence_id` — each entry carries `label`
  (`"clean"`/`"poisoned"`) and, for poisoned sequences, `attack_type` (one of `minja_query_only`,
  `direct_write`, `incremental_slow` — see `configs/default.yaml`'s `attacks.types`).
- Loads the **main** detector checkpoint (`run_current_arch/phase2_best.pt`), not the ablation.
- Iterates the test dataset **one window at a time** (`test_ds[i]`, then
  `collate_trajectories([item])` to wrap a single item into the batch format the model expects) —
  slower than batching, but simple, and this section only runs once (330 windows).
- For each window, records a triple `(attack_type, item["graph_label"], score)`, where
  `attack_type` is `"clean"` for non-poisoned windows and the actual attack name otherwise (looked
  up via `metadata[seq_id]`, but only trusted `if m["label"] == "poisoned"` — a clean sequence has
  no meaningful `attack_type` field).
- `by_type` — a `defaultdict(list)` grouping `(label, score)` pairs by attack type string
  (including `"clean"`).
- `clean_records` — every clean window's `(label, score)` pair, pulled out once since it's reused
  as the shared negative pool for every attack type's comparison.
- **Per-attack-type loop:** for each real attack type, builds `pool = clean_records +
  by_type[attack_type]` — i.e. *this attack type's poisoned windows vs. every clean window in the
  test set* — and computes AUROC/AP against that pool. This is the deliberate design choice
  explained in the markdown cell 5: **without** pooling against the *shared* set of clean windows,
  each attack type would only be compared against its own (much smaller) set of negatives, which
  would make small-sample per-type numbers noisier and non-comparable across types.
- `results[attack_type]` records `auroc`, `ap`, `n` (pool size), `n_positive` (poisoned count in
  the pool), and `n_windows_this_type` (raw count of windows of that type, before pooling).
- **Pooled row** — AUROC/AP computed once over *all* records together (all attack types + clean),
  as the standard, single-number summary metric to compare against.

The markdown cell 5 also flags an important caveat: **the split is not stratified** by attack
type — `direct_write` contributes only 2 poisoned test windows vs. 3 each for the other two types.
This is why the `n` column doesn't divide evenly, and it's worth knowing before treating any single
type's AUROC as a precise estimate (2 positive examples gives a very coarse AUROC).

**Result:** all three attack types individually reach AUROC = AP = 1.0000, and so does the pooled
number — the near-perfect score is *not* an artifact of one easy attack type dragging up an
average; the model separates every attack type cleanly.

---

## 5. Section 3 — Attribution Probes (cells 7–8)

**Question:** is the near-perfect detector actually using the relational/temporal mechanism it was
architected around (small mutually-similar clusters forming dense subgraphs — see
`docs/near_perfect_metrics_defense.md` §5), or could a much dumber signal — raw graph size, or just
the average embedding content — already achieve most of that performance?

This section fits two **plain logistic regressions**, each restricted to a deliberately
impoverished slice of the available information, to measure how much performance that restricted
information alone supports. The gap between each probe and the full detector attributes the
remaining performance to whatever that probe was denied.

### `extract_structural_features` (cell 8)

```python
def extract_structural_features(dataset):
```

For every window in a dataset, uses only the **last snapshot's graph shape** — no node content,
no embeddings, no edges' semantic meaning, just:
- `n = g.x.size(0)` — number of nodes (memory records) in that snapshot.
- `e = g.edge_index.size(1)` — number of directed edges.
- `density = e / max(n * (n - 1), 1)` — edge density normalized by the maximum possible edge count
  for `n` nodes (directed, no self-loops); the `max(..., 1)` guards division by zero for a
  single-node graph.

Returns `X` (an `(N, 3)` float array of `[n, e, density]` per window) and `y` (the window's
graph-level label). This tests: **"are poisoned graphs just structurally different in size/density
alone, independent of what's actually in them?"**

### `extract_content_features` (cell 8)

```python
_EMBEDDING_DIM = 384
def extract_content_features(dataset):
```

For every window, takes the last snapshot's node features, slices off just the first 384 columns
(`last_graph.x[:, :_EMBEDDING_DIM]` — the raw sentence-embedding portion of each node's feature
vector, excluding the 3 extra meta-features `node_feature_dim + 3` that `TGNDetector.__init__`
adds internally: recency, retrieval frequency, retrieval recency), then **mean-pools across
nodes** (`.mean(axis=0)`) to get one 384-dim vector per window. This tests: **"is a poisoned
window's content just distinguishable on average, with no relational structure or temporal
ordering at all?"** — it's deliberately the same "mean-pool everything" shortcut that
`AttentionPool` exists to avoid, applied here as a diagnostic baseline rather than as part of the
detector.

### The fitting/scoring block (cell 8)

For each probe:
```python
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train, y_train)
scores = clf.predict_proba(X_test)[:, 1]
```
- `class_weight="balanced"` compensates for the fact that poisoned windows are a minority class
  (attack ratio 0.4 at the *sequence* level per `configs/default.yaml`, but far fewer at the
  *window* level once negative snapshots from clean sequences are counted) — without it, logistic
  regression would be biased toward always predicting "clean."
- `max_iter=2000` — logistic regression's default iteration cap (100) is often too low to converge
  on real feature scales; raised generously so a convergence warning doesn't need to be
  interpreted.
- Metrics computed exactly as elsewhere: AUROC, AP, and (content probe only) F1 at the default 0.5
  threshold.

**Result:**
- Structural probe: AUROC 0.5946, AP 0.3888 — barely above chance (0.5). Graph size/density alone
  explains almost nothing.
- Content probe: AUROC 0.7729, AP 0.4388, F1 0.5577 — meaningfully above chance, but far below the
  full detector's ~1.0.
- **Gap from content probe to full detector: 0.2271** (`1.0 - 0.7729`) AUROC points.

The markdown commentary explains how to decompose that 0.2271 gap further: the *content* probe has
neither relational structure nor temporal ordering. The **static-graph ablation** (in notebook 05,
trained with node loss added) has relational structure (it's still a GNN over one snapshot) but
*no* temporal ordering (no GRU over time). So:
- `static_graph_auroc - content_auroc` ≈ structure's independent contribution.
- `full_detector_auroc - static_graph_auroc` ≈ temporal ordering's independent contribution.

This notebook doesn't recompute the static-graph ablation's number itself (it lives in notebook
05) — it just points at it for that comparison, which is why this section's own output ends with a
`print` reminding the reader to go compare.

---

## 6. Section 4 — Computational Profile (cells 9–10)

**Question:** what does this model actually cost to run — how big is it, how many of those
parameters live in which submodule, how fast is single-window inference, and how much GPU memory
does it need?

### `count_params` (cell 10)

```python
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    by_submodule = {name: sum(p.numel() for p in module.parameters())
                    for name, module in model.named_children()}
```

`model.named_children()` walks only the model's **direct** submodules (`node_memory`,
`graph_encoder`, `attn_pool`, `temporal_aggregator`, `graph_head`, `node_head` — the six
attributes assigned in `TGNDetector.__init__`), so this gives a clean per-component parameter
breakdown without double-counting nested layers.

### `time_inference` (cell 10)

```python
def time_inference(model, batches, device, n_repeats=30, warmup=5):
```

Standard latency-benchmarking pattern:
- **Warmup** (5 iterations, results discarded) — the first few CUDA calls pay for kernel
  compilation/caching and memory-allocator warmup; including them would overstate typical latency.
- **Timed loop** (30 repeats) — for each: `model.reset_memory()` (clears the TGN's per-node memory
  bank so timing reflects a cold-start inference, not one that benefits from another window's
  cached memory), `torch.cuda.synchronize()` *before and after* the timed region (GPU calls are
  asynchronous — without synchronizing, `time.perf_counter()` would measure "time to enqueue the
  kernel," not "time for it to actually finish"), then the model's forward pass.
- Cycles through the `batches` list (`batches[i % len(batches)]`) so 30 timed repeats can reuse a
  smaller pool of pre-built batches.
- Returns mean, median (p50), and 95th-percentile (p95) latency in milliseconds — p95 matters more
  than mean for a detector that might gate real-time agent actions, since a rare slow outlier is
  what would cause a timeout.

### The profiling block (cell 10)

- `checkpoint_size_mb` — the `.pt` file's size on disk, a proxy for deployment footprint.
- `prof_ds` / `items` — pulls 20 items from the test set once (`n_pool = min(20, len(prof_ds))`)
  to build reusable batches without reloading data during each timed iteration.
- `make_batches(batch_size, dev)` — chunks the 20 pre-loaded items into groups of `batch_size`,
  runs each chunk through `collate_trajectories`, and moves every graph in the resulting
  time-indexed list to `dev`. Used here only with `batch_size=1` (single-window latency — the
  realistic serving scenario of scoring one agent trajectory at a time).
- `torch.cuda.reset_peak_memory_stats(device)` before timing, then
  `torch.cuda.max_memory_allocated(device)` after — measures the single largest GPU memory
  footprint reached during any of the 30 timed forward passes, reported in MB.

**Result:** 667,331 total parameters (GAT graph encoder is the largest single piece at 332,416;
node memory GRU next at 198,528; temporal aggregator 99,072; node head 20,673; attention pool and
graph head 8,321 each). Batch-1 latency: ~70ms mean/median, ~80ms p95. Peak GPU memory: 56.8MB —
this is a genuinely small, cheap-to-serve model.

---

## 7. Section 5 — Holdout Attack-Type Generalization (cells 11–12)

**Question:** the detector sees only three fixed attack templates during training
(`minja_query_only`, `direct_write`, `incremental_slow`). Does it generalize to an attack *pattern*
it has never seen, or has it just memorized those three templates by some template-specific
signature?

### The `Trainer` holdout mechanism (from [`src/models/train.py`](../src/models/train.py))

The notebook doesn't retrain — it loads a checkpoint that *was* trained with one type excluded, via
`Trainer(config, data_dir, device, holdout_attack_type="incremental_slow")`. Inside
`Trainer.__init__` → `_build_splits`:

```python
def is_holdout(seq_id):
    return metadata[seq_id].get("attack_type") == holdout_attack_type

train_ids = [s for s in splits["train"] if not is_holdout(s)]
val_ids   = [s for s in splits["val"] if not is_holdout(s)]
holdout_ids = [s for s in (splits["train"] + splits["val"] + splits["test"]) if is_holdout(s)]
```

- Every sequence whose `attack_type` is `incremental_slow` is stripped out of `train` and `val`
  entirely (not just moved to test) — so the model's weights never receive a gradient from any
  `incremental_slow` example.
- `holdout_ids` instead pools *every* `incremental_slow` sequence from **all three** original
  splits (train + val + test) into one dedicated generalization-eval set — maximizing the sample
  size available for that one held-out check, since none of it can leak into training anyway.
- The printed line `"removed 37 sequences from train/val, 40 sequences reserved for generalization
  eval"` is exactly this bookkeeping: 37 sequences were pulled out of what would have been
  train/val, and the union of those plus any that were already in test totals 40.

### `holdout_trainer.evaluate_holdout()` (from `train.py`)

```python
def evaluate_holdout(self):
    if self.holdout_loader is None: return None
    loss, graph_auroc, node_auroc = self._evaluate(self.holdout_loader, alpha, beta)
    return {"loss": loss, "graph_auroc": graph_auroc, "node_auroc": node_auroc,
            "n_sequences": len(self.holdout_ids)}
```

Runs the shared `_evaluate` loop (same one used for validation during training: forward pass,
`BCELoss` on graph scores, `FocalLoss` on node scores, `alpha`/`beta` weighting from
`config["training"]["loss_alpha"/"loss_beta"]`) over the holdout loader.

**Important subtlety — read this before trusting the `graph_auroc: 0.0` number at face value.**
The holdout set is built from sequences whose `attack_type == "incremental_slow"` — every single
one of those sequences is, by construction, **poisoned** (`label: "poisoned"` in the metadata).
There are no clean sequences in this holdout pool at all. `_evaluate`'s AUROC computation is
guarded exactly the same way as in §3 above:
```python
graph_auroc = (roc_auc_score(...) if len(np.unique(all_graph_labels)) > 1 else 0.0)
```
With only one class present (`all_graph_labels` are all `1`), AUROC is mathematically undefined,
and the code's fallback is to report `0.0` rather than crash. **The `graph_auroc: 0.0` in the
output does not mean the model failed to detect `incremental_slow` at the graph level** — it means
there was no clean/negative example in this holdout set to compute a ranking metric against. The
informative number here is **`node_auroc: 0.8543`**, which *is* meaningful because within any
single sequence's last snapshot there are both poisoned nodes (the planted records) and clean
nodes (the pre-existing legitimate memory), giving both classes needed for AUROC.

For the fuller picture — including a graph-level *detection rate* (not AUROC) on this same holdout
set — see `docs/near_perfect_metrics_defense.md` §3 row 7, which reports 99.5% (437/440 windows)
using a threshold-based hit rate computed by a separate standalone script
(`src/experiments/eval_holdout_checkpoint.py`) rather than AUROC. That number and this notebook's
`node_auroc: 0.8543` are two different, complementary metrics on the same experiment, not a
contradiction.

**Result:** node AUROC 0.8543 on an attack type the model never saw during training — real,
if imperfect, generalization rather than memorization of the three fixed templates.

---

## 8. Section 6 — Save Results (cell 14)

```python
combined_output = {
    "attention_pooling_ablation": meanpool_results,
    "attack_type_breakdown": attack_type_results,
    "attribution_probes": attribution_results,
    "computational_profile": profile_results,
    "holdout_generalization": holdout_result,
}
```

Just bundles every dict produced above into one JSON file,
`experiments/notebook_07_results.json`, using `default=str` in `json.dump` as a catch-all so any
stray non-JSON-serializable value (e.g. a numpy scalar that slipped through a `float(...)` cast)
gets stringified instead of raising.

---

## 9. Shared building blocks from `src/models/tgn_detector.py`

Since sections 1, 2, and 5 all load and run the actual `TGNDetector` (or its mean-pool subclass),
understanding its `forward` pass is central to reading this notebook. Each submodule:

- **`NodeMemoryModule`** — a `GRUCell` giving every memory-graph node a persistent hidden state
  across timesteps. `forward(node_features, prev_memory)`: if there's no previous memory for a
  node (its first appearance), initializes it to zeros; otherwise runs one GRU step. This is what
  lets the model notice *how a node's representation has evolved*, not just its current snapshot.
- **`GraphEncoder`** — a 2-layer Graph Attention Network (`GATConv`). Layer 1 uses 4 attention
  heads with `concat=True` (so its output width is `hidden_dim * heads`); layer 2 collapses back
  to `hidden_dim` with a single head (`concat=False`). `edge_attr` (the cosine-similarity edge
  weight between semantically-linked memory records) is fed into both GAT layers so attention can
  use edge strength, not just node content. Produces one embedding vector per node.
- **`AttentionPool`** — the MIL-style (Ilse et al.) attention pooling this section's ablation
  targets. A small 2-layer MLP (`Linear → Tanh → Linear`) scores each node, `scatter_softmax`
  normalizes those scores *within each graph in the batch* (so weights for different graphs don't
  interfere), and the weighted node embeddings are summed per graph (`index_add`) into one
  graph-level vector.
- **`TemporalAggregator`** — a single-layer `GRU` reading the sequence of `T` graph-level
  embeddings (one per snapshot in the window) and returning its final hidden state as the
  trajectory embedding — the model's summary of "what happened across this whole window."
- **`GraphAnomalyHead`** / **`NodeAnomalyHead`** — small MLPs mapping the trajectory embedding (and,
  for the node head, each node's own embedding concatenated with that trajectory embedding
  broadcast to every node via `trajectory_emb[batch_idx]`) to a sigmoid probability.
- **`TGNDetector.forward`** — for each timestep `t` in the window: look up each node's previous
  memory by its persistent `node_ids` (`_get_node_memories`), run one GRU step
  (`self.node_memory`), store the updated memory back (`_store_node_memories`), concatenate raw
  features with the fresh memory state as GAT input, run the GAT encoder, pool to a graph
  embedding, and stash it. After all `T` steps, stack the per-timestep graph embeddings, run the
  temporal GRU over them, and score both heads off the resulting trajectory embedding (the node
  head additionally uses `last_node_embeddings`/`last_batch_idx` — only the final timestep's raw
  per-node embeddings, since node-level labels only exist for the present moment).
- **`reset_memory` / `_ensure_memory_tensor` / `_get_node_memories` / `_store_node_memories`** —
  the vectorized (GPU-tensor-indexed, not Python-dict-based) memory bank. `_ensure_memory_tensor`
  lazily grows a `(capacity, memory_dim)` tensor (doubling capacity when a bigger node ID appears)
  so memory lookups/writes are simple tensor indexing (`self._memory_tensor[node_ids]`) rather than
  per-node Python-level dict operations — this is what the project's docstring calls out as "100x
  faster evaluation" versus an earlier, unvectorized implementation (see
  `docs/near_perfect_metrics_defense.md` §4.4 for the historical context: an earlier, divergent
  notebook still had the slow dict-based version).

---

## 10. Shared building blocks from `src/data_generation/dataset.py`

- **`MemoryTrajectoryDataset`** — one item = one temporal window (`T` consecutive graph snapshots)
  from one sequence. `_build_window_index` enumerates every valid `(sequence_id, start_snapshot)`
  pair per sequence (sliding window of width `T`) so `__len__`/`__getitem__` can treat the whole
  dataset as a flat list of windows rather than nesting sequence/window indices. `sequence_ids` can
  be given explicitly (bypassing `splits.json`) — this is exactly the mechanism the holdout trainer
  uses to build a custom train/val/holdout partition.
- **`_load_snapshot`** — reads one `.npz` snapshot file into a PyG `Data` object: `x` (node
  features), `edge_index`/`edge_attr` (graph structure + similarity weights), `y` (graph label),
  `node_labels` (per-node label), and `node_ids` — sequential integers assigned by node position,
  which is what lets `NodeMemoryModule` track "the same node" across snapshots within a sequence
  (records are append-only, so a node's index position is stable over time).
- **`_empty_graph`** — padding for windows near the start of a sequence that don't yet have `T`
  real snapshots behind them (a single dummy node, no edges, label 0).
- **`collate_trajectories`** — the custom collate function every `DataLoader` here uses instead of
  PyTorch's default, because each item is a *list of T graphs*, not one flat tensor. It transposes
  batch-of-sequences into sequence-of-batches: for each timestep `t`, gathers that timestep's graph
  from every item in the batch and merges them with PyG's `Batch.from_data_list` (which also
  produces the `batch_idx` vector `GraphEncoder`/`AttentionPool`/pooling ops rely on to know which
  nodes belong to which graph).

---

## 11. Key takeaways to remember when citing this notebook's numbers

| Claim | Evidence in this notebook | Caveat |
|---|---|---|
| Attention pooling matters, not just extra parameters | Mean-pool ablation scores lower (graph AUROC 0.9944 vs ~1.0) despite parameter-count-verified apples-to-apples comparison | — |
| Not one easy attack type inflating the average | All three types individually hit AUROC/AP 1.0000 | Test split is unstratified — `direct_write` has only 2 positive examples, so its per-type number is coarse |
| Not a graph-size/structure-only shortcut | Structural probe AUROC 0.5946 (near chance) | — |
| Not a content-averaging shortcut | Content probe AUROC 0.7729 | Still meaningfully above chance — content matters, just isn't sufficient alone |
| Not memorization of fixed templates | Holdout `incremental_slow` node AUROC 0.8543 | `graph_auroc: 0.0` here is a metric-undefined artifact (no clean sequences in the holdout pool), **not** a failure — see §7 |
| Cheap to run | 667K params, ~70ms batch-1 latency, 56.8MB peak GPU memory | Measured on whatever GPU the notebook executed on; absolute latency will vary by hardware |
