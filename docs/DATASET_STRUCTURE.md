# Dataset Structure & Composition

This document explains what the dataset represents, how it's built, and what the model actually sees.

---

## The Big Picture

You are simulating an **LLM agent's long-term memory** — like a chatbot that remembers past conversations. An attacker can **poison** this memory by injecting fake records that look legitimate but carry malicious instructions. Your TGN detector tries to catch this.

```
┌─────────────────────────────────────────────────────────┐
│                   ONE SEQUENCE                          │
│                                                         │
│  An LLM agent having 200 conversations over time.       │
│  Each conversation = 1 memory record = 1 node.          │
│                                                         │
│  Time ──────────────────────────────────────────►        │
│  t=0    t=10   t=20   ...   t=100  ...  t=190           │
│   │      │      │             │           │             │
│   ▼      ▼      ▼             ▼           ▼             │
│  snap   snap   snap         snap        snap            │
│  #0     #1     #2           #10         #19             │
│                                                         │
│  Each snapshot = graph of ALL memory records so far      │
│  → Nodes grow over time (10 at t=10, 100 at t=100)      │
│  → Edges = semantic similarity between records           │
└─────────────────────────────────────────────────────────┘
```

---

## Level 1: The Sequence

A **sequence** = one simulated agent lifetime (200 interactions).

- **Clean sequence**: 200 normal conversations. No attack.
- **Poisoned sequence**: 200 normal conversations + 10-20 injected malicious records.

With `num_sequences: 300` and `attack_ratio: 0.4`:
- 180 clean sequences
- 120 poisoned sequences (40 each of 3 attack types)

```
data/raw/synthetic/
├── clean_0000/                    # Clean sequence #0
├── clean_0001/                    # Clean sequence #1
├── ...
├── clean_0179/                    # Clean sequence #179
├── poisoned_minja_query_only_0000/    # Attack type 1
├── poisoned_direct_write_0001/        # Attack type 2
├── poisoned_incremental_slow_0002/    # Attack type 3
├── ...
├── metadata.json                  # Labels for all sequences
└── splits.json                    # Train/val/test assignment
```

---

## Level 2: The Snapshot

Every 10 interactions (`snapshot_interval: 10`), we take a **snapshot** of the entire memory graph.

A sequence of 200 interactions produces **20 snapshots** (at t=9, t=19, ..., t=199).

```
clean_0000/
├── snapshot_0000.npz    # Graph at t=9    (has ~10 nodes)
├── snapshot_0001.npz    # Graph at t=19   (has ~20 nodes)
├── snapshot_0002.npz    # Graph at t=29   (has ~30 nodes)
├── ...
└── snapshot_0019.npz    # Graph at t=199  (has ~200 nodes)
```

Each snapshot is a **complete graph** of all memory records that exist at that point in time. Graphs grow because records accumulate — the agent doesn't forget.

---

## Level 3: Inside a Snapshot (.npz file)

Each `.npz` file contains these arrays:

### node_features: (N, 387)

Each row = one memory record. The 387 features are:

| Indices | Dimension | What it is |
|---------|-----------|------------|
| 0–383 | 384 | Sentence embedding of the conversation |
| 384 | 1 | **Recency**: how long ago this record was created (0 = just now, 1 = oldest) |
| 385 | 1 | **Retrieval frequency**: how often this record was looked up (normalized) |
| 386 | 1 | **Retrieval recency**: how long since this record was last retrieved |

The 384-dim embedding encodes **what** the conversation was about.
The 3 meta features encode **how** the record behaves over time.

#### Stage A (Synthetic)
- Embeddings are random vectors clustered around 10 topic centers
- Fast, no LLM needed, good for architecture validation

#### Stage B (LLM-powered)
- Embeddings come from `all-MiniLM-L6-v2` encoding real LLM-generated text
- Realistic semantic relationships between records

### edge_index: (2, E)

Sparse edge list connecting semantically similar records.

Two records get an edge if their **cosine similarity > 0.6** (configurable via `semantic_edge_threshold`).

```
edge_index = [[0, 1, 0, 3, 2, 5, ...],    # source nodes
              [1, 0, 3, 0, 5, 2, ...]]    # target nodes
```

Edges are undirected (stored as two directed edges: A→B and B→A).

### edge_weights: (E,)

The cosine similarity value for each edge (between 0.6 and 1.0).

### node_labels: (N,)

Per-node ground truth: `0` = benign record, `1` = poisoned record.

In a clean sequence: all zeros.
In a poisoned sequence: mostly zeros, with a few 1s where injected records sit.

### graph_label: (1,)

Per-snapshot ground truth: `0` = no poisoning has occurred yet, `1` = at least one poisoned record exists in the memory.

---

## Level 4: The Attack Types

### minja_query_only (MINJA-style)
- **15 records** injected over **30 interactions**
- High coherence (0.85) — poisoned records are very similar to legitimate ones
- Mimics the MINJA paper's query-only attack vector

### direct_write
- **10 records** injected **all at once** (spread = 1)
- Very high coherence (0.90) — hard to distinguish
- Models a privileged attacker with direct memory access

### incremental_slow
- **20 records** spread over **100 interactions**
- Lower coherence (0.75) — records are more scattered semantically
- The hardest to detect because the signal is so diffuse

```
Timeline of a poisoned sequence (minja_query_only):
────────────────────────────────────────────────────────►
t=0                    t=70        t=100              t=199
│                       │    ▼▼▼▼▼▼▼│                   │
│   normal records      │ injection │   normal records  │
│   (benign)            │  window   │   (+ poison sits  │
│                       │           │    in memory)      │
```

---

## Level 5: What the Model Sees

The model doesn't see individual snapshots — it sees a **temporal window** of T=10 consecutive snapshots.

```
One training sample = a window of 10 snapshots from one sequence:

  snap_5  snap_6  snap_7  snap_8  snap_9  snap_10 snap_11 snap_12 snap_13 snap_14
  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐
  │ G  │  │ G  │  │ G  │  │ G  │  │ G  │  │ G  │  │ G  │  │ G  │  │ G  │  │ G  │
  │ 60 │→ │ 70 │→ │ 80 │→ │ 90 │→ │100 │→ │110 │→ │120 │→ │130 │→ │140 │→ │150 │
  │nods│  │nods│  │nods│  │nods│  │nods│  │nods│  │nods│  │nods│  │nods│  │nods│
  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘

  Label = from the LAST snapshot (snap_14):
    graph_label: 0 or 1 (is this trajectory poisoned?)
    node_labels: [0,0,0,1,0,0,1,0,...] (which specific nodes are poisoned?)
```

From one sequence of 20 snapshots, you can extract **11 overlapping windows**:
- Window 0: snapshots [0–9]
- Window 1: snapshots [1–10]
- ...
- Window 10: snapshots [10–19]

### Why this matters for dataset size

With 300 sequences × ~11 windows each = **~3,300 training samples**.
With 1000 sequences × ~21 windows each (300 interactions) = **~21,000 training samples**.

This is why 1000 sequences already gives you plenty of data — you're not training on 1000 samples, you're training on **~21,000 overlapping temporal windows**.

---

## Summary: Data Volume Math

| Config | Sequences | Interactions | Snapshots/seq | Windows/seq | Total Training Samples |
|--------|-----------|-------------|---------------|-------------|----------------------|
| default.yaml | 300 | 200 | 20 | 11 | ~2,640 (80% train) |
| large.yaml | 1,000 | 300 | 30 | 21 | ~16,800 (80% train) |
| Hypothetical 60K | 60,000 | 200 | 20 | 11 | ~528,000 |

Your model has **650K parameters**. The rule of thumb is you need roughly 5-50x your parameter count in training samples. That means:

- **3,250 – 32,500 samples** is the sweet spot
- `large.yaml` with 1,000 sequences → ~16,800 samples → **right in the middle**
- 60,000 sequences → 528K samples → **16x more than you'd ever need**

---

## File Reference

| File | What it does |
|------|-------------|
| `src/data_generation/synthetic_generator.py` | Stage A: creates synthetic embeddings + graph snapshots |
| `src/data_generation/llm_generator.py` | Stage B: uses real LLM to generate text, encodes with MiniLM |
| `src/data_generation/dataset.py` | Loads `.npz` snapshots into PyG Data objects for training |
| `configs/default.yaml` | 300 sequences, 200 interactions — for local testing |
| `configs/large.yaml` | 1000 sequences, 300 interactions — for vast.ai training |
