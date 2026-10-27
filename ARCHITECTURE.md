# Architecture

Two views: the detector itself, and the pipeline that builds data, trains models and produces
the reported numbers. Every dimension and parameter count below is read from the code
(`src/models/tgn_detector.py`, `src/data_generation/`) and the saved runs.

## 1. The detector (TGNDetector, 670,915 parameters)

One input is a **window**: 10 consecutive memory-graph snapshots of one agent. The same three
per-snapshot stages run on each snapshot in order; a GRU then reads the 10 snapshot summaries.

```mermaid
flowchart TB
    subgraph IN["Input: one window = 10 snapshots"]
        X["Node features, 391 per record<br/>384 sentence embedding (all-MiniLM-L6-v2)<br/>3 behavioural: recency, retrieval frequency, retrieval recency<br/>4 structural: degree x2 scalings, mean and max incident edge weight"]
        E["Edges: cosine similarity > 0.6<br/>edge weight = similarity (edge feature)"]
    end
    subgraph SNAP["Per snapshot t = 1..10 (shared weights)"]
        M["Node memory<br/>GRUCell 391 -> 128<br/>bank addressed per window, reset each window<br/>200,064 params"]
        C["Concatenate features + memory<br/>391 + 128 = 519"]
        G["Graph encoder<br/>GAT 519 -> 4 x 128 = 512, ELU<br/>GAT 512 -> 128, 1 head<br/>edge weights in attention, dropout 0.3<br/>334,464 params"]
        P["Attention pooling<br/>score = Linear 128->64, tanh, Linear 64->1<br/>softmax over the snapshot's nodes, weighted sum<br/>8,321 params"]
    end
    T["Temporal aggregator<br/>GRU 128 -> 128 over the 10 snapshot vectors<br/>99,072 params"]
    GH["Graph head<br/>Linear 128->64, ReLU, dropout, Linear 64->1, sigmoid<br/>8,321 params"]
    NH["Node head<br/>[final node embedding 128 + trajectory 128]<br/>MLP 256->64->64->1, sigmoid<br/>20,673 params"]
    OUT1(["P(window poisoned)"])
    OUT2(["P(record injected), final snapshot"])
    SEQ(["Sequence score = max over its 11 windows"])

    X --> M --> C
    X --> C
    C --> G
    E --> G
    G --> P --> T
    T --> GH --> OUT1 --> SEQ
    G -- "final snapshot node embeddings" --> NH
    T --> NH --> OUT2
```

Plain-text version:

```
 window of 10 snapshots ──────────────────────────────────────────────────────────────┐
                                                                                      │
 for each snapshot t:                                                                 │
   node features (391) ──► NODE MEMORY  GRUCell 391→128  (per-window memory bank)     │
          │                         │                                                 │
          └────────── concat ◄──────┘   = 519 per node                                │
                         │                                                            │
   edges (cos > 0.6, ──► GRAPH ENCODER  GAT 519→4×128 (ELU) → GAT 512→128            │
   weight = cos)                │                                                     │
                                ├─────────────► (final snapshot only) ──┐             │
                                ▼                                       │             │
                         ATTENTION POOLING  128 → 1 vector per snapshot │             │
                                │                                       │             │
 10 snapshot vectors ──► TEMPORAL AGGREGATOR  GRU 128→128 ──► h (128)  │             │
                                                         │              │             │
                                    GRAPH HEAD  MLP 128→64→1 ◄──┤       │             │
                                         │                      │       ▼             │
                              P(window poisoned)          NODE HEAD  [node 128 ‖ h 128]
                                         │                MLP 256→64→64→1             │
                     sequence score = max over windows    P(record injected) ◄────────┘
```

**Training objective.** Binary cross-entropy on the window score (weight 1.0) plus focal loss on
the record scores (weight 1.5, gamma 3.0, positive-class weight 0.85). Adam, learning rate
0.0001, weight decay 0.00001, cosine schedule over 12 epochs, batch of 4 windows, gradient clip
1.0, Gaussian noise on the embedding columns of half the batches (sigma up to 0.15). The
checkpoint kept is the epoch with the highest validation window AUROC.

**Ablations: what each removes** (sequence AUROC on the main benchmark, five seeds):

| Variant | Node memory | GAT | Pooling | Temporal GRU | Structural features | AUROC |
|---|---|---|---|---|---|---|
| Full detector (`tgn`) | yes | yes | attention | yes | yes | 0.6614 |
| `no_aggregator` | yes | yes (final snapshot) | attention | no | yes | 0.6515 |
| `meanpool` | yes | yes | mean | yes | yes | 0.6187 |
| `no_memory` | no | yes | attention | yes | yes | 0.5841 |
| `static_attn`, `static_wide` | no | yes (final snapshot) | attention | no | yes | 0.5408, 0.5410 |
| `nostruct` | yes | yes | attention | yes | no | 0.5396 |

## 2. The pipeline

```mermaid
flowchart LR
    KB[("AgentPoison StrategyQA<br/>knowledge base<br/>2,821 entries")]
    GEN["agentpoison_sequences.py<br/>2,000 agents x 200 interactions<br/>flood attack vs matched clean control<br/>20 snapshots each"]
    DS[("data/raw/ap_burst2k_nodup<br/>snapshot .npz files<br/>split 1200 / 300 / 500")]
    CACHE["cached_dataset.py<br/>one file per sequence<br/>verified bit-identical"]
    TRAIN["train_variant.py<br/>(Trainer in models/train.py)<br/>any variant x seed"]
    Q["seed_queue.py<br/>main_benchmark_plan.py<br/>resumable, STATUS.md"]
    RUNS[("experiments/seeds_nodup/<br/>checkpoint, run_config.json,<br/>validation and test scores")]
    SUM["seed_summary, seed_tables,<br/>oracle_comparison, rebuild,<br/>duplicates_study"]
    BASE["baselines and controls<br/>sequence_eval, separability_probe,<br/>benchmark_gate, unsupervised_baselines,<br/>detection_latency"]
    FREEZE[["experiments/audit/RESULTS_FREEZE.md<br/>every table and figure,<br/>each with its source file"]]

    KB --> GEN --> DS --> CACHE --> TRAIN
    Q --> TRAIN --> RUNS --> SUM --> FREEZE
    DS --> BASE --> FREEZE
```

Plain-text version:

```
 knowledge base ─► agentpoison_sequences ─► benchmark (.npz) ─► cached_dataset ─┐
                                                   │                             │
                                                   └─► baselines and controls ───┼──────┐
                                                       (sequence_eval, probes,   │      │
                                                        gate, unsupervised)      ▼      │
 seed_queue / main_benchmark_plan ─► train_variant (Trainer) ─► runs: checkpoint,       │
                                                                run_config, scores      │
                                                                     │                  │
                     seed_summary · seed_tables · oracle_comparison · rebuild ·         │
                     duplicates_study  ─────────────────────────────────────────►  RESULTS_FREEZE.md
```

### How the benchmark is built (`agentpoison_sequences.py`)

| Step | Detail |
|---|---|
| Background traffic | Records arrive in topical sessions of 12, each drawn from the 60 nearest neighbours of a random entry |
| Retrieval | At each interaction, with probability 0.3, a random stored record is retrieved; this is independent of the label |
| Attack (poisoned sequences) | k records (5 to 20), one per consecutive interaction, drawn without replacement from the max(8, k) nearest neighbours of one entry; they replace benign interactions, so the record count does not change |
| Matched control (clean sequences) | A cluster of the same size, drawn the same way, at random positions within up to 120 interactions |
| Snapshots | Every 10 interactions (20 per sequence); window = 10 snapshots, 11 windows per sequence |
| Labels | A record is injected or not; a window is positive once any injected record is present; a sequence is poisoned or clean |

### Module responsibilities

| Layer | Modules |
|---|---|
| Data | `agentpoison_sequences` (build), `dataset` (windows, structural features), `cached_dataset` (fast exact loader), `noise_transforms` (degradations) |
| Model | `tgn_detector` (the detector), `static_gnn_model`, `ablation_meanpool_model`, `component_ablation_models` (ablations) |
| Training | `models/train.py` (Trainer, losses, checkpoint rule), `train_variant`, `seed_queue`, `main_benchmark_plan` |
| Scoring | `score_windows` (any checkpoint, any split, stated batch size) |
| Results | `seed_summary`, `seed_tables`, `oracle_comparison`, `rebuild`, `duplicates_study`, `compute_profile`, `robustness_sweep` |
| Baselines and controls | `sequence_eval`, `separability_probe`, `benchmark_gate`, `unsupervised_baselines`, `detection_latency`, `complementarity` |
| Audit analysis | `attack_size_grid` (size, duplication and burst length varied apart) |
