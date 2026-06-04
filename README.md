# TGN-Based Detection of Persistent Memory Poisoning in LLM Agents

Master thesis implementation: Temporal Graph Neural Network detector for identifying persistent memory poisoning in autonomous LLM agents.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (adjust for your CUDA version)
# CPU only:
pip install torch-geometric
# With CUDA 11.8:
# pip install torch-geometric --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### 2. Generate Synthetic Dataset (Stage A — No LLM Required)

This generates labeled memory graph trajectories using synthetic embeddings. Fast, runs on CPU, good for validating the architecture.

```bash
python -m src.data_generation.synthetic_generator --config configs/default.yaml --output data/raw/synthetic
```

This produces:
- 300 interaction sequences (180 clean, 120 poisoned across 3 attack types)
- Each sequence = 20 graph snapshots (200 interactions / 10 snapshot interval)
- Saved as `.npz` files with node features, edges, and labels

### 3. Train the Detector

```bash
# Full training (Phase 1 contrastive + Phase 2 supervised)
python -m src.models.train --config configs/default.yaml --data data/raw/synthetic --output experiments/run_001

# Skip contrastive pre-training (faster, slightly worse)
python -m src.models.train --config configs/default.yaml --data data/raw/synthetic --output experiments/run_001 --skip-phase1
```

### 4. Evaluate

```bash
python -m src.models.evaluate --config configs/default.yaml --checkpoint experiments/run_001/phase2_best.pt --data data/raw/synthetic --output experiments/run_001/eval_results.json
```

### 5. (Optional) Generate Realistic Dataset with LLM (Stage B)

Requires either a local GGUF model or an API key:

```bash
# With OpenAI API
set OPENAI_API_KEY=<your-key>
python -m src.data_generation.llm_generator --config configs/default.yaml --mode api --num-sequences 50

# With local LLaMA model
set LLAMA_MODEL_PATH=models/llama-3-8b-instruct.Q4_K_M.gguf
python -m src.data_generation.llm_generator --config configs/default.yaml --mode local
```

## Project Structure

```
AI-poisoned/
├── configs/
│   └── default.yaml              # All hyperparameters and settings
├── src/
│   ├── data_generation/
│   │   ├── synthetic_generator.py  # Stage A: synthetic dataset (no LLM)
│   │   ├── llm_generator.py       # Stage B: LLM-powered realistic dataset
│   │   └── dataset.py             # PyTorch dataset loader
│   ├── models/
│   │   ├── tgn_detector.py        # TGN model architecture
│   │   ├── train.py               # Training script (Phase 1 + 2)
│   │   └── evaluate.py            # Evaluation with all metrics
│   └── utils/
├── data/
│   ├── raw/                       # Generated datasets
│   └── processed/                 # Preprocessed data
├── experiments/                   # Training runs and results
├── notebooks/                     # Jupyter notebooks for analysis
└── requirements.txt
```

## Dataset Format

Each sequence is a directory containing graph snapshots as `.npz` files:

```
data/raw/synthetic/
├── clean_0000/
│   ├── snapshot_0000.npz
│   ├── snapshot_0001.npz
│   └── ...
├── poisoned_minja_query_only_0000/
│   └── ...
├── metadata.json                  # Labels and attack info
└── splits.json                    # Train/val/test split
```

Each `.npz` contains:
- `node_features`: (N, 388) — 384-dim embedding + 4 meta features
- `edge_index`: (2, E) — sparse edge list
- `edge_weights`: (E,) — cosine similarity weights
- `node_labels`: (N,) — 0=benign, 1=poisoned
- `graph_label`: (1,) — 0=clean, 1=poisoned at this timestep

## Key Design Decisions

1. **Start with synthetic data** — validates architecture without expensive LLM calls
2. **Snapshot-based** — batch processing of graph states (not streaming, per proposal scope)
3. **Multi-task loss** — graph-level + node-level detection trained jointly
4. **Focal loss** — handles extreme class imbalance at node level (few poisoned records vs many benign)

## Next Steps (Research Roadmap)

- [ ] Generate synthetic dataset and verify graph statistics
- [ ] Train baseline model, confirm it learns something
- [ ] Implement baselines (IsolationForest, DOMINANT, adapted GUARDIAN)
- [ ] Generate LLM-powered dataset for realistic evaluation
- [ ] Hyperparameter sweep (temporal window T, loss weights)
- [ ] Ablation studies
- [ ] Out-of-distribution generalization test
- [ ] Write thesis chapters alongside experiments
