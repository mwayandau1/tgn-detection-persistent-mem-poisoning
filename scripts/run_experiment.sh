#!/bin/bash
# ============================================================
# Experiment Runner for vast.ai
# ============================================================
# Usage:
#   ./scripts/run_experiment.sh generate              # Generate dataset only
#   ./scripts/run_experiment.sh train                  # Train only (data must exist)
#   ./scripts/run_experiment.sh evaluate               # Evaluate only
#   ./scripts/run_experiment.sh all                    # Full pipeline
#   ./scripts/run_experiment.sh all --large            # Full pipeline with large dataset
#   ./scripts/run_experiment.sh sweep                  # Hyperparameter sweep
# ============================================================

set -e

# --- Configuration ---
CONFIG="${CONFIG:-configs/default.yaml}"
DATA_DIR="${DATA_DIR:-data/raw/llm_generated}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="experiments/${RUN_NAME}"
SEED="${SEED:-42}"

# Parse --large flag
LARGE_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--large" ]; then
        LARGE_MODE=true
    fi
done

# Large mode overrides
if [ "$LARGE_MODE" = true ]; then
    CONFIG="configs/large.yaml"
    echo "=== LARGE MODE: Using configs/large.yaml ==="
fi

echo "=== Experiment: ${RUN_NAME} ==="
echo "  Config: ${CONFIG}"
echo "  Data:   ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Seed:   ${SEED}"
echo ""

# --- Functions ---

generate_data() {
    echo "=== Step 1: Generating LLM Dataset ==="
    START_TIME=$SECONDS
    python -m src.data_generation.llm_generator \
        --config "$CONFIG" \
        --output "$DATA_DIR" \
        --num-sequences 167
    echo "  Done in $((SECONDS - START_TIME))s"
    echo ""

    # Sanity check
    python -c "
import numpy as np, json
from pathlib import Path
d = Path('${DATA_DIR}')
meta = json.load(open(d / 'metadata.json'))
s = np.load(d / meta[0]['sequence_id'] / 'snapshot_0000.npz')
print(f'  Sequences: {len(meta)}')
print(f'  Feature dim: {s[\"node_features\"].shape[1]}')
print(f'  Clean: {sum(1 for m in meta if m[\"label\"]==\"clean\")}')
print(f'  Poisoned: {sum(1 for m in meta if m[\"label\"]==\"poisoned\")}')
"
}

train_model() {
    echo "=== Step 2: Training ==="
    mkdir -p "$OUTPUT_DIR"
    START_TIME=$SECONDS
    python -m src.models.train \
        --config "$CONFIG" \
        --data "$DATA_DIR" \
        --output "$OUTPUT_DIR" \
        --seed "$SEED"
    echo "  Done in $((SECONDS - START_TIME))s"
    echo ""
}

evaluate_model() {
    echo "=== Step 3: Evaluating ==="
    CHECKPOINT="${OUTPUT_DIR}/phase2_best.pt"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "  Error: Checkpoint not found at ${CHECKPOINT}"
        exit 1
    fi
    python -m src.models.evaluate \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --data "$DATA_DIR" \
        --output "${OUTPUT_DIR}/eval_results.json"

    echo ""
    echo "=== Results ==="
    python -c "
import json
r = json.load(open('${OUTPUT_DIR}/eval_results.json'))
print(f'  Graph AUROC:     {r[\"graph_level\"][\"auroc\"]:.4f}')
print(f'  Graph F1:        {r[\"graph_level\"][\"best_f1\"]:.4f}')
print(f'  Graph AP:        {r[\"graph_level\"][\"average_precision\"]:.4f}')
print(f'  Node AUROC:      {r[\"node_level\"][\"node_auroc\"]:.4f}')
print(f'  Node AP:         {r[\"node_level\"][\"node_average_precision\"]:.4f}')
dl = r.get('detection_latency', {})
if 'mean_detection_latency' in dl and dl['mean_detection_latency'] >= 0:
    print(f'  Mean Latency:    {dl[\"mean_detection_latency\"]:.1f} snapshots')
"
}

visualize() {
    echo "=== Step 4: Generating Plots ==="
    mkdir -p "${OUTPUT_DIR}/plots"
    python -m src.utils.visualize \
        --data "$DATA_DIR" \
        --mode all \
        --save "${OUTPUT_DIR}/plots"
}

hyperparameter_sweep() {
    echo "=== Hyperparameter Sweep ==="
    for LR in 0.0001 0.0005 0.001; do
        for DROPOUT in 0.2 0.3 0.4; do
            SWEEP_NAME="sweep_lr${LR}_do${DROPOUT}"
            SWEEP_DIR="experiments/${SWEEP_NAME}"
            echo "--- ${SWEEP_NAME} ---"
            mkdir -p "$SWEEP_DIR"

            # Create temp config with overrides
            python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['training']['learning_rate'] = ${LR}
cfg['model']['dropout'] = ${DROPOUT}
with open('${SWEEP_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
            python -m src.models.train \
                --config "${SWEEP_DIR}/config.yaml" \
                --data "$DATA_DIR" \
                --output "$SWEEP_DIR" \
                --skip-phase1 \
                --seed "$SEED" 2>&1 | tail -5

            python -m src.models.evaluate \
                --config "${SWEEP_DIR}/config.yaml" \
                --checkpoint "${SWEEP_DIR}/phase2_best.pt" \
                --data "$DATA_DIR" \
                --output "${SWEEP_DIR}/eval_results.json" 2>&1 | tail -3
        done
    done

    # Summary table
    echo ""
    echo "=== Sweep Summary ==="
    python -c "
import json, os
from pathlib import Path
print(f'{\"Run\":<30} {\"Graph AUROC\":>12} {\"Node AUROC\":>12} {\"Graph AP\":>12}')
print('-' * 70)
for d in sorted(Path('experiments').glob('sweep_*')):
    rf = d / 'eval_results.json'
    if rf.exists():
        r = json.load(open(rf))
        print(f'{d.name:<30} {r[\"graph_level\"][\"auroc\"]:>12.4f} {r[\"node_level\"][\"node_auroc\"]:>12.4f} {r[\"graph_level\"][\"average_precision\"]:>12.4f}')
"
}

# --- Main ---
ACTION="${1:-all}"

case "$ACTION" in
    generate)
        generate_data
        ;;
    train)
        train_model
        ;;
    evaluate)
        evaluate_model
        ;;
    visualize)
        visualize
        ;;
    sweep)
        generate_data
        hyperparameter_sweep
        ;;
    all)
        generate_data
        train_model
        evaluate_model
        visualize
        ;;
    *)
        echo "Usage: $0 {generate|train|evaluate|visualize|sweep|all} [--large]"
        exit 1
        ;;
esac

echo ""
echo "=== Complete ==="
