#!/bin/bash
# ============================================================
# vast.ai Instance Setup Script
# ============================================================
# Run this ONCE after SSH-ing into your vast.ai instance.
# Assumes a PyTorch Docker template (e.g., pytorch/pytorch:2.4.1-cuda12.4)
#
# Usage:
#   chmod +x scripts/setup_vastai.sh
#   ./scripts/setup_vastai.sh
# ============================================================

set -e  # Exit on any error

echo "=== vast.ai Instance Setup ==="
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"

# 1. Install Python dependencies
echo ""
echo "=== Installing Python dependencies ==="
pip install --no-cache-dir -r requirements.txt

# 2. Install PyTorch Geometric with CUDA support
echo ""
echo "=== Installing PyTorch Geometric ==="
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.',''))" 2>/dev/null || echo "cpu")

echo "  Torch: $TORCH_VERSION, CUDA: $CUDA_VERSION"

pip install --no-cache-dir torch-geometric
pip install --no-cache-dir pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html" 2>/dev/null || \
    echo "  Warning: Pre-built PyG wheels not found for this torch/cuda combo. Using CPU fallback."

# 3. Create directories
mkdir -p data/raw/synthetic data/raw/llm_generated experiments

# 4. Verify installation
echo ""
echo "=== Verifying Installation ==="
python -c "
import torch
import torch_geometric
from src.models.tgn_detector import TGNDetector
from src.data_generation.dataset import MemoryTrajectoryDataset
print(f'  PyTorch: {torch.__version__}')
print(f'  PyG: {torch_geometric.__version__}')
print(f'  CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB' if torch.cuda.is_available() else '')
print('  All imports OK')
"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Generate data:   ./scripts/run_experiment.sh generate"
echo "  2. Train:           ./scripts/run_experiment.sh train"
echo "  3. Full pipeline:   ./scripts/run_experiment.sh all"
