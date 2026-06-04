FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps for PyG sparse operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl rsync openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric with CUDA support
RUN pip install --no-cache-dir torch-geometric \
    pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Copy project
COPY . .

# Default: run training
CMD ["python", "-m", "src.models.train", "--config", "configs/default.yaml", "--data", "data/raw/synthetic", "--output", "experiments/run_vastai"]
