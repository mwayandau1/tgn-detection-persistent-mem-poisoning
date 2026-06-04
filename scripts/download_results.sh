#!/bin/bash
# ============================================================
# Download Results from vast.ai Instance
# ============================================================
# Run this FROM YOUR LOCAL MACHINE (Windows via Git Bash/WSL)
# to pull experiment results back from your vast.ai instance.
#
# Usage:
#   bash scripts/download_results.sh <vast_ip> <vast_port> [run_name]
#
# Example:
#   bash scripts/download_results.sh 209.20.156.51 12345
#   bash scripts/download_results.sh 209.20.156.51 12345 run_20260604_150000
# ============================================================

set -e

VAST_IP="${1:?Usage: $0 <vast_ip> <vast_port> [run_name]}"
VAST_PORT="${2:?Usage: $0 <vast_ip> <vast_port> [run_name]}"
RUN_NAME="${3:-}"  # Optional: specific run to download

LOCAL_DIR="experiments/vastai_results"
REMOTE_DIR="/workspace/experiments"

mkdir -p "$LOCAL_DIR"

echo "=== Downloading Results from vast.ai ==="
echo "  Host: ${VAST_IP}:${VAST_PORT}"

if [ -n "$RUN_NAME" ]; then
    echo "  Run:  ${RUN_NAME}"
    rsync -avz --progress \
        -e "ssh -p ${VAST_PORT}" \
        "root@${VAST_IP}:${REMOTE_DIR}/${RUN_NAME}/" \
        "${LOCAL_DIR}/${RUN_NAME}/"
else
    echo "  Downloading ALL experiments..."
    rsync -avz --progress \
        --include='*/' \
        --include='*.json' \
        --include='*.pt' \
        --include='*.yaml' \
        --include='*.png' \
        --exclude='*' \
        -e "ssh -p ${VAST_PORT}" \
        "root@${VAST_IP}:${REMOTE_DIR}/" \
        "${LOCAL_DIR}/"
fi

echo ""
echo "=== Downloaded to: ${LOCAL_DIR} ==="
ls -la "$LOCAL_DIR"
