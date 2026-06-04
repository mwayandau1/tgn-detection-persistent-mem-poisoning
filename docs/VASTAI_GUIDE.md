# vast.ai Deployment Guide

## Files Created for vast.ai

| File | Purpose |
|------|---------|
| `Dockerfile` | Reproducible container with PyTorch + PyG + CUDA |
| `.dockerignore` | Excludes venv/data/experiments from Docker builds |
| `.gitignore` | Keeps repo clean — code+configs versioned, data/results not |
| `configs/large.yaml` | 1000 sequences, batch 32, 30 epochs for GPU training |
| `scripts/setup_vastai.sh` | One-shot env setup on the instance |
| `scripts/run_experiment.sh` | Full pipeline runner (generate/train/eval/sweep) |
| `scripts/download_results.sh` | Pull results back to your laptop |

---

## Step-by-Step: vast.ai Workflow

### 1. Push Code to GitHub (Recommended)

Initialize a repo so you can clone onto vast.ai:

```powershell
cd C:\PRJTS\msearch\impl\AI-poisoned
git init
git add -A
git commit -m "TGN detector with fixes"
# Create repo on GitHub, then:
git remote add origin https://github.com/<your-username>/AI-poisoned.git
git push -u origin main
```

> If you don't want to use GitHub, you can use `scp` to upload directly (see Step 3 alternative).

### 2. Rent a vast.ai Instance

1. Go to [vast.ai/console/create](https://cloud.vast.ai/create/)
2. **Recommended specs for your workload:**

| Spec | Minimum | Recommended |
|------|---------|-------------|
| GPU | RTX 3060 (12GB) | RTX 3090 / A5000 (24GB) |
| RAM | 16 GB | 32 GB |
| Disk | 30 GB | 50 GB |
| Docker Image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` | Same |

3. Under **Docker Image**, enter: `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`
4. Click **RENT** — note the SSH connection details (IP + port)

> Your current model is ~650K params — even an RTX 3060 is plenty.
> You're paying for **dataset generation speed** and **sweep parallelism**, not model size.

### 3. Connect and Upload Code

```bash
# SSH into the instance (use the IP and port from vast.ai dashboard)
ssh -p <PORT> root@<VAST_IP>

# Clone your repo
cd /workspace
git clone https://github.com/<your-username>/AI-poisoned.git
cd AI-poisoned
```

**Alternative (no GitHub)** — upload from your local PowerShell:

```powershell
# From your Windows machine, using scp
scp -P <PORT> -r C:\PRJTS\msearch\impl\AI-poisoned\src root@<VAST_IP>:/workspace/AI-poisoned/
scp -P <PORT> -r C:\PRJTS\msearch\impl\AI-poisoned\configs root@<VAST_IP>:/workspace/AI-poisoned/
scp -P <PORT> -r C:\PRJTS\msearch\impl\AI-poisoned\scripts root@<VAST_IP>:/workspace/AI-poisoned/
scp -P <PORT> C:\PRJTS\msearch\impl\AI-poisoned\requirements.txt root@<VAST_IP>:/workspace/AI-poisoned/
```

### 4. Set Up the Environment

```bash
cd /workspace/AI-poisoned
chmod +x scripts/*.sh
./scripts/setup_vastai.sh
```

This installs all dependencies and verifies CUDA + PyG work correctly.

### 5. Run Experiments

#### Option A: Full pipeline with large config (recommended)

```bash
./scripts/run_experiment.sh all --large
```

This generates 1000 sequences → trains Phase 1+2 → evaluates → generates plots.

#### Option B: Standard config

```bash
./scripts/run_experiment.sh all
```

#### Option C: Hyperparameter sweep

```bash
./scripts/run_experiment.sh sweep
```

Sweeps over `learning_rate × dropout` and prints a comparison table.

#### Option D: Step-by-step

```bash
# Generate data only
./scripts/run_experiment.sh generate

# Train only (data must exist)
./scripts/run_experiment.sh train

# Evaluate only
./scripts/run_experiment.sh evaluate
```

### 6. Download Results to Your Laptop

From **Git Bash or WSL** on your Windows machine:

```bash
cd /c/PRJTS/msearch/impl/AI-poisoned
bash scripts/download_results.sh <VAST_IP> <PORT>
```

This rsync's all experiment JSONs, checkpoints, and plots to `experiments/vastai_results/`.

### 7. Destroy the Instance

Once you've downloaded results, **destroy the instance** from the vast.ai dashboard to stop charges.

---

## Cost Estimate

| GPU | $/hr | 1000-seq Generate | Full Train | Sweep (9 runs) |
|-----|------|-------------------|------------|----------------|
| RTX 3060 | ~$0.10 | ~15 min | ~30 min | ~3 hr |
| RTX 3090 | ~$0.20 | ~8 min | ~15 min | ~1.5 hr |
| A5000 | ~$0.25 | ~8 min | ~12 min | ~1 hr |

A full pipeline run (generate + train + eval) on 1000 sequences should cost **under $0.50**.
