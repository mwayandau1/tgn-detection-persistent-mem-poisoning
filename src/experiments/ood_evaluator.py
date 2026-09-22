"""
OOD and Noise Stress-Test Evaluator
====================================
Evaluates trained TGN memory poisoning detector checkpoints under
algorithmic noise perturbations: Gaussian embedding noise, cosine similarity threshold shifts,
metadata corruption, and edge dropouts.

Unbuffered progress logging.
"""

import sys
import yaml
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.tgn_detector import TGNDetector
from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories
from src.data_generation.noise_transforms import GraphNoiseTransform


def calculate_fpr_at_tpr90(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate False Positive Rate (FPR) when True Positive Rate (TPR) is at least 90%."""
    from sklearn.metrics import roc_curve
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    target_idx = np.where(tpr >= 0.90)[0]
    if len(target_idx) == 0:
        return 1.0
    return float(fpr[target_idx[0]])


def preload_dataset_to_gpu(dataset: MemoryTrajectoryDataset, device: torch.device, max_samples: int = 100, seed: int = 0) -> List[dict]:
    """Pre-load a random sample of trajectory windows into memory and move tensors to GPU.

    The sample is RANDOM, and deliberately so. This previously took `range(max_samples)` --
    the first N windows in dataset order. Because a dataset enumerates every window of one
    sequence before moving to the next, the default of 100 evaluated roughly nine sequences
    out of five hundred, selected by file order, and every figure in the robustness section
    rested on that slice. On ap_burst2k the first 150 windows span 14 of 500 sequences and
    score about 0.05 above the full test partition, so the bias was not academic.

    The whole split is not used instead because preloading it to VRAM needs far more memory
    than the 4 GB card this project reports timings on; sampling is a memory constraint, and
    it is the non-randomness that was the defect.
    """
    num_items = min(len(dataset), max_samples)
    idx = np.random.default_rng(seed).choice(len(dataset), size=num_items, replace=False)
    idx.sort()
    print(f"Pre-loading {num_items} of {len(dataset)} test windows "
          f"(random sample, seed {seed}) to {device}...", flush=True)

    items = []
    for i in idx:
        item = dataset[int(i)]
        gpu_graphs = [g.to(device) for g in item["graphs"]]
        items.append({
            "graphs": gpu_graphs,
            "graph_label": item["graph_label"],
            "node_labels": item["node_labels"].to(device) if isinstance(item["node_labels"], torch.Tensor) else torch.tensor(item["node_labels"], device=device),
            "sequence_id": item["sequence_id"],
        })

    print(f"Successfully pre-loaded {len(items)} items to {device}.", flush=True)
    return items


def evaluate_preloaded_data(model: nn.Module,
                            preloaded_items: List[dict],
                            device: torch.device,
                            batch_size: int = 4,
                            embedding_sigma: float = 0.0,
                            threshold: float = None,
                            meta_corrupt_prob: float = 0.0,
                            edge_drop_rate: float = 0.0) -> Dict[str, float]:
    """
    Evaluates model on GPU-preloaded dataset split under specific perturbation parameters.
    """
    model.eval()
    # The detector carries per-node GRU memory across calls, so without this the state left
    # behind by the previous condition leaks into the next one and every result depends on the
    # order the sweeps happened to run in. It showed up as the three zero-perturbation
    # conditions (sigma=0, p=0, r=0) disagreeing when all three are no-ops and must be
    # identical. Every other evaluation path in src/ resets here; this one did not.
    model.reset_memory()

    graph_targets = []
    graph_preds = []
    node_targets = []
    node_preds = []

    num_items = len(preloaded_items)
    with torch.no_grad():
        for start_idx in range(0, num_items, batch_size):
            batch_items = preloaded_items[start_idx : start_idx + batch_size]
            
            batch_dicts = []
            for item in batch_items:
                p_graphs = GraphNoiseTransform.perturb_trajectory(
                    graphs=item["graphs"],
                    embedding_sigma=embedding_sigma,
                    threshold=threshold,
                    meta_corrupt_prob=meta_corrupt_prob,
                    edge_drop_rate=edge_drop_rate,
                    device=device,
                )
                batch_dicts.append({
                    "graphs": p_graphs,
                    "graph_label": item["graph_label"],
                    "sequence_id": item["sequence_id"],
                })

            batch_data = collate_trajectories(batch_dicts)
            batched_sequence = batch_data["graphs"]

            graph_scores, node_scores = model(batched_sequence)

            graph_targets.extend(batch_data["graph_labels"].cpu().numpy())
            graph_preds.extend(graph_scores.cpu().numpy())

            final_snapshot = batched_sequence[-1]
            if hasattr(final_snapshot, 'node_labels') and final_snapshot.node_labels is not None:
                node_targets.extend(final_snapshot.node_labels.cpu().numpy())
                node_preds.extend(node_scores.cpu().numpy())

    graph_targets = np.array(graph_targets)
    graph_preds = np.array(graph_preds)
    node_targets = np.array(node_targets)
    node_preds = np.array(node_preds)

    g_auroc = float(roc_auc_score(graph_targets, graph_preds)) if len(np.unique(graph_targets)) > 1 else 1.0
    g_ap = float(average_precision_score(graph_targets, graph_preds)) if len(np.unique(graph_targets)) > 1 else 1.0
    g_f1 = float(f1_score(graph_targets, (graph_preds >= 0.5).astype(int), zero_division=0))
    g_fpr90 = calculate_fpr_at_tpr90(graph_targets, graph_preds)

    n_auroc = float(roc_auc_score(node_targets, node_preds)) if len(np.unique(node_targets)) > 1 and len(node_targets) > 0 else 1.0
    n_ap = float(average_precision_score(node_targets, node_preds)) if len(np.unique(node_targets)) > 1 and len(node_targets) > 0 else 1.0

    return {
        "graph_auroc": round(g_auroc, 4),
        "graph_ap": round(g_ap, 4),
        "graph_f1": round(g_f1, 4),
        "fpr_at_tpr90": round(g_fpr90, 4),
        "node_auroc": round(n_auroc, 4),
        "node_ap": round(n_ap, 4),
    }


def run_all_ood_experiments(config_path: str, checkpoint_path: str, data_dir: str, output_path: str, batch_size: int = 4, max_samples: int = 100, structural_features: bool = False):
    """Run full OOD stress-test suite and write summary JSON."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Must match how the checkpoint was trained; see train.py --structural-features. The two
    # settings below have to move together: node_meta_dim widens the model's input layer and
    # structural_features widens what the dataset emits. Setting one without the other loads
    # the checkpoint against a differently-shaped model, which does not always raise -- see
    # experiments/eval_ap_scale.INVALID_train_eval_mismatch.json for what that silently
    # produced. Every ap_burst2k run is trained with this flag on.
    if structural_features:
        config["model"]["node_meta_dim"] = 3 + 4
        print("Structural node features enabled: node_meta_dim=7", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load test split into GPU VRAM once
    print(f"Loading test trajectory dataset from {data_dir}...", flush=True)
    test_dataset = MemoryTrajectoryDataset(
        data_dir=data_dir,
        split="test",
        temporal_window=config["training"]["temporal_window"],
        structural_features=structural_features,
    )
    preloaded_items = preload_dataset_to_gpu(test_dataset, device, max_samples=max_samples)
    pos = sum(int(it["graph_label"]) for it in preloaded_items)
    print(f"  sampled {len(preloaded_items)} windows: {pos} positive, "
          f"{len(preloaded_items) - pos} negative, "
          f"{len({it['sequence_id'] for it in preloaded_items})} distinct sequences", flush=True)

    # Load model checkpoint
    print(f"Loading model checkpoint from {checkpoint_path}...", flush=True)
    model = TGNDetector(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    results = {
        "baseline_clean": {},
        "embedding_noise_sweep": {},
        "threshold_sensitivity_sweep": {},
        "metadata_corruption_sweep": {},
        "edge_loss_sweep": {},
    }

    # 1. Baseline Clean Test
    print("\n--- Running Clean Baseline Evaluation ---", flush=True)
    clean_res = evaluate_preloaded_data(model, preloaded_items, device, batch_size=batch_size)
    results["baseline_clean"] = clean_res
    print(f"Baseline Clean -> Graph AUROC: {clean_res['graph_auroc']}, Node AUROC: {clean_res['node_auroc']}", flush=True)

    # 2. Embedding Noise Sweep (Gaussian sigma)
    print("\n--- Running Experiment 1: Embedding Noise Sweep (sigma) ---", flush=True)
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for s in sigmas:
        res = evaluate_preloaded_data(model, preloaded_items, device, batch_size=batch_size, embedding_sigma=s)
        results["embedding_noise_sweep"][str(s)] = res
        print(f"  sigma={s:.2f} -> Graph AUROC: {res['graph_auroc']:.4f}, AP: {res['graph_ap']:.4f}, Node AUROC: {res['node_auroc']:.4f}", flush=True)

    # 3. Threshold Sensitivity Sweep (theta)
    print("\n--- Running Experiment 2: Cosine Edge Threshold Sensitivity (theta) ---", flush=True)
    thetas = [0.4, 0.5, 0.6, 0.7, 0.8]
    for t in thetas:
        res = evaluate_preloaded_data(model, preloaded_items, device, batch_size=batch_size, threshold=t)
        results["threshold_sensitivity_sweep"][str(t)] = res
        print(f"  theta={t:.1f} -> Graph AUROC: {res['graph_auroc']:.4f}, AP: {res['graph_ap']:.4f}, Node AUROC: {res['node_auroc']:.4f}", flush=True)

    # 4. Metadata Corruption Sweep (probability p)
    print("\n--- Running Experiment 3: Metadata Corruption Sweep (prob p) ---", flush=True)
    probs = [0.0, 0.05, 0.10, 0.20, 0.30]
    for p in probs:
        res = evaluate_preloaded_data(model, preloaded_items, device, batch_size=batch_size, meta_corrupt_prob=p)
        results["metadata_corruption_sweep"][str(p)] = res
        print(f"  corrupt_p={p:.2f} -> Graph AUROC: {res['graph_auroc']:.4f}, AP: {res['graph_ap']:.4f}, Node AUROC: {res['node_auroc']:.4f}", flush=True)

    # 5. Random Edge Loss Sweep (drop rate)
    print("\n--- Running Experiment 4: Random Edge Loss Sweep (drop rate) ---", flush=True)
    rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    for r in rates:
        res = evaluate_preloaded_data(model, preloaded_items, device, batch_size=batch_size, edge_drop_rate=r)
        results["edge_loss_sweep"][str(r)] = res
        print(f"  drop_rate={r:.2f} -> Graph AUROC: {res['graph_auroc']:.4f}, AP: {res['graph_ap']:.4f}, Node AUROC: {res['node_auroc']:.4f}", flush=True)

    # Save to JSON
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSuccessfully saved OOD evaluation results to {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OOD and noise stress tests on TGN detector.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="experiments/notebook_run01/phase2_best.pt", help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data/raw/llm_generated", help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="experiments/ood_results.json", help="Path to output JSON")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum test samples to evaluate")
    parser.add_argument("--structural-features", action="store_true",
                        help="Checkpoint was trained with structural node features "
                             "(node_meta_dim=7). Required for every ap_burst2k run.")
    args = parser.parse_args()

    run_all_ood_experiments(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        output_path=args.output,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        structural_features=args.structural_features,
    )
