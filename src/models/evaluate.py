"""
Evaluation script for the TGN Memory Poisoning Detector.
Computes all metrics defined in the thesis proposal.

Usage:
    python -m src.models.evaluate --config configs/default.yaml --checkpoint experiments/run_001/phase2_best.pt --data data/raw/synthetic
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import yaml

from src.models.tgn_detector import TGNDetector
from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories


class Evaluator:
    """Evaluates the trained detector on test set."""

    def __init__(self, config: dict, checkpoint_path: str, data_dir: str, device: str = "auto"):
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = TGNDetector(config).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # Load test dataset
        T = config["training"]["temporal_window"]
        embedding_dim = config["simulation"]["embedding_dim"]
        self.test_dataset = MemoryTrajectoryDataset(data_dir, split="test", temporal_window=T, embedding_dim=embedding_dim)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_trajectories,
        )

    @torch.no_grad()
    def evaluate_graph_level(self) -> dict:
        """Evaluate graph-level detection metrics."""
        all_scores = []
        all_labels = []

        for batch in self.test_loader:
            graphs = [g.to(self.device) for g in batch["graphs"]]
            graph_labels = batch["graph_labels"]

            graph_scores, _ = self.model(graphs)

            all_scores.extend(graph_scores.cpu().numpy())
            all_labels.extend(graph_labels.numpy())

        scores = np.array(all_scores)
        labels = np.array(all_labels)

        results = {}

        # AUROC
        if len(np.unique(labels)) > 1:
            results["auroc"] = float(roc_auc_score(labels, scores))
        else:
            results["auroc"] = 0.0

        # F1 at threshold 0.5
        preds = (scores >= 0.5).astype(int)
        results["f1_at_0.5"] = float(f1_score(labels, preds))

        # Average Precision
        results["average_precision"] = float(average_precision_score(labels, scores))

        # FPR at TPR=0.9
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = np.argmin(np.abs(tpr - 0.9))
        results["fpr_at_tpr_0.9"] = float(fpr[idx])

        # Best F1 threshold
        precisions, recalls, thresholds_pr = precision_recall_curve(labels, scores)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1s)
        results["best_f1"] = float(f1s[best_idx])
        results["best_threshold"] = float(thresholds_pr[best_idx]) if best_idx < len(thresholds_pr) else 0.5

        return results

    @torch.no_grad()
    def evaluate_node_level(self) -> dict:
        """Evaluate node-level detection (average precision)."""
        all_node_scores = []
        all_node_labels = []

        for batch in self.test_loader:
            graphs = [g.to(self.device) for g in batch["graphs"]]

            _, node_scores = self.model(graphs)

            last_graph = graphs[-1]
            if hasattr(last_graph, "node_labels"):
                node_labels = last_graph.node_labels.cpu().numpy()
                ns = node_scores.cpu().numpy()
                min_len = min(len(ns), len(node_labels))
                all_node_scores.extend(ns[:min_len])
                all_node_labels.extend(node_labels[:min_len])

        scores = np.array(all_node_scores)
        labels = np.array(all_node_labels)

        results = {}
        if len(np.unique(labels)) > 1:
            results["node_average_precision"] = float(average_precision_score(labels, scores))
            results["node_auroc"] = float(roc_auc_score(labels, scores))
        else:
            results["node_average_precision"] = 0.0
            results["node_auroc"] = 0.0

        return results

    @torch.no_grad()
    def evaluate_detection_latency(self) -> dict:
        """
        Measure detection latency: how many timesteps after injection
        until the model detects poisoning (score > threshold).
        """
        # Load metadata to find poisoned sequences
        data_dir = Path(self.test_dataset.data_dir)
        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)

        poisoned_seqs = {m["sequence_id"] for m in metadata if m["label"] == "poisoned"}

        latencies = []
        threshold = 0.5
        T = self.config["training"]["temporal_window"]

        for seq_id in poisoned_seqs:
            seq_dir = data_dir / seq_id
            if not seq_dir.exists():
                continue

            snapshots = sorted(seq_dir.glob("snapshot_*.npz"))
            if len(snapshots) < T:
                continue

            # Find first snapshot where graph_label becomes 1 (poisoning started)
            first_poisoned_idx = None
            for i, snap_path in enumerate(snapshots):
                snap = np.load(snap_path)
                if snap["graph_label"][0] == 1:
                    first_poisoned_idx = i
                    break

            if first_poisoned_idx is None:
                continue

            # Check when model first detects it
            detected_idx = None
            for start in range(max(0, first_poisoned_idx - T + 1), len(snapshots) - T + 1):
                # Load window
                window_graphs = []
                for i in range(start, start + T):
                    snap = np.load(snapshots[i])
                    from torch_geometric.data import Data, Batch
                    edge_weights = torch.tensor(snap["edge_weights"], dtype=torch.float32)
                    edge_attr = edge_weights.unsqueeze(-1) if edge_weights.numel() > 0 else torch.zeros((0, 1), dtype=torch.float32)
                    num_nodes = snap["node_features"].shape[0]
                    g = Data(
                        x=torch.tensor(snap["node_features"], dtype=torch.float32),
                        edge_index=torch.tensor(snap["edge_index"], dtype=torch.long),
                        edge_attr=edge_attr,
                        node_labels=torch.tensor(snap["node_labels"], dtype=torch.long),
                        node_ids=torch.arange(num_nodes, dtype=torch.long),
                    )
                    window_graphs.append(g)

                batched = [Batch.from_data_list([g]).to(self.device) for g in window_graphs]
                graph_score, _ = self.model(batched)

                if graph_score.item() > threshold:
                    detected_idx = start + T - 1
                    break

            if detected_idx is not None and first_poisoned_idx is not None:
                latency = detected_idx - first_poisoned_idx
                latencies.append(max(0, latency))

        results = {}
        if latencies:
            results["mean_detection_latency"] = float(np.mean(latencies))
            results["median_detection_latency"] = float(np.median(latencies))
            results["std_detection_latency"] = float(np.std(latencies))
            results["min_detection_latency"] = int(np.min(latencies))
            results["max_detection_latency"] = int(np.max(latencies))
        else:
            results["mean_detection_latency"] = -1
            results["note"] = "No poisoned sequences detected in test set"

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TGN Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/raw/synthetic")
    parser.add_argument("--output", type=str, default="experiments/run_001/eval_results.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluator = Evaluator(config, args.checkpoint, args.data, device=args.device)

    print("Evaluating graph-level detection...")
    graph_results = evaluator.evaluate_graph_level()
    print(f"  AUROC: {graph_results['auroc']:.4f}")
    print(f"  F1@0.5: {graph_results['f1_at_0.5']:.4f}")
    print(f"  AP: {graph_results['average_precision']:.4f}")
    print(f"  FPR@TPR=0.9: {graph_results['fpr_at_tpr_0.9']:.4f}")

    print("\nEvaluating node-level detection...")
    node_results = evaluator.evaluate_node_level()
    print(f"  Node AP: {node_results['node_average_precision']:.4f}")
    print(f"  Node AUROC: {node_results['node_auroc']:.4f}")

    print("\nEvaluating detection latency...")
    latency_results = evaluator.evaluate_detection_latency()
    if "mean_detection_latency" in latency_results:
        print(f"  Mean latency: {latency_results['mean_detection_latency']:.1f} snapshots")

    # Combine and save
    all_results = {
        "graph_level": graph_results,
        "node_level": node_results,
        "detection_latency": latency_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
