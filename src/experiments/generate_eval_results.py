"""
Generates eval_results.json (graph-level, node-level, detection-latency) for a
checkpoint, matching the schema the all_in_one_tgn_detector notebook's
"Model Evaluation Suite" section reads. Mirrors that notebook's Evaluator class
exactly, using the current src/ TGNDetector/MemoryTrajectoryDataset.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    precision_recall_curve, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.tgn_detector import TGNDetector
from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories


class Evaluator:
    """Evaluates the trained detector on test set."""

    def __init__(self, config: dict, checkpoint_path: str, data_dir: str, device: torch.device,
                 model_cls: type = TGNDetector, batch_size: int = None,
                 structural_features: bool = False):
        self.config = config
        self.device = device
        self.structural_features = structural_features

        # Must match how the checkpoint was trained; see train.py --structural-features.
        if structural_features:
            config["model"]["node_meta_dim"] = 3 + 4

        self.model = model_cls(config).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

        T = config["training"]["temporal_window"]
        embedding_dim = config["simulation"]["embedding_dim"]
        self.test_dataset = MemoryTrajectoryDataset(data_dir, split="test", temporal_window=T,
                                                     embedding_dim=embedding_dim,
                                                     structural_features=structural_features)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size if batch_size is not None else config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_trajectories,
        )

    @torch.no_grad()
    def evaluate_graph_level(self) -> dict:
        self.model.reset_memory()
        all_scores, all_labels = [], []

        for batch in self.test_loader:
            graphs = [g.to(self.device) for g in batch["graphs"]]
            graph_labels = batch["graph_labels"]

            graph_scores, _ = self.model(graphs)

            all_scores.extend(graph_scores.cpu().numpy())
            all_labels.extend(graph_labels.numpy())

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        scores = np.array(all_scores)
        labels = np.array(all_labels)

        results = {}
        if len(np.unique(labels)) > 1:
            results["auroc"] = float(roc_auc_score(labels, scores))
        else:
            results["auroc"] = 0.0

        preds = (scores >= 0.5).astype(int)
        results["f1_at_0.5"] = float(f1_score(labels, preds))
        results["average_precision"] = float(average_precision_score(labels, scores))

        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = np.argmin(np.abs(tpr - 0.9))
        results["fpr_at_tpr_0.9"] = float(fpr[idx])

        precisions, recalls, thresholds_pr = precision_recall_curve(labels, scores)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1s)
        results["best_f1"] = float(f1s[best_idx])
        results["best_threshold"] = float(thresholds_pr[best_idx]) if best_idx < len(thresholds_pr) else 0.5

        return results

    @torch.no_grad()
    def evaluate_node_level(self) -> dict:
        self.model.reset_memory()
        all_node_scores, all_node_labels = [], []

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

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

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
        data_dir = Path(self.test_dataset.data_dir)
        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)

        test_seq_ids = set(self.test_dataset.sequence_ids)
        poisoned_seqs = {m["sequence_id"] for m in metadata
                         if m["label"] == "poisoned" and m["sequence_id"] in test_seq_ids}
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

            first_poisoned_idx = None
            for i, snap_path in enumerate(snapshots):
                snap = np.load(snap_path)
                if snap["graph_label"][0] == 1:
                    first_poisoned_idx = i
                    break

            if first_poisoned_idx is None:
                continue

            detected_idx = None
            for start in range(max(0, first_poisoned_idx - T + 1), len(snapshots) - T + 1):
                window_graphs = []
                for i in range(start, start + T):
                    snap = np.load(snapshots[i])
                    edge_weights = torch.tensor(snap["edge_weights"], dtype=torch.float32)
                    edge_attr = edge_weights.unsqueeze(-1) if edge_weights.numel() > 0 else torch.zeros((0, 1), dtype=torch.float32)
                    num_nodes = snap["node_features"].shape[0]

                    # This path builds graphs straight from the .npz, bypassing the
                    # dataset, so it must reproduce the same feature width the model
                    # was trained on.
                    feats = snap["node_features"]
                    if self.structural_features:
                        feats = np.concatenate(
                            [feats, MemoryTrajectoryDataset._structural_features(
                                num_nodes, snap["edge_index"], snap["edge_weights"])],
                            axis=1,
                        )

                    g = Data(
                        x=torch.tensor(feats, dtype=torch.float32),
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

        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/raw/llm_generated")
    parser.add_argument("--run-dir", default="experiments/run_current_arch")
    parser.add_argument("--checkpoint", default="phase2_best.pt")
    parser.add_argument("--model-cls", default="TGNDetector",
                         choices=["TGNDetector", "TGNDetectorMeanPool"],
                         help="Model class to instantiate before loading the checkpoint")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Override config['training']['batch_size'] for the eval DataLoader. "
                              "Use 1 to avoid node_ids colliding across sequences batched together "
                              "in the node-memory module (see Section 4.11).")
    parser.add_argument("--structural-features", action="store_true",
                         help="Must match the flag the checkpoint was trained with.")
    parser.add_argument("--output", default=None,
                         help="Override output path (default: <run-dir>/eval_results.json)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model_cls == "TGNDetectorMeanPool":
        from src.experiments.ablation_meanpool_model import TGNDetectorMeanPool
        model_cls = TGNDetectorMeanPool
    else:
        model_cls = TGNDetector

    checkpoint_path = Path(args.run_dir) / args.checkpoint
    evaluator = Evaluator(config, str(checkpoint_path), args.data, device=device, model_cls=model_cls,
                           batch_size=args.batch_size,
                           structural_features=args.structural_features)

    print("Running graph-level evaluation...")
    graph_res = evaluator.evaluate_graph_level()
    print("Running node-level evaluation...")
    node_res = evaluator.evaluate_node_level()
    print("Running detection-latency evaluation...")
    latency_res = evaluator.evaluate_detection_latency()

    results = {"graph_level": graph_res, "node_level": node_res, "detection_latency": latency_res}
    print(json.dumps(results, indent=2))

    out_path = Path(args.output) if args.output else Path(args.run_dir) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
