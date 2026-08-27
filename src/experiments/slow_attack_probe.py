"""
Slow-Attack Partial-Visibility Probe
=======================================
Evaluates the trained TGN against the "genuinely slow" incremental_slow OOD
set (data/raw/llm_generated_slow_ood, injection_spread=200 - well beyond a
single temporal_window*snapshot_interval=100-interaction detection window).

Because the memory graph is cumulative (records are appended, never pruned),
every window sampled *after* injection fully completes still sees the whole
poisoned cluster - same as the original spread=100 config. The meaningful
test is windows sampled *while* injection is still in progress, where only
some of the eventual poisoned records have landed. This script buckets every
window by exactly how many poisoned records were actually present in its
final snapshot, and reports detection performance per bucket - i.e. does the
model need to see most of the cluster, or does 1-2 records already give it
away.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.tgn_detector import TGNDetector
from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/llm_generated_slow_ood")
    parser.add_argument("--run-dir", default="experiments/run_current_arch")
    parser.add_argument("--checkpoint", default="phase2_best.pt")
    parser.add_argument("--config", default="configs/default.yaml",
                         help="Model architecture config (must match the checkpoint's dims)")
    parser.add_argument("--temporal-window", type=int, default=10)
    parser.add_argument("--output", default="experiments/slow_attack_probe_results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TGNDetector(config).to(device)
    state_dict = torch.load(Path(args.run_dir) / args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with open(Path(args.data) / "metadata.json") as f:
        metadata = json.load(f)
    all_seq_ids = [m["sequence_id"] for m in metadata]

    ds = MemoryTrajectoryDataset(args.data, temporal_window=args.temporal_window, sequence_ids=all_seq_ids)
    print(f"Scoring {len(ds)} windows across {len(all_seq_ids)} slow-attack sequences...")

    # bucket key = number of poisoned nodes present in the window's final snapshot
    buckets = defaultdict(list)  # n_poisoned_visible -> list of (score,)
    all_scores, all_labels = [], []

    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            batch = collate_trajectories([item])
            graphs = [g.to(device) for g in batch["graphs"]]
            graph_scores, _ = model(graphs)
            score = graph_scores.squeeze().item()

            n_poisoned_visible = int(item["node_labels"].sum().item())
            buckets[n_poisoned_visible].append(score)
            all_scores.append(score)
            all_labels.append(item["graph_label"])

    print("\nDetection rate (score >= 0.5) by number of poisoned records visible in-window:")
    print(f"  {'n_poisoned':>10s}  {'n_windows':>9s}  {'mean_score':>10s}  {'detect_rate':>11s}")
    results = {}
    for n_poisoned in sorted(buckets.keys()):
        scores = np.array(buckets[n_poisoned])
        mean_score = float(scores.mean())
        detect_rate = float((scores >= 0.5).mean())
        results[str(n_poisoned)] = {
            "n_windows": len(scores), "mean_score": mean_score, "detect_rate": detect_rate,
        }
        print(f"  {n_poisoned:>10d}  {len(scores):>9d}  {mean_score:>10.4f}  {detect_rate:>11.4f}")

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    overall_auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")

    print(f"\nOverall AUROC (this OOD set): {overall_auroc:.4f}  (n={len(y_true)}, pos={int(y_true.sum())})")

    early_scores, early_labels = [], []
    late_scores, late_labels = [], []
    for n_poisoned, scores in buckets.items():
        target = (early_scores, early_labels) if n_poisoned < 10 else (late_scores, late_labels)
        target[0].extend(scores)
        target[1].extend([1 if n_poisoned > 0 else 0] * len(scores))  # any poisoned record present -> label 1

    if len(set(early_labels)) > 1:
        early_auroc = roc_auc_score(early_labels, early_scores)
        print(f"Early-injection windows only (<10/20 poisoned records visible): "
              f"AUROC={early_auroc:.4f}, n={len(early_labels)}, mean_score={np.mean(early_scores):.4f}")
        results["_early_injection_auroc"] = float(early_auroc)
    else:
        print(f"Early-injection windows only (<10/20 poisoned records visible): "
              f"n={len(early_labels)}, mean_score={np.mean(early_scores) if early_scores else float('nan'):.4f} "
              f"(single class, AUROC undefined)")

    results["_overall_auroc"] = float(overall_auroc) if not np.isnan(overall_auroc) else None
    results["_n_windows"] = int(len(y_true))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
