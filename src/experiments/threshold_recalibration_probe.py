"""
Threshold Recalibration for the Temporal-vs-Static Single-Record Comparison (v2)
===================================================================================
External review finding B1: Table 4.6's single-record detection-rate comparison
(83.4% detector vs 38.6% static baseline) is made at a fixed threshold of 0.5,
which Sections 4.3, 5.3 and 6.4 argue is the wrong threshold operationally.

v1 of this script picked each model's best-F1 threshold on the validation
partition. That was flagged as methodologically unsound on a second pass, for
two reasons:

  1. The static model's validation AUROC is 1.0000 (perfect separation), which
     means F1 = 1.0 for EVERY threshold strictly between the highest-scoring
     clean window and the lowest-scoring poisoned window. "Best-F1 threshold"
     is therefore not a selected value there, it is an arbitrary point inside
     an indifference band, and the exact point returned depends on
     implementation detail rather than evidence.
  2. F1 weights false positives and false negatives equally. Section 2.6 of
     the thesis argues the opposite: a false negative permits an attack, a
     false positive costs an investigation. The defensible, well-determined
     rule for both models is a fixed operating point on the ROC curve: pick
     the threshold that holds validation clean-window FPR at a target rate
     (here 1%), which does not have an indifference-band problem even when
     one model achieves perfect separation.

This version:
  - Reports the full max-F1 indifference band for the static model (and the
    detector's, for symmetry) rather than a single point, and the
    single-record detection rate at both ends of that band.
  - Selects operating thresholds for both models at a fixed 1% validation
    FPR and re-reports single-record / extended-spread detection rates there.
  - Recomputes test-partition AUROC for both models directly (not from
    memory) alongside validation AUROC, to check for a val/test ordering
    flip.
  - Aggregates the single-record set's bucket-1 windows by sequence_id, since
    223 windows there come from only 25 independent sequences (~9 highly
    correlated windows per sequence), and reports both window-level and
    sequence-level detection rates.

One evaluation pass. No retraining. Both checkpoints are loaded as-is.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.tgn_detector import TGNDetector
from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StaticGNNDetector(nn.Module):
    """Verbatim architecture from static_gnn_with_node_loss.py / notebook 05 sec. 5."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        input_dim = model_cfg["node_feature_dim"] + 3
        gat_hidden = model_cfg["gat_hidden_dim"]
        gat_heads = model_cfg["gat_heads"]
        mlp_hidden = model_cfg["mlp_hidden_dim"]
        dropout = model_cfg["dropout"]

        self.gat1 = GATConv(input_dim, gat_hidden, heads=gat_heads, dropout=dropout, concat=True, edge_dim=1)
        self.gat2 = GATConv(gat_hidden * gat_heads, gat_hidden, heads=1, dropout=dropout, concat=False, edge_dim=1)
        self.graph_head = nn.Sequential(
            nn.Linear(gat_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(mlp_hidden, 1)
        )
        self.node_head = nn.Sequential(
            nn.Linear(gat_hidden, mlp_hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(mlp_hidden, 1)
        )
        self.dropout = dropout

    def forward(self, graph_sequence):
        g = graph_sequence[-1]
        x = g.x
        edge_index = g.edge_index
        batch_idx = g.batch
        edge_attr = g.edge_attr if hasattr(g, 'edge_attr') else None

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        node_emb = self.gat2(x, edge_index, edge_attr=edge_attr)

        graph_emb = global_mean_pool(node_emb, batch_idx)
        graph_scores = torch.sigmoid(self.graph_head(graph_emb)).squeeze(-1)
        node_scores = torch.sigmoid(self.node_head(node_emb)).squeeze(-1)
        return graph_scores, node_scores


def score_dataset(model, data_dir, temporal_window, split=None, sequence_ids=None):
    """Returns (scores, labels, n_poisoned_visible, sequence_ids) for every window."""
    if sequence_ids is None:
        with open(Path(data_dir) / "metadata.json") as f:
            metadata = json.load(f)
        sequence_ids = [m["sequence_id"] for m in metadata]

    kwargs = {"temporal_window": temporal_window}
    if split is not None:
        kwargs["split"] = split
    else:
        kwargs["sequence_ids"] = sequence_ids
    ds = MemoryTrajectoryDataset(data_dir, **kwargs)

    scores, labels, n_poisoned, seq_ids = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            batch = collate_trajectories([item])
            graphs = [g.to(device) for g in batch["graphs"]]
            graph_scores, _ = model(graphs)
            scores.append(graph_scores.squeeze().item())
            labels.append(item["graph_label"])
            n_poisoned.append(int(item["node_labels"].sum().item()))
            seq_ids.append(item["sequence_id"])
    return np.array(scores), np.array(labels), np.array(n_poisoned), np.array(seq_ids)


def max_f1_band(y_true, y_score):
    """Full range of thresholds achieving the maximum validation F1, not just one point.
    For perfectly-separable data (AUROC=1.0) this is computed exactly as the open-closed
    interval (max negative score, min positive score], since every threshold in that gap
    misclassifies nothing. For non-separable data it falls back to the tied thresholds in
    sklearn's precision_recall_curve output."""
    neg = y_score[y_true == 0]
    pos = y_score[y_true == 1]
    if len(neg) > 0 and len(pos) > 0 and neg.max() < pos.min():
        low = float(neg.max())
        high = float(pos.min())
        return low, high, 1.0, True

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    f1_valid = f1[:-1]
    if len(thresholds) == 0:
        return 0.5, 0.5, float(f1[-1]), False
    max_f1 = f1_valid.max()
    band_idx = np.where(f1_valid >= max_f1 - 1e-9)[0]
    return float(thresholds[band_idx.min()]), float(thresholds[band_idx.max()]), float(max_f1), False


def threshold_at_fpr(y_true, y_score, target_fpr=0.01):
    """The LOWEST threshold satisfying FPR <= target_fpr, computed explicitly from the
    correct negative count (not sklearn's discretised roc_curve, whose candidate
    thresholds are spaced by whatever score happens to be observed next and can
    overshoot the true tightest bound when there is a gap between classes).

    allowed_fp = floor(target_fpr * n_neg) false positives are tolerated. The lowest
    threshold admitting at most that many negatives is set just above the
    (allowed_fp+1)-th highest negative score, which maximises sensitivity subject to
    the constraint - the correct direction, since a lower threshold is weakly more
    sensitive on any out-of-sample data.
    """
    neg = np.sort(y_score[y_true == 0])[::-1]
    n_neg = len(neg)
    allowed_fp = int(np.floor(target_fpr * n_neg))

    if allowed_fp >= n_neg:
        t = float(neg.min()) - 1e-9 if n_neg > 0 else -1.0
    elif allowed_fp == 0:
        t = float(neg[0]) + 1e-9
    else:
        t = float(neg[allowed_fp]) + 1e-9

    achieved_fp = int((y_score[y_true == 0] >= t).sum())
    achieved_fpr = achieved_fp / n_neg if n_neg > 0 else 0.0
    n_pos = int((y_true == 1).sum())
    achieved_tp = int((y_score[y_true == 1] >= t).sum())
    achieved_tpr = achieved_tp / n_pos if n_pos > 0 else 0.0
    return t, achieved_fpr, achieved_tpr, n_neg, allowed_fp


def mcnemar_exact(b, c):
    """Exact McNemar sign test on discordant pairs. b = pairs where model A detects and
    B does not, c = the reverse. Two-sided exact p-value via the binomial distribution."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def sequence_fully_detected_set(scores, n_poisoned, seq_ids, threshold):
    """Returns {sequence_id: bool} - whether every bucket-1 window in that sequence
    scores >= threshold."""
    mask = n_poisoned == 1
    per_seq = defaultdict(list)
    for score, seq in zip(scores[mask], seq_ids[mask]):
        per_seq[seq].append(score)
    return {seq: bool(np.all(np.array(v) >= threshold)) for seq, v in per_seq.items()}


def bucket_rates(scores, n_poisoned, thresholds: dict):
    """thresholds: {name: value}. Returns {bucket: {name: rate, ...}} plus mean_score/n_windows."""
    buckets = defaultdict(list)
    for s, n in zip(scores, n_poisoned):
        buckets[n].append(s)
    out = {}
    for n_val in sorted(buckets.keys()):
        arr = np.array(buckets[n_val])
        row = {"n_windows": len(arr), "mean_score": float(arr.mean())}
        for name, t in thresholds.items():
            row[f"rate@{name}"] = float((arr >= t).mean())
        out[str(n_val)] = row
    return out


def sequence_level_bucket1(scores, n_poisoned, seq_ids, thresholds: dict):
    """For bucket n_poisoned==1 windows only: group by sequence_id and report the
    per-sequence detection rate distribution, at each named threshold."""
    mask = n_poisoned == 1
    s, sid = scores[mask], seq_ids[mask]
    per_seq = defaultdict(list)
    for score, seq in zip(s, sid):
        per_seq[seq].append(score)

    result = {"n_sequences": len(per_seq), "n_windows": int(mask.sum())}
    for name, t in thresholds.items():
        seq_rates = np.array([float((np.array(v) >= t).mean()) for v in per_seq.values()])
        result[name] = {
            "mean_sequence_detect_rate": float(seq_rates.mean()),
            "n_sequences_fully_detected": int((seq_rates == 1.0).sum()),
            "n_sequences_zero_detected": int((seq_rates == 0.0).sum()),
            "n_sequences_partial": int(((seq_rates > 0) & (seq_rates < 1.0)).sum()),
        }
    return result


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    T = config["training"]["temporal_window"]
    print(f"Using device: {device}", flush=True)

    print("\n=== Loading models ===")
    detector = TGNDetector(config).to(device)
    detector.load_state_dict(torch.load("experiments/run_current_arch/phase2_best.pt",
                                         map_location=device, weights_only=True))
    static = StaticGNNDetector(config).to(device)
    static.load_state_dict(torch.load("experiments/static_gnn_node_trained.pt",
                                       map_location=device, weights_only=True))

    print("\n=== Step 1: scoring validation + test partitions ===")
    det_val_s, det_val_y, _, _ = score_dataset(detector, "data/raw/llm_generated", T, split="val")
    static_val_s, static_val_y, _, _ = score_dataset(static, "data/raw/llm_generated", T, split="val")
    det_test_s, det_test_y, _, _ = score_dataset(detector, "data/raw/llm_generated", T, split="test")
    static_test_s, static_test_y, _, _ = score_dataset(static, "data/raw/llm_generated", T, split="test")

    det_val_auroc = roc_auc_score(det_val_y, det_val_s)
    static_val_auroc = roc_auc_score(static_val_y, static_val_s)
    det_test_auroc = roc_auc_score(det_test_y, det_test_s)
    static_test_auroc = roc_auc_score(static_test_y, static_test_s)

    print(f"Detector : val AUROC={det_val_auroc:.4f}   test AUROC={det_test_auroc:.4f}")
    print(f"Static   : val AUROC={static_val_auroc:.4f}   test AUROC={static_test_auroc:.4f}")
    if (det_val_auroc > static_val_auroc) != (det_test_auroc > static_test_auroc):
        print(">>> ORDERING FLIP: which model ranks higher reverses between validation and test.")

    print("\n=== Step 2: max-F1 indifference band (diagnostic, not used as the operating rule) ===")
    det_band_lo, det_band_hi, det_band_f1, det_perfect = max_f1_band(det_val_y, det_val_s)
    static_band_lo, static_band_hi, static_band_f1, static_perfect = max_f1_band(static_val_y, static_val_s)
    print(f"Detector : band=({det_band_lo:.6f}, {det_band_hi:.6f}]  max_F1={det_band_f1:.4f}  perfect_separation={det_perfect}")
    print(f"Static   : band=({static_band_lo:.6f}, {static_band_hi:.6f}]  max_F1={static_band_f1:.4f}  perfect_separation={static_perfect}")

    print("\n=== Step 3: fixed-FPR operating thresholds (lowest threshold satisfying FPR <= target) ===")
    TARGET_FPR = 0.01
    det_t_fpr, det_achieved_fpr, det_achieved_tpr, det_n_neg, det_allowed_fp = threshold_at_fpr(det_val_y, det_val_s, TARGET_FPR)
    static_t_fpr, static_achieved_fpr, static_achieved_tpr, static_n_neg, static_allowed_fp = threshold_at_fpr(static_val_y, static_val_s, TARGET_FPR)
    print(f"Validation negative-window count: detector n_neg={det_n_neg}  static n_neg={static_n_neg}  "
          f"(target {TARGET_FPR:.0%} -> allowed_fp: detector={det_allowed_fp}, static={static_allowed_fp})")
    print(f"Detector : threshold={det_t_fpr:.6f}  achieved_fpr={det_achieved_fpr:.4f}  achieved_tpr(val)={det_achieved_tpr:.4f}")
    print(f"Static   : threshold={static_t_fpr:.6f}  achieved_fpr={static_achieved_fpr:.4f}  achieved_tpr(val)={static_achieved_tpr:.4f}")

    out = {
        "validation_test_auroc": {
            "detector": {"val_auroc": float(det_val_auroc), "test_auroc": float(det_test_auroc)},
            "static": {"val_auroc": float(static_val_auroc), "test_auroc": float(static_test_auroc)},
            "ordering_flip": bool((det_val_auroc > static_val_auroc) != (det_test_auroc > static_test_auroc)),
        },
        "max_f1_indifference_band": {
            "detector": {"low": det_band_lo, "high": det_band_hi, "max_f1": det_band_f1, "perfect_separation": det_perfect},
            "static": {"low": static_band_lo, "high": static_band_hi, "max_f1": static_band_f1, "perfect_separation": static_perfect},
        },
        "fixed_fpr_threshold": {
            "target_fpr": TARGET_FPR,
            "detector": {"threshold": det_t_fpr, "achieved_fpr": det_achieved_fpr, "achieved_tpr_val": det_achieved_tpr,
                         "n_neg_windows": det_n_neg, "allowed_fp": det_allowed_fp},
            "static": {"threshold": static_t_fpr, "achieved_fpr": static_achieved_fpr, "achieved_tpr_val": static_achieved_tpr,
                       "n_neg_windows": static_n_neg, "allowed_fp": static_allowed_fp},
        },
        "single_record_set": {},
        "extended_spread_set": {},
    }

    print("\n=== Step 4: re-scoring OOD probe sets ===")
    single_record_data = {}
    for probe_name, probe_dir in [("single_record_set", "data/raw/llm_generated_single_ood"),
                                   ("extended_spread_set", "data/raw/llm_generated_slow_ood")]:
        det_scores, det_labels, det_npois, det_seqids = score_dataset(detector, probe_dir, T)
        static_scores, static_labels, static_npois, static_seqids = score_dataset(static, probe_dir, T)

        det_thresholds = {"0.5": 0.5, "band_low": det_band_lo, "band_high": det_band_hi, "fpr1pct": det_t_fpr}
        static_thresholds = {"0.5": 0.5, "band_low": static_band_lo, "band_high": static_band_hi, "fpr1pct": static_t_fpr}

        det_res = bucket_rates(det_scores, det_npois, det_thresholds)
        static_res = bucket_rates(static_scores, static_npois, static_thresholds)
        out[probe_name] = {"detector": det_res, "static": static_res}

        print(f"\n[{probe_name}] detector bucket n=1:  {det_res.get('1', {})}")
        print(f"[{probe_name}] static   bucket n=1:  {static_res.get('1', {})}")

        if probe_name == "single_record_set":
            single_record_data = {
                "det_scores": det_scores, "det_npois": det_npois, "det_seqids": det_seqids,
                "static_scores": static_scores, "static_npois": static_npois, "static_seqids": static_seqids,
            }

    print("\n=== Step 5: full-band sweep, single-record set, window-level and sequence-level (with McNemar) ===")
    d = single_record_data
    # Sweep across the union of the corrected-budget threshold and the static model's
    # max-F1 band, plus a few intermediate points, so the reported rate is a curve not
    # a single cherry-picked value.
    sweep_lo = min(static_t_fpr, static_band_lo)
    sweep_hi = static_band_hi
    sweep_points = sorted(set([sweep_lo, static_band_lo, (sweep_lo + sweep_hi) / 2, static_band_hi]))

    band_sweep = []
    print(f"{'threshold':>12s}  {'det_window':>10s}  {'stat_window':>11s}  {'det_seq':>9s}  {'stat_seq':>9s}  "
          f"{'b(det-only)':>11s}  {'c(stat-only)':>12s}  {'McNemar_p':>10s}")
    for t in sweep_points:
        det_window_rate = float((d["det_scores"][d["det_npois"] == 1] >= det_t_fpr).mean())
        stat_window_rate = float((d["static_scores"][d["static_npois"] == 1] >= t).mean())

        det_seq_map = sequence_fully_detected_set(d["det_scores"], d["det_npois"], d["det_seqids"], det_t_fpr)
        static_seq_map = sequence_fully_detected_set(d["static_scores"], d["static_npois"], d["static_seqids"], t)

        common_seqs = sorted(set(det_seq_map) & set(static_seq_map))
        b = sum(1 for s in common_seqs if det_seq_map[s] and not static_seq_map[s])
        c = sum(1 for s in common_seqs if static_seq_map[s] and not det_seq_map[s])
        p = mcnemar_exact(b, c)

        det_seq_count = sum(det_seq_map.values())
        static_seq_count = sum(static_seq_map.values())

        print(f"{t:>12.6f}  {det_window_rate:>10.4f}  {stat_window_rate:>11.4f}  "
              f"{det_seq_count:>6d}/{len(common_seqs):<2d}  {static_seq_count:>6d}/{len(common_seqs):<2d}  "
              f"{b:>11d}  {c:>12d}  {p:>10.4f}")

        band_sweep.append({
            "threshold": float(t), "static_window_rate": stat_window_rate, "detector_window_rate": det_window_rate,
            "static_seq_fully_detected": static_seq_count, "detector_seq_fully_detected": det_seq_count,
            "n_sequences": len(common_seqs), "discordant_b_detector_only": b, "discordant_c_static_only": c,
            "mcnemar_p": p,
        })

    out["single_record_band_sweep"] = band_sweep

    out_path = Path("experiments/threshold_recalibration_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n=== Headline ===")
    print(f"Corrected FPR-budget threshold (n_neg={static_n_neg}, allowed_fp={static_allowed_fp}):")
    row = band_sweep[0]
    print(f"  Static window-level rate={row['static_window_rate']:.4f}  "
          f"sequence-level={row['static_seq_fully_detected']}/{row['n_sequences']}  "
          f"vs detector window={row['detector_window_rate']:.4f}  sequence={row['detector_seq_fully_detected']}/{row['n_sequences']}  "
          f"McNemar p={row['mcnemar_p']:.4f}")


if __name__ == "__main__":
    main()
