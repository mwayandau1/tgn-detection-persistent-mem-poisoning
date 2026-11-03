"""
Score every window of one split with a trained checkpoint, at a stated batch size.
=================================================================================

Why the batch size has to be stated
-----------------------------------
The node-memory bank in TGNDetector is addressed by `node_ids`, and the dataset sets
`node_ids = arange(num_nodes)` per snapshot. `Batch.from_data_list` offsets only attributes
whose name contains "index" (edge_index and the like), so `node_ids` is concatenated WITHOUT an
offset: with batch_size > 1, node i of every window in the batch reads and writes the same
memory slot. The architecture as specified (Section 3.8: "the bank was cleared at the start of
each window, so that no state was carried between windows") holds only at batch_size 1.

With shuffle=False the windows in a batch are consecutive windows of the same sequence, so the
shared slot can hold state written by a window that ends one to three snapshots LATER. That is
look-ahead within a sequence, and across a batch boundary it is state from another sequence.

Every evaluation script in this directory inherited the training batch size of 4, so this
tool exists to measure what that does: score the same checkpoint at batch 1 and at batch 4
and compare. It also replaces the per-window file loading of MemoryTrajectoryDataset with a
per-sequence cache, which is what makes batch-1 scoring of a whole split affordable on the
4 GB card.

Output (npz): seq, start, y, score for every window in dataset order; and, with --node-scores,
node_window (index into the window arrays), node_score and node_label for the final snapshot
of every window.

Usage:
    python -m src.experiments.score_windows --data data/raw/ap_burst2k \
        --run-dir experiments/run_burst2k_base --split test --batch-size 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.dataset import (COHERENCE_DIM, MemoryTrajectoryDataset,  # noqa: E402
                                         with_arrival_edges, with_coherence_features)
from src.models.tgn_detector import TGNDetector  # noqa: E402


def build_model(kind: str, cfg: dict):
    """Instantiate the architecture a checkpoint was trained as."""
    if kind == "tgn":
        return TGNDetector(cfg)
    if kind == "meanpool":
        from src.experiments.ablation_meanpool_model import TGNDetectorMeanPool
        return TGNDetectorMeanPool(cfg)
    if kind in ("static_attn", "static_mean"):
        from src.experiments.static_gnn_model import (StaticGNNDetectorAttention,
                                                      StaticGNNDetectorMeanPool)
        return (StaticGNNDetectorAttention if kind == "static_attn"
                else StaticGNNDetectorMeanPool)(cfg)
    from src.experiments.component_ablation_models import MODEL_KINDS
    if kind in MODEL_KINDS:
        return MODEL_KINDS[kind](cfg)
    raise ValueError(f"unknown model kind {kind!r}")


class SnapshotCache:
    """All snapshots of the most recently used sequences, loaded once each."""

    def __init__(self, ds: MemoryTrajectoryDataset, keep: int = 3):
        self.ds, self.keep, self.store = ds, keep, {}

    def get(self, seq_id: str):
        if seq_id not in self.store:
            if len(self.store) >= self.keep:
                self.store.pop(next(iter(self.store)))
            if hasattr(self.ds, "store"):          # CachedTrajectoryDataset
                n = len(self.ds.store.get(seq_id)["n_nodes"])
                self.store[seq_id] = [self.ds.store.snapshot(seq_id, t) for t in range(n)]
            else:
                paths = sorted((self.ds.data_dir / seq_id).glob("snapshot_*.npz"))
                self.store[seq_id] = [self.ds._load_snapshot(p) for p in paths]
            # Once per loaded sequence, as the dataset does once per window: the snapshots
            # are reused across windows, so adding them per window would duplicate edges.
            if getattr(self.ds, "arrival_window", 0) > 0:
                self.store[seq_id] = [with_arrival_edges(g, self.ds.embedding_dim,
                                                         self.ds.arrival_window)
                                      for g in self.store[seq_id]]
            if getattr(self.ds, "coherence_features", False):
                self.store[seq_id] = [with_coherence_features(g, self.ds.embedding_dim)
                                      for g in self.store[seq_id]]
        return self.store[seq_id]


@torch.no_grad()
def score_split(model, ds: MemoryTrajectoryDataset, device, batch_size: int, T: int,
                node_scores: bool = False, log_every: int = 500):
    cache = SnapshotCache(ds, keep=batch_size + 2)
    windows = ds.windows
    scores, labels = [], []
    nw, ns, nl = [], [], []
    collisions = 0
    t0 = time.time()
    model.eval()
    for b0 in range(0, len(windows), batch_size):
        chunk = windows[b0:b0 + batch_size]
        per_window = [cache.get(sid)[start:start + T] for sid, start in chunk]
        graphs = [Batch.from_data_list([w[t] for w in per_window]).to(device) for t in range(T)]
        if batch_size > 1 and b0 == 0 and hasattr(model, "_slot_ids"):
            ids = model._slot_ids(graphs[-1], model._memory_stride(graphs))
            collisions = int(ids.numel() - torch.unique(ids).numel())
        model.reset_memory()
        g, n = model(graphs)
        scores.extend(g.cpu().numpy().tolist())
        labels.extend(int(w[-1].y.item()) for w in per_window)
        if node_scores:
            last = graphs[-1]
            bidx = last.batch.cpu().numpy()
            ns.extend(n.cpu().numpy().tolist())
            nl.extend(last.node_labels.cpu().numpy().tolist())
            nw.extend((b0 + bidx).tolist())
        if log_every and (b0 // batch_size) % max(1, log_every // batch_size) == 0 and b0:
            rate = b0 / (time.time() - t0)
            print(f"  {b0}/{len(windows)} windows  ({rate:.1f}/s)", flush=True)
    out = {"seq": np.array([w[0] for w in windows]),
           "start": np.array([w[1] for w in windows], dtype=int),
           "y": np.array(labels, dtype=int),
           "score": np.array(scores, dtype=float)}
    if node_scores:
        out.update(node_window=np.array(nw, dtype=int), node_score=np.array(ns, dtype=float),
                   node_label=np.array(nl, dtype=int))
    return out, collisions, time.time() - t0


# The frozen windows were 10 snapshots long and so ended at snapshots 9 to 19.
ONLINE_FIRST_END = 9


@torch.no_grad()
def score_online(model, ds: MemoryTrajectoryDataset, device, first_end: int = ONLINE_FIRST_END,
                 node_scores: bool = False):
    """Score every sequence as it would be scored in deployment: the model run on snapshots
    0 to e, read after snapshot e, for every e from first_end to the last snapshot.

    For a model trained on whole sequences (one window of 20) this is what a time axis means:
    a score after each snapshot, from memory that has seen everything so far. With first_end 9
    the end points and labels are those of the frozen 10-snapshot windows, so the online
    scores stand where the frozen window scores stood (Table 4.1's window column, latency,
    Figure 4.3). The score at the last snapshot is the whole-sequence score. Batch size 1.

    Output as score_split's, plus `end` (the snapshot each score is read after); `start` is 0.
    """
    cache = SnapshotCache(ds, keep=2)
    seqs = list(dict.fromkeys(sid for sid, _ in ds.windows))
    seq_col, ends, scores, labels = [], [], [], []
    nw, ns, nl = [], [], []
    t0 = time.time()
    model.eval()
    for sid in seqs:
        snaps = cache.get(sid)
        batched = [Batch.from_data_list([g]).to(device) for g in snaps]
        for e in range(first_end, len(snaps)):
            model.reset_memory()
            g, n = model(batched[:e + 1])
            if node_scores:
                nw.extend([len(scores)] * int(n.numel()))
                ns.extend(n.cpu().numpy().tolist())
                nl.extend(snaps[e].node_labels.numpy().tolist())
            seq_col.append(sid)
            ends.append(e)
            scores.append(float(g.item()))
            labels.append(int(snaps[e].y.item()))
    out = {"seq": np.array(seq_col), "start": np.zeros(len(ends), dtype=int),
           "end": np.array(ends, dtype=int), "y": np.array(labels, dtype=int),
           "score": np.array(scores, dtype=float)}
    if node_scores:
        out.update(node_window=np.array(nw, dtype=int), node_score=np.array(ns, dtype=float),
                   node_label=np.array(nl, dtype=int))
    return out, time.time() - t0


def variant_flags(run_dir) -> list:
    """score_windows flags for the inputs a run was trained with, from its variant.json.
    Runs from before these settings existed have none of the keys and get the defaults."""
    p = Path(run_dir) / "variant.json"
    v = json.load(open(p)) if p.exists() else {}
    flags = ["--arrival-window", str(v.get("arrival_window") or 0)]
    if v.get("temporal_window"):
        flags += ["--temporal-window", str(v["temporal_window"])]
    if v.get("coherence_features"):
        flags += ["--coherence-features"]
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="phase2_best.pt")
    ap.add_argument("--model", default="tgn",
                    help="tgn | meanpool | static_attn | static_mean | a kind from "
                         "component_ablation_models.MODEL_KINDS")
    ap.add_argument("--readout", default="attention", choices=["attention", "scale", "gated"])
    ap.add_argument("--no-structural-features", dest="structural_features",
                    action="store_false", default=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--node-scores", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cached", action="store_true",
                    help="Read from the verified per-sequence cache (bit-identical tensors).")
    ap.add_argument("--memory-slots", choices=["shared", "per_window"], default="per_window",
                    help="Bank addressing; 'shared' is the legacy setting. Irrelevant at batch "
                         "size 1, where both agree.")
    ap.add_argument("--arrival-window", type=int, default=0,
                    help="The value the checkpoint was trained with (its variant.json).")
    ap.add_argument("--temporal-window", type=int, default=None,
                    help="The value the checkpoint was trained with (its variant.json); "
                         "config default if omitted.")
    ap.add_argument("--coherence-features", action="store_true",
                    help="Set if the checkpoint was trained with them (its variant.json).")
    ap.add_argument("--online", action="store_true",
                    help="Online scores (score_online) instead of window scores, written to "
                         "online_scores_<split>_b1.npz. Batch size 1.")
    ap.add_argument("--no-metrics", action="store_true",
                    help="Write the scores without printing AUROC or AP, for a split whose "
                         "figures are to be read only in the summaries.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    T = args.temporal_window or cfg["training"]["temporal_window"]
    cfg["model"]["node_meta_dim"] = ((3 + 4 if args.structural_features else 3)
                                     + (COHERENCE_DIM if args.coherence_features else 0))
    cfg["model"]["readout"] = args.readout
    cfg["model"]["scale_readout"] = args.readout == "scale"
    cfg["model"]["memory_slots"] = args.memory_slots
    cfg["model"]["edge_dim"] = 2 if args.arrival_window > 0 else 1
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = build_model(args.model, cfg).to(device)
    model.load_state_dict(torch.load(Path(args.run_dir) / args.checkpoint, map_location=device,
                                     weights_only=True))
    ds_cls = MemoryTrajectoryDataset
    if args.cached:
        from src.data_generation.cached_dataset import CachedTrajectoryDataset as ds_cls
    ds = ds_cls(args.data, split=args.split, temporal_window=T,
                                 embedding_dim=cfg["simulation"]["embedding_dim"],
                                 structural_features=args.structural_features,
                                 arrival_window=args.arrival_window,
                                 coherence_features=args.coherence_features)
    print(f"{args.run_dir} | {args.model} | {args.split}: {len(ds.windows)} windows | "
          f"batch {args.batch_size} | memory {args.memory_slots} | {device}")

    if args.online:
        if args.batch_size != 1:
            raise SystemExit("--online scores at batch size 1 only")
        out, secs = score_online(model, ds, device, node_scores=args.node_scores)
    else:
        out, collisions, secs = score_split(model, ds, device, args.batch_size, T,
                                            node_scores=args.node_scores)
        if args.batch_size > 1:
            print(f"  memory slots shared between windows in the first batch: {collisions}")
    kind = "online_scores" if args.online else "window_scores"
    path = Path(args.output) if args.output else (
        Path(args.run_dir) / f"{kind}_{args.split}_b{args.batch_size}.npz")
    np.savez(path, **out, batch_size=args.batch_size, checkpoint=args.checkpoint,
             model=args.model, device=str(device), memory_slots=args.memory_slots)

    if args.no_metrics:
        print(f"saved {path} ({secs:.0f}s; metrics not printed)")
        return
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, s = out["y"], out["score"]
    summary = {"windows": int(len(y)), "positives": int(y.sum()), "seconds": round(secs, 1),
               "window_auroc": float(roc_auc_score(y, s)),
               "window_ap": float(average_precision_score(y, s))}
    if args.node_scores:
        summary["node_auroc"] = float(roc_auc_score(out["node_label"], out["node_score"]))
        summary["node_ap"] = float(average_precision_score(out["node_label"], out["node_score"]))
    print(json.dumps(summary, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
