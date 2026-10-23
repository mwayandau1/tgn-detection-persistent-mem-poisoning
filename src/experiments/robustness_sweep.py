"""
The four input-degradation sweeps of Section 4.9, on the full test partition (E8).
=================================================================================

ood_evaluator.py, which produced Tables 4.7 to 4.10, scored a random sample of 1,100 test
windows (a figure that survives only in a console log), with batch size 4 and shared memory
slots (see score_windows.py), under a single random draw per condition. Its five no-op
conditions disagree on FPR at 90% TPR by up to 0.035, which is larger than several of the
"effects" in those tables.

This keeps the perturbations exactly as GraphNoiseTransform defines them, so the curves stay
comparable, and changes how they are measured:

  * every test window, read from the verified cache;
  * per-window memory slots, so each window is scored as the architecture specifies;
  * three draws for every stochastic condition (seeds 0, 1, 2), reported with mean and SD;
    the threshold sweep is deterministic and is run once;
  * sequence-level AUROC (window max) beside the window-level metrics;
  * the sample, batch size, memory mode, draws and checkpoint written into the output.

What the perturbations do, stated because Section 4.9 describes the first one wrongly:
embedding noise changes only the 384 embedding columns and is redrawn independently at every
snapshot; edges, edge weights and the structural columns are NOT recomputed, so the similarity
graph is untouched. The threshold sweep rebuilds edges (>= theta) and refreshes the structural
columns. Edge loss drops each directed edge entry independently and refreshes the structural
columns. Metadata corruption replaces each of the three behavioural columns with a uniform
draw with the stated probability.

Usage:
    python -m src.experiments.robustness_sweep --data data/raw/ap_burst2k \
        --run-dir experiments/run_burst2k_base --output experiments/audit/robustness_full.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.cached_dataset import CachedTrajectoryDataset  # noqa: E402
from src.data_generation.noise_transforms import GraphNoiseTransform  # noqa: E402
from src.experiments.score_windows import build_model  # noqa: E402

SWEEPS = {
    "embedding_noise": ("embedding_sigma", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], True),
    "edge_threshold": ("threshold", [0.4, 0.5, 0.6, 0.7, 0.8], False),
    "metadata_corruption": ("meta_corrupt_prob", [0.0, 0.05, 0.10, 0.20, 0.30], True),
    "edge_loss": ("edge_drop_rate", [0.0, 0.05, 0.10, 0.20, 0.30], True),
}


@torch.no_grad()
def run_condition(model, ds, device, T, batch_size, **perturb):
    gs, ns, nl = [], [], []
    for b0 in range(0, len(ds), batch_size):
        items = [ds[i]["graphs"] for i in range(b0, min(b0 + batch_size, len(ds)))]
        items = [GraphNoiseTransform.perturb_trajectory([g.to(device) for g in w],
                                                         device=device, **perturb)
                 for w in items]
        graphs = [Batch.from_data_list([w[t] for w in items]) for t in range(T)]
        model.reset_memory()
        g, n = model(graphs)
        gs.extend(g.cpu().numpy().tolist())
        ns.extend(n.cpu().numpy().tolist())
        nl.extend(graphs[-1].node_labels.cpu().numpy().tolist())
    return np.asarray(gs), np.asarray(ns), np.asarray(nl)


def metrics(y, s, nl, ns, seq):
    fpr, tpr, _ = roc_curve(y, s)
    ids = sorted(set(seq))
    ys = np.array([y[seq == i].max() for i in ids])
    ss = np.array([s[seq == i].max() for i in ids])
    return {"graph_auroc": roc_auc_score(y, s), "graph_ap": average_precision_score(y, s),
            "graph_f1": f1_score(y, s >= 0.5), "fpr_at_tpr90": float(fpr[np.argmin(np.abs(tpr - 0.9))]),
            "node_auroc": roc_auc_score(nl, ns), "node_ap": average_precision_score(nl, ns),
            "sequence_auroc": roc_auc_score(ys, ss)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="phase2_best.pt")
    ap.add_argument("--model", default="tgn")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["node_meta_dim"] = 7
    cfg["model"]["readout"] = "attention"
    cfg["model"]["memory_slots"] = "per_window"
    T = cfg["training"]["temporal_window"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, cfg).to(device)
    model.load_state_dict(torch.load(Path(args.run_dir) / args.checkpoint, map_location=device,
                                     weights_only=True))
    model.eval()
    ds = CachedTrajectoryDataset(args.data, split="test", temporal_window=T,
                                 structural_features=True)
    y = np.array([int(ds.store.get(sid)["graph_label"][st + T - 1]) for sid, st in ds.windows])
    seq = np.array([sid for sid, _ in ds.windows])

    out = {"checkpoint": str(Path(args.run_dir) / args.checkpoint), "n_windows": len(ds),
           "n_sequences": int(len(set(seq))), "batch_size": args.batch_size,
           "memory_slots": "per_window", "draws": args.draws, "sweeps": {}}
    if Path(args.output).exists():
        out = json.load(open(args.output))
    clean = None
    for name in args.sweeps:
        key, values, stochastic = SWEEPS[name]
        out["sweeps"].setdefault(name, {})
        for v in values:
            if str(v) in out["sweeps"][name]:
                continue
            noop = (v == 0.0 and key != "threshold")
            n_draws = 1 if (noop or not stochastic) else args.draws
            runs = []
            for d in range(n_draws):
                if noop and clean is not None:
                    runs.append(clean)
                    continue
                torch.manual_seed(d)
                t0 = time.time()
                g, ns, nl = run_condition(model, ds, device, T, args.batch_size,
                                          **({} if noop else {key: v}))
                runs.append({k: float(x) for k, x in metrics(y, g, nl, ns, seq).items()})
                if noop:
                    clean = runs[-1]
                print(f"{name} {key}={v} draw {d}: graph AUROC {runs[-1]['graph_auroc']:.4f} "
                      f"seq {runs[-1]['sequence_auroc']:.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
            out["sweeps"][name][str(v)] = {
                "draws": runs,
                "mean": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
                "sd": {k: float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0
                       for k in runs[0]}}
            json.dump(out, open(args.output, "w"), indent=2)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
