"""
Sequence-level evaluation: the comparison that cannot be contaminated by window position.
=========================================================================================

Every window-level AUROC in this project sits on an unearned floor. Window labels are
presence-based, so a window turns positive only once the injection has appeared, and
injections start part-way through a sequence; meanwhile node count is an exact linear
function of snapshot index. The window's position therefore predicts its label while
describing no structure at all -- measured at 0.609 on ap_burst2k, 0.586 on ap_sequences.

At sequence level that floor disappears. Every sequence has the same length and contributes
exactly one observation, so position carries no information by construction. It is also the
operationally meaningful question: an operator asks "is this agent's memory poisoned?", not
"is window 14 of this agent's memory poisoned?".

Three baselines, so a T-GNN result can be placed against something:

  final-state features   what a single-snapshot detector can see: node count, edge count,
                         density, max degree, tightest-cluster size and mean similarity.
                         On a benchmark with the final state matched these should be at
                         chance, and that is the check that the matching held.
  temporal features      growth-rate summaries across snapshots: the largest jump in cluster
                         density between consecutive snapshots, its ratio to the final
                         density, and the standard deviation of growth. These are hand-built
                         for the known generative process -- an oracle in the sense that they
                         encode what the attack IS, so they bound what a model could find.
  all features           both sets together, fitted jointly.

The T-GNN produces one score per window, so it needs an aggregator to reach sequence level.
Three are reported rather than one, because the choice is a real degree of freedom and
picking the best after the fact would be selection on the test set: max (does any window look
poisoned), mean (does the sequence look poisoned on average) and last (the operational
online case -- decide from the most recent window only).

Intervals are cluster bootstrap over sequences. At sequence level each sequence is already
one observation, so this is an ordinary bootstrap; it is kept in the same form as the
window-level code so the two are directly comparable.

Usage:
    python -m src.experiments.sequence_eval --data data/raw/ap_burst2k \
        --run-dir experiments/run_burst2k_base --label baseline
    python -m src.experiments.sequence_eval --data data/raw/ap_burst2k --skip-model
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# The split between these two lists is the whole point of the comparison, so each feature
# has to be classified by what it READS, not by what it is named after. `final_clustersize`
# was originally in the final-state list and is not a final-state feature: it counts how many
# snapshots shared the final snapshot's max degree, which reads the entire trajectory. A burst
# attack plants its cluster abruptly and then sits flat, so many snapshots match; a gradually
# grown benign cluster keeps climbing, so few do. It scores 0.719 at sequence level -- the
# strongest single feature here -- and leaving it mislabelled inflated the "final-state"
# baseline from 0.65 to 0.70 with a temporal signal. Renamed to say what it measures.
FINAL_FEATS = ["n_nodes", "n_edges", "density", "final_maxdeg",
               "final_tightness", "final_meansim"]
TEMPORAL_FEATS = ["snapshots_at_peak_maxdeg", "max_jump_maxdeg", "max_jump_density",
                  "jump_ratio", "growth_std", "max_jump_tightness", "time_to_peak_jump"]


def sequence_features(data: Path, seq_ids, threshold, emb_dim):
    """One row per sequence: final-snapshot statistics and growth-rate summaries."""
    rows = []
    for sid in seq_ids:
        snaps = sorted((data / sid).glob("snapshot_*.npz"))
        if not snaps:
            continue
        sizes, dens, tight, msim = [], [], [], []
        n_nodes = n_edges = 0
        for p in snaps:
            with np.load(p) as z:
                x = z["node_features"][:, :emb_dim]
            u = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            sim = u @ u.T
            np.fill_diagonal(sim, 0.0)
            adj = sim > threshold
            deg = adj.sum(axis=1)
            n_nodes, n_edges = x.shape[0], int(adj.sum()) // 2
            sizes.append(float(deg.max()) if deg.size else 0.0)
            dens.append(float(adj.sum()) / 2.0)
            above = sim[adj]
            msim.append(float(above.mean()) if above.size else 0.0)
            if deg.size and deg.max() > 0:
                hub = int(np.argmax(deg))
                nb = np.flatnonzero(adj[hub])
                tight.append(float(sim[np.ix_(nb, nb)].mean()) if nb.size > 1 else 0.0)
            else:
                tight.append(0.0)
        sizes, dens = np.array(sizes), np.array(dens)
        tight, msim = np.array(tight), np.array(msim)
        d_size, d_dens, d_tight = np.diff(sizes), np.diff(dens), np.diff(tight)
        rows.append({
            "sequence_id": sid,
            "n_nodes": float(n_nodes),
            "n_edges": float(n_edges),
            "density": float(n_edges) / max(n_nodes * (n_nodes - 1) / 2, 1),
            "final_maxdeg": sizes[-1],
            # Temporal, despite reading the final snapshot's value: how many snapshots
            # shared the final max degree, i.e. how long the cluster has been at full size.
            "snapshots_at_peak_maxdeg": float((np.asarray(sizes) == sizes[-1]).sum()),
            "final_tightness": tight[-1],
            "final_meansim": msim[-1],
            "max_jump_maxdeg": float(d_size.max()) if d_size.size else 0.0,
            "max_jump_density": float(d_dens.max()) if d_dens.size else 0.0,
            "jump_ratio": float(d_dens.max() / (dens[-1] + 1e-6)) if d_dens.size else 0.0,
            "growth_std": float(d_dens.std()) if d_dens.size else 0.0,
            "max_jump_tightness": float(d_tight.max()) if d_tight.size else 0.0,
            # Where the sharpest growth happened, normalised. An attack burst has a single
            # sharp jump; a gradual benign cluster has no distinguished moment.
            "time_to_peak_jump": (float(np.argmax(d_dens)) / max(len(d_dens), 1)
                                  if d_dens.size else 0.0),
        })
    return rows


def to_matrix(rows, feats):
    return np.array([[r[f] for f in feats] for r in rows], float)


def boot_ci(y, score, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], score[idx]))
    a = np.array(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def paired_boot(y, a, b, n_boot=2000, seed=0, metric="auroc"):
    """Paired bootstrap on a difference in AUROC or in average precision.

    Both are reported because they answer different questions and, measured here, they
    disagree: the baseline T-GNN ties the 13-feature oracle on AUROC (+0.010) while leading
    it on AP (0.818 against 0.720). AUROC integrates over the whole ranking and is dominated
    by the bulk of easy negatives; AP weights the top of the ranking, which is where a
    detector is actually operated -- an analyst reads the highest-scoring sequences, not the
    median one. A gap that appears in AP and not in AUROC means the model orders its most
    confident calls better while ranking the tail no better, which is worth stating precisely
    rather than collapsing into one number.
    """
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint", default="phase2_best.pt")
    ap.add_argument("--readout", default="attention",
                    choices=["attention", "scale", "gated"])
    ap.add_argument("--no-structural-features", dest="structural_features",
                    action="store_false", default=True,
                    help="Evaluate a checkpoint trained WITHOUT the 4 structural node "
                         "columns. This must match how the checkpoint was trained: the "
                         "model's input width depends on it, and a mismatch loads the "
                         "weights into a differently-shaped model and reports a number that "
                         "means nothing.")
    ap.add_argument("--label", default="T-GNN")
    ap.add_argument("--skip-model", action="store_true",
                    help="Baselines only -- runs with no GPU, so it can be done while a "
                         "training job holds the card.")
    ap.add_argument("--window-scores", default=None,
                    help="Score from a precomputed window_scores_test.npz instead of "
                         "building a TGNDetector. This is how the static-GNN baseline is "
                         "evaluated: the aggregation, the bootstrap and the paired test are "
                         "then literally the same code for both models, so a difference "
                         "between them cannot come from the evaluation.")
    ap.add_argument("--cache", default=None,
                    help="Where to cache the per-sequence feature table; recomputing the "
                         "similarity matrices for every snapshot is the expensive part.")
    ap.add_argument("--memory-slots", choices=["shared", "per_window"], default="shared",
                    help="Node-memory bank addressing for batched windows (see "
                         "TGNDetector.memory_slots). 'per_window' at any batch size equals "
                         "batch size 1; 'shared' reproduces the run_burst2k_* figures.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Scoring batch size; defaults to the training batch size (4).")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    threshold = cfg["simulation"]["semantic_edge_threshold"]
    emb_dim = cfg["simulation"]["embedding_dim"]
    T = cfg["training"]["temporal_window"]

    data = Path(args.data)
    meta = {m["sequence_id"]: m for m in json.load(open(data / "metadata.json"))}
    splits = json.load(open(data / "splits.json"))

    cache_path = Path(args.cache) if args.cache else data / "sequence_features.json"
    if cache_path.exists():
        cached = json.load(open(cache_path))
        print(f"Loaded cached sequence features from {cache_path}")
    else:
        cached = {}
        for name in ("train", "val", "test"):
            print(f"  computing {name} sequence features ({len(splits[name])} sequences)...")
            cached[name] = sequence_features(data, splits[name], threshold, emb_dim)
        json.dump(cached, open(cache_path, "w"))
        print(f"Cached to {cache_path}")

    ytr = np.array([int(meta[r["sequence_id"]]["label"] == "poisoned") for r in cached["train"]])
    yte = np.array([int(meta[r["sequence_id"]]["label"] == "poisoned") for r in cached["test"]])
    print(f"\n{data.name}: {len(ytr)} train sequences ({ytr.sum()} poisoned), "
          f"{len(yte)} test sequences ({yte.sum()} poisoned)")

    results = {"n_train": int(len(ytr)), "n_test": int(len(yte)),
               "n_test_poisoned": int(yte.sum()), "models": {}}

    print(f"\n  {'model':<34} {'AUROC':>7} {'AP':>7}   {'95% CI':>18}")
    print("  " + "-" * 72)

    scores = {}
    for name, feats in (("final-state features (6)", FINAL_FEATS),
                        ("temporal features (7)", TEMPORAL_FEATS),
                        ("all features (13)", FINAL_FEATS + TEMPORAL_FEATS)):
        Xtr, Xte = to_matrix(cached["train"], feats), to_matrix(cached["test"], feats)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
            sc.transform(Xtr), ytr)
        s = clf.decision_function(sc.transform(Xte))
        scores[name] = s
        lo, hi = boot_ci(yte, s, args.n_boot)
        results["models"][name] = {"auroc": float(roc_auc_score(yte, s)),
                                   "ap": float(average_precision_score(yte, s)),
                                   "ci95": [lo, hi]}
        print(f"  {name:<34} {roc_auc_score(yte, s):>7.4f} "
              f"{average_precision_score(yte, s):>7.4f}   [{lo:.4f}, {hi:.4f}]")

    if not args.skip_model:
        if args.window_scores:
            # Precomputed path: the model has already been run over the test windows and its
            # per-window scores dumped. Everything below this point is identical either way.
            z = np.load(args.window_scores, allow_pickle=True)
            win = np.asarray(z["score"], float)
            wseq = np.asarray(z["window_sequence_id"])
            print(f"\n  window scores loaded from {args.window_scores} "
                  f"({len(win)} windows)")
        else:
            import torch
            from torch.utils.data import DataLoader
            from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories
            from src.models.tgn_detector import TGNDetector

            cfg["model"]["node_meta_dim"] = 3 + 4 if args.structural_features else 3
            cfg["model"]["readout"] = args.readout
            cfg["model"]["memory_slots"] = args.memory_slots
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = TGNDetector(cfg).to(device)
            model.load_state_dict(torch.load(Path(args.run_dir) / args.checkpoint,
                                             map_location=device, weights_only=True))
            model.eval()

            ds = MemoryTrajectoryDataset(args.data, split="test", temporal_window=T,
                                         embedding_dim=emb_dim,
                                         structural_features=args.structural_features)
            loader = DataLoader(ds, batch_size=args.batch_size or cfg["training"]["batch_size"],
                                shuffle=False,
                                collate_fn=collate_trajectories)
            model.reset_memory()
            win = []
            with torch.no_grad():
                for batch in loader:
                    graphs = [g.to(device) for g in batch["graphs"]]
                    s, _ = model(graphs)
                    win.extend(s.cpu().numpy().tolist())
            win = np.array(win, float)
            wseq = np.array([w[0] for w in ds.windows])

        order = [r["sequence_id"] for r in cached["test"]]
        for agg, fn in (("max", np.max), ("mean", np.mean), ("last", lambda v: v[-1])):
            s = np.array([fn(win[wseq == sid]) if (wseq == sid).any() else 0.0
                          for sid in order], float)
            key = f"{args.label} (window {agg})"
            scores[key] = s
            lo, hi = boot_ci(yte, s, args.n_boot)
            results["models"][key] = {"auroc": float(roc_auc_score(yte, s)),
                                      "ap": float(average_precision_score(yte, s)),
                                      "ci95": [lo, hi]}
            print(f"  {key:<34} {roc_auc_score(yte, s):>7.4f} "
                  f"{average_precision_score(yte, s):>7.4f}   [{lo:.4f}, {hi:.4f}]")

        print(f"\n  paired bootstrap against the strongest hand-built baseline "
              f"({args.n_boot} resamples)")
        base_name = max(("final-state features (6)", "temporal features (7)",
                         "all features (13)"),
                        key=lambda k: results["models"][k]["auroc"])
        print(f"  baseline: {base_name} (AUROC {results['models'][base_name]['auroc']:.4f})")
        for agg in ("max", "mean", "last"):
            key = f"{args.label} (window {agg})"
            results["models"][key]["delta_vs_best_baseline"] = {"baseline": base_name}
            for metric in ("auroc", "ap"):
                m, lo, hi = paired_boot(yte, scores[key], scores[base_name],
                                        args.n_boot, metric=metric)
                sig = "significant" if (lo > 0 or hi < 0) else "NOT significant"
                print(f"    {key:<32} {metric.upper():<5} delta={m:+.4f}  "
                      f"95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")
                results["models"][key]["delta_vs_best_baseline"][metric] = {
                    "delta": m, "ci95": [lo, hi]}

        # Per-sequence scores, so any further comparison can be made without re-running the
        # model over 5,500 windows on a card that is usually busy.
        np.savez(Path(args.run_dir) / "sequence_scores_test.npz",
                 sequence_id=np.array(order), y=yte,
                 **{k.replace(" ", "_"): v for k, v in scores.items()})

    out = Path(args.output) if args.output else (
        Path(args.run_dir) / "sequence_eval.json" if args.run_dir
        else data / "sequence_eval_baselines.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
