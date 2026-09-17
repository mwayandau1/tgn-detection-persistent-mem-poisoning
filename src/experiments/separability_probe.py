"""
Cheap Separability Probe (no GPU, no training)
===============================================
Measures how much of a dataset's poisoned-vs-clean signal is reachable by a plain
logistic regression on hand-crafted features. Runs in seconds, so it is worth
running on any newly generated benchmark *before* committing to a multi-hour TGN
training run.

Three probes, in increasing order of information:

  structural  : [n_nodes, n_edges, density, max_degree, best_cluster_density,
                 best_cluster_mean_similarity, mean/max/p95 edge weight]
                No embeddings. Answers "is this just a graph-size/density/tightness tell?"
                Includes CONTINUOUS edge-weight magnitude, not only thresholded counts:
                the model receives edge weights as `edge_attr`, so a probe restricted to
                thresholded features understates what is reachable and will read as
                harder than the benchmark really is.
  content     : mean-pooled node embeddings. No edges, no temporal order.
                Answers "is the poisoned content linearly separable on its own?"
  combined    : both feature sets concatenated.

Interpretation
--------------
High probe AUROC means the benchmark has a shortcut and a GNN is not being tested.
Probe AUROC near 0.5 does NOT mean the task is unlearnable — the TGN also uses
relational structure, node memory and temporal ordering, none of which these probes
capture. It only means the *cheap* shortcuts are gone.

This restores, in dataset-agnostic form, the `structural_leak_probe` and
`shortcut_baseline` scripts referenced in docs/near_perfect_metrics_defense.md §7,
which were folded into notebook 07 and no longer exist as standalone modules.

Usage:
    python -m src.experiments.separability_probe --data data/raw/synthetic_medium
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.dataset import MemoryTrajectoryDataset


def _best_cluster_stats(sim: np.ndarray, k: int, threshold: float):
    """Density AND mean similarity of the tightest k-node neighbourhood.

    The mean-similarity version matters more than the density version. Density
    thresholds the similarities and so discards their magnitude: when both the attack
    and ordinary same-topic traffic sit above the edge threshold, density looks nearly
    identical while the raw weights separate cleanly. Edge weights are fed to the model
    as `edge_attr`, so the continuous quantity is what a detector actually reads. An
    earlier calibration that used density alone understated separability badly.
    """
    n = sim.shape[0]
    if n <= k or k < 2:
        return 0.0, 0.0
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    order = np.argsort(-np.sort(s, axis=1)[:, k - 2])[:48]
    iu = np.triu_indices(k, k=1)
    best_den, best_sim = 0.0, 0.0
    for i in order:
        nb = np.argpartition(-s[i], k - 2)[:k - 1]
        g = np.concatenate([[i], nb])
        vals = s[np.ix_(g, g)][iu]
        best_den = max(best_den, float((vals > threshold).mean()))
        best_sim = max(best_sim, float(vals.mean()))
    return best_den, best_sim


def extract(dataset, embedding_dim: int, threshold: float, cluster_k: int):
    struct, content, labels = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        g = item["graphs"][-1]
        x = g.x[:, :embedding_dim].numpy()
        n = x.shape[0]
        e = g.edge_index.size(1)

        nx_ = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = nx_ @ nx_.T
        np.fill_diagonal(sim, 0.0)
        deg = (sim > threshold).sum(axis=1)

        best_den, best_sim = _best_cluster_stats(sim, min(cluster_k, n - 1), threshold)

        # Continuous edge-weight magnitude, over the pairs that actually form edges.
        above = sim[sim > threshold]
        w_mean = float(above.mean()) if above.size else 0.0
        w_max = float(above.max()) if above.size else 0.0
        w_p95 = float(np.percentile(above, 95)) if above.size else 0.0

        struct.append([
            n, e,
            e / max(n * (n - 1), 1),
            float(deg.max()) if n else 0.0,
            best_den,
            best_sim,      # continuous analogue of best_den — usually the strongest feature
            w_mean, w_max, w_p95,
        ])
        content.append(x.mean(axis=0))
        labels.append(item["graph_label"])

    return (np.array(struct, dtype=np.float64),
            np.stack(content),
            np.array(labels, dtype=np.int64))


def fit_and_score(name, Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        print(f"  {name:12s}: skipped (single class present)")
        return None
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    res = {"auroc": float(roc_auc_score(yte, s)),
           "ap": float(average_precision_score(yte, s))}
    print(f"  {name:12s}: AUROC={res['auroc']:.4f}  AP={res['ap']:.4f}")
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--cluster-k", type=int, default=15,
                        help="Neighbourhood size for the best-cluster-density feature; "
                             "set near the typical number of injected records.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    T = config["training"]["temporal_window"]
    emb_dim = config["simulation"]["embedding_dim"]
    threshold = config["simulation"]["semantic_edge_threshold"]

    train_ds = MemoryTrajectoryDataset(args.data, split="train", temporal_window=T)
    test_ds = MemoryTrajectoryDataset(args.data, split="test", temporal_window=T)
    print(f"{args.data}: {len(train_ds)} train windows, {len(test_ds)} test windows")

    Xs_tr, Xc_tr, y_tr = extract(train_ds, emb_dim, threshold, args.cluster_k)
    Xs_te, Xc_te, y_te = extract(test_ds, emb_dim, threshold, args.cluster_k)
    print(f"positives: {int(y_tr.sum())}/{len(y_tr)} train, {int(y_te.sum())}/{len(y_te)} test\n")

    results = {
        "structural": fit_and_score("structural", Xs_tr, y_tr, Xs_te, y_te),
        "content": fit_and_score("content", Xc_tr, y_tr, Xc_te, y_te),
        "combined": fit_and_score("combined",
                                  np.hstack([Xs_tr, Xc_tr]), y_tr,
                                  np.hstack([Xs_te, Xc_te]), y_te),
    }

    out = Path(args.output) if args.output else Path(args.data) / "separability_probe.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    print("\nReference (data/raw/llm_generated): structural 0.595, content 0.773, TGN 1.000.")


if __name__ == "__main__":
    main()
