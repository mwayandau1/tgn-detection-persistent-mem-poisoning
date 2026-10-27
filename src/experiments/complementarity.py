"""
Does the T-GNN carry information the nine hand-crafted features do not?
======================================================================

A tie in AUROC does not answer this. Two models can score alike and still disagree on which
sequences they get right. The test is whether adding the T-GNN's score to a logistic
regression on the nine features improves it, and whether that improvement survives a
confidence interval.

Three methodological choices, made explicit because getting any of them wrong invalidates the
result:

1.  WHERE THE COMBINER IS FIT.  The T-GNN was trained on the train split, so its scores there
    are in-sample and inflated. A combiner fit on them would learn to trust a score that looks
    better than it will be at test time. Both logistic regressions are therefore fit on the
    VALIDATION split, where the T-GNN's scores are out-of-sample, and evaluated on TEST. The
    two models then differ only by the presence of the extra column, which is what a nested
    comparison requires. The train-fit nine-feature model is also reported, as the stronger
    baseline previously quoted, but it is not the one used for the nested test.

2.  WHAT THE BOOTSTRAP RESAMPLES.  Each sequence contributes 11 heavily overlapping windows,
    so windows are not independent observations. Resampling windows would understate the
    variance badly. This resamples SEQUENCES with replacement and takes all their windows --
    a cluster bootstrap -- so the interval reflects the number of sequences, which is the real
    sample size.

3.  WHY NOT DELONG.  DeLong's test assumes independent observations and would be
    anti-conservative on clustered windows for the same reason. The paired cluster bootstrap
    answers the same question without that assumption. A likelihood-ratio test is reported for
    the nested pair, flagged as anti-conservative here for the same clustering reason.

Usage:
    python -m src.experiments.complementarity --data data/raw/ap_burst2k_nodup \
        --run-dir experiments/seeds_nodup/tgn_s1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories
from src.models.tgn_detector import TGNDetector
from src.experiments.separability_probe import _best_cluster_stats


def structural_features(dataset, emb_dim, threshold, cluster_k):
    """The nine features, exactly as in separability_probe.py."""
    out, labels, seqs = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        g = item["graphs"][-1]
        x = g.x[:, :emb_dim].numpy()
        n, e = x.shape[0], g.edge_index.size(1)
        nx_ = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = nx_ @ nx_.T
        np.fill_diagonal(sim, 0.0)
        deg = (sim > threshold).sum(axis=1)
        bd, bs = _best_cluster_stats(sim, min(cluster_k, max(n - 1, 2)), threshold)
        above = sim[sim > threshold]
        out.append([n, e, e / max(n * (n - 1), 1), float(deg.max()) if n else 0.0, bd, bs,
                    float(above.mean()) if above.size else 0.0,
                    float(above.max()) if above.size else 0.0,
                    float(np.percentile(above, 95)) if above.size else 0.0])
        labels.append(int(item["graph_label"]))
        seqs.append(item["sequence_id"])
    return np.array(out, float), np.array(labels, int), np.array(seqs)


@torch.no_grad()
def tgn_scores(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_trajectories)
    model.reset_memory()
    out = []
    for batch in loader:
        graphs = [g.to(device) for g in batch["graphs"]]
        s, _ = model(graphs)
        out.extend(s.cpu().numpy().tolist())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.array(out, float)


def cluster_bootstrap(seqs, y, score_a, score_b, n_boot=2000, seed=0):
    """Paired cluster bootstrap on AUROC(a) - AUROC(b). Resamples sequences, not windows."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seqs)
    idx_by_seq = {s: np.flatnonzero(seqs == s) for s in uniq}
    deltas, a_s, b_s = [], [], []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_seq[s] for s in picked])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        aa = roc_auc_score(yy, score_a[idx])
        bb = roc_auc_score(yy, score_b[idx])
        a_s.append(aa); b_s.append(bb); deltas.append(aa - bb)
    d = np.array(deltas)
    return {
        "delta_mean": float(d.mean()),
        "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_delta_gt_0": float((d > 0).mean()),
        "a_ci95": [float(np.percentile(a_s, 2.5)), float(np.percentile(a_s, 97.5))],
        "b_ci95": [float(np.percentile(b_s, 2.5)), float(np.percentile(b_s, 97.5))],
        "n_resamples": int(len(d)),
    }


def loglik(clf, X, y):
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="phase2_best.pt")
    ap.add_argument("--structural-features", action="store_true", default=True)
    ap.add_argument("--scale-readout", action="store_true")
    ap.add_argument("--cluster-k", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    T = cfg["training"]["temporal_window"]
    emb = cfg["simulation"]["embedding_dim"]
    thr = cfg["simulation"]["semantic_edge_threshold"]
    if args.structural_features:
        cfg["model"]["node_meta_dim"] = 3 + 4
    cfg["model"]["scale_readout"] = bool(args.scale_readout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGNDetector(cfg).to(device)
    model.load_state_dict(torch.load(Path(args.run_dir) / args.checkpoint,
                                     map_location=device, weights_only=True))
    model.eval()

    splits = {}
    for name in ("train", "val", "test"):
        ds = MemoryTrajectoryDataset(args.data, split=name, temporal_window=T,
                                     embedding_dim=emb,
                                     structural_features=args.structural_features)
        X, y, s = structural_features(ds, emb, thr, args.cluster_k)
        g = tgn_scores(model, ds, device, cfg["training"]["batch_size"])
        assert len(g) == len(y), f"{name}: {len(g)} scores for {len(y)} windows"
        splits[name] = {"X": X, "y": y, "seq": s, "tgn": g}
        print(f"{name:>5}: {len(y)} windows, {int(y.sum())} positive, "
              f"{len(np.unique(s))} sequences")

    te = splits["test"]
    va = splits["val"]
    tr = splits["train"]

    # Scaler fit on val, the same data the combiners are fit on.
    sc = StandardScaler().fit(va["X"])
    Xva, Xte = sc.transform(va["X"]), sc.transform(te["X"])

    def fit(Xa, ya):
        return LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xa, ya)

    m9 = fit(Xva, va["y"])
    m10 = fit(np.hstack([Xva, va["tgn"][:, None]]), va["y"])
    s9 = m9.decision_function(Xte)
    s10 = m10.decision_function(np.hstack([Xte, te["tgn"][:, None]]))
    stgn = te["tgn"]

    sc_tr = StandardScaler().fit(tr["X"])
    m9_train = fit(sc_tr.transform(tr["X"]), tr["y"])
    s9_train = m9_train.decision_function(sc_tr.transform(te["X"]))

    auc = lambda s: float(roc_auc_score(te["y"], s))
    apr = lambda s: float(average_precision_score(te["y"], s))

    print("\n=== test-set performance (window level) ===")
    rows = [("9 features (fit on train)", s9_train),
            ("9 features (fit on val)", s9),
            ("T-GNN alone", stgn),
            ("9 features + T-GNN score", s10)]
    print(f"  {'model':<30} {'AUROC':>7} {'AP':>7}")
    for name, s in rows:
        print(f"  {name:<30} {auc(s):>7.4f} {apr(s):>7.4f}")

    print("\n=== paired cluster bootstrap over sequences "
          f"({args.n_boot} resamples, 125 test sequences) ===")
    results = {"test_auroc": {n: auc(s) for n, s in rows},
               "test_ap": {n: apr(s) for n, s in rows}}
    for label, a, b in [("combined - 9features", s10, s9),
                        ("T-GNN - 9features", stgn, s9)]:
        r = cluster_bootstrap(te["seq"], te["y"], a, b, n_boot=args.n_boot)
        results[label] = r
        sig = "significant" if (r["delta_ci95"][0] > 0 or r["delta_ci95"][1] < 0) else "NOT significant"
        print(f"  {label:<22} delta={r['delta_mean']:+.4f}  "
              f"95% CI [{r['delta_ci95'][0]:+.4f}, {r['delta_ci95'][1]:+.4f}]  {sig}")

    ll0, ll1 = loglik(m9, Xte, te["y"]), loglik(m10, np.hstack([Xte, te["tgn"][:, None]]), te["y"])
    lr_stat = 2 * (ll1 - ll0)
    from scipy.stats import chi2
    p = float(chi2.sf(lr_stat, df=1)) if lr_stat > 0 else 1.0
    results["likelihood_ratio"] = {"stat": float(lr_stat), "df": 1, "p": p,
                                   "caveat": "anti-conservative: windows within a sequence "
                                             "are not independent"}
    print(f"\n  likelihood-ratio (nested, df=1): stat={lr_stat:.2f}  p={p:.2e}")
    print("    treat as anti-conservative -- the cluster bootstrap above is the honest test")

    out = Path(args.output) if args.output else Path(args.run_dir) / "complementarity.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
