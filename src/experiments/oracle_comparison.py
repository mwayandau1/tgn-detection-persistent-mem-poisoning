"""
The detector against every engineered-feature oracle, at sequence level, in one table.
=====================================================================================

Section 5.5 reports one paired comparison against the thirteen-feature regression and
Chapter 4 reports none. This puts the whole family side by side, on the 500 test sequences,
with a bootstrap interval on each model and a paired interval on each difference from the
detector:

  LR, final-state (6)        what a single-snapshot model can see
  LR, temporal (7)           growth-rate summaries
  LR, all (13)               both, fitted jointly -- the oracle Section 5.5 quotes
  GB, final-state / temporal / all
                             gradient boosting on the same columns, because a reviewer will
                             ask whether the tie with the linear oracle survives a non-linear
                             one. Hyperparameters are chosen on VALIDATION AUROC from a small
                             grid stated below, then the model fitted on TRAIN is scored on
                             TEST -- the same fit/select/score protocol the detector had.
  LR, nine features, final snapshot
                             the window-level structural features of Section 3.11 computed on
                             each sequence's final snapshot: one row per sequence
  LR, nine features, max over windows
                             the window-level regression of Table 4.2, aggregated to sequence
                             level by max exactly as the detector is

Every model is fitted on the training partition only; nothing is fitted on test.

Usage:
    python -m src.experiments.oracle_comparison --data data/raw/ap_burst2k \
        --detector experiments/audit/base_test_b1.npz \
        --window-lr experiments/run_burst2k_base/window_scores_test.npz
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.sequence_eval import (FINAL_FEATS, TEMPORAL_FEATS, boot_ci,  # noqa: E402
                                           paired_boot, to_matrix)

GB_GRID = {"max_depth": [2, 3, None], "learning_rate": [0.03, 0.1], "max_iter": [100, 300]}


def nine_final_snapshot(data: Path, seq_ids, threshold: float, emb_dim: int, cluster_k: int):
    """The nine window-level features, computed on each sequence's last snapshot."""
    from src.experiments.separability_probe import _best_cluster_stats
    rows = []
    for sid in seq_ids:
        last = sorted((data / sid).glob("snapshot_*.npz"))[-1]
        with np.load(last) as z:
            x = z["node_features"][:, :emb_dim]
            e = z["edge_index"].shape[1]
        n = x.shape[0]
        u = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        sim = u @ u.T
        np.fill_diagonal(sim, 0.0)
        deg = (sim > threshold).sum(axis=1)
        bd, bs = _best_cluster_stats(sim, min(cluster_k, max(n - 1, 2)), threshold)
        above = sim[sim > threshold]
        rows.append([n, e, e / max(n * (n - 1), 1), float(deg.max()) if n else 0.0, bd, bs,
                     float(above.mean()) if above.size else 0.0,
                     float(above.max()) if above.size else 0.0,
                     float(np.percentile(above, 95)) if above.size else 0.0])
    return np.array(rows, float)


def fit_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def fit_gb(Xtr, ytr, Xva, yva, Xte, seed=0):
    best = None
    for md, lr, it in itertools.product(*GB_GRID.values()):
        clf = HistGradientBoostingClassifier(max_depth=md, learning_rate=lr, max_iter=it,
                                             random_state=seed).fit(Xtr, ytr)
        a = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if best is None or a > best[0]:
            best = (a, {"max_depth": md, "learning_rate": lr, "max_iter": it}, clf)
    return best[2].predict_proba(Xte)[:, 1], best[1], best[0]


def seq_max(npz_path: Path, key: str, order):
    z = np.load(npz_path, allow_pickle=True)
    seq = np.asarray(z["seq"] if "seq" in z.files else z["window_sequence_id"])
    s = np.asarray(z[key], float)
    return np.array([s[seq == sid].max() for sid in order], float)


def multi_seed_comparison(y, det_runs, oracles, n_boot, seed=0):
    """Seed-averaged detector metrics and seed-averaged paired differences from each oracle."""
    rng = np.random.default_rng(seed)
    boots = []
    while len(boots) < n_boot:
        i = rng.choice(len(y), size=len(y), replace=True)
        if 0 < y[i].sum() < len(i):
            boots.append(i)
    fns = {"auroc": roc_auc_score, "ap": average_precision_score}
    det = {m: [fn(y, s) for s in det_runs] for m, fn in fns.items()}
    out = {"n_test": int(len(y)), "n_seeds": len(det_runs),
           "detector": {m: {"per_seed": v, "mean": float(np.mean(v)),
                            "sd": float(np.std(v, ddof=1))} for m, v in det.items()},
           "oracles": {}}
    print(f"\ndetector over {len(det_runs)} seeds: AUROC {np.mean(det['auroc']):.4f} "
          f"+/- {np.std(det['auroc'], ddof=1):.4f}, AP {np.mean(det['ap']):.4f}")
    print(f"{'oracle':<40}{'AUROC':>8}{'AP':>8}   {'det - oracle AUROC [CI]':>28}   "
          f"{'det - oracle AP [CI]':>26}")
    for name, s in oracles.items():
        row = {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}
        line = f"{name:<40}{row['auroc']:>8.4f}{row['ap']:>8.4f}"
        for m, fn in fns.items():
            point = float(np.mean([fn(y, d) - fn(y, s) for d in det_runs]))
            dist = [np.mean([fn(y[i], d[i]) - fn(y[i], s[i]) for d in det_runs]) for i in boots]
            lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
            row[f"detector_minus_{m}"] = {"delta": point, "ci95": [lo, hi],
                                          "significant": bool(lo > 0 or hi < 0)}
            line += f"   {point:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        out["oracles"][name] = row
        print(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--detector", nargs="+", required=True,
                    help="One or more window-score npz files (key 'score', or 'tgn'); "
                         "with several, each is reported and their mean score is the "
                         "detector row.")
    ap.add_argument("--window-lr", default=None,
                    help="npz with an 'lr' column of window-level nine-feature scores "
                         "(written by detection_latency.py)")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--emb-dim", type=int, default=384)
    ap.add_argument("--cluster-k", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default="experiments/audit/oracle_comparison.json")
    args = ap.parse_args()

    data = Path(args.data)
    meta = {m["sequence_id"]: m for m in json.load(open(data / "metadata.json"))}
    cached = json.load(open(data / "sequence_features.json"))
    lab = lambda rows: np.array([int(meta[r["sequence_id"]]["label"] == "poisoned")
                                 for r in rows])
    ytr, yva, yte = lab(cached["train"]), lab(cached["val"]), lab(cached["test"])
    order = [r["sequence_id"] for r in cached["test"]]

    scores, notes = {}, {}
    for name, feats in (("final-state (6)", FINAL_FEATS), ("temporal (7)", TEMPORAL_FEATS),
                        ("all (13)", FINAL_FEATS + TEMPORAL_FEATS)):
        Xtr, Xva, Xte = (to_matrix(cached[s], feats) for s in ("train", "val", "test"))
        scores[f"LR, {name}"] = fit_lr(Xtr, ytr, Xte)
        s, hp, va = fit_gb(Xtr, ytr, Xva, yva, Xte)
        scores[f"GB, {name}"] = s
        notes[f"GB, {name}"] = {"selected": hp, "val_auroc": va}

    ids = {s: [r["sequence_id"] for r in cached[s]] for s in ("train", "test")}
    X9tr = nine_final_snapshot(data, ids["train"], args.threshold, args.emb_dim, args.cluster_k)
    X9te = nine_final_snapshot(data, ids["test"], args.threshold, args.emb_dim, args.cluster_k)
    scores["LR, nine features, final snapshot"] = fit_lr(X9tr, ytr, X9te)
    if args.window_lr:
        scores["LR, nine features, max over windows"] = seq_max(Path(args.window_lr), "lr",
                                                                order)

    det_runs = []
    for p in args.detector:
        z = np.load(p, allow_pickle=True)
        key = "score" if "score" in z.files else "tgn"
        det_runs.append(seq_max(Path(p), key, order))
    if len(det_runs) > 1:
        # Several seeds: compare each seed with each oracle on the same bootstrap resample and
        # average the differences, so the comparison is against a typical single detector,
        # not against an ensemble of all seeds (which would be a different, stronger model).
        out = multi_seed_comparison(yte, det_runs, scores, args.n_boot)
        out["detector_inputs"] = args.detector
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"\nsaved {args.output}")
        return
    det_name = "detector (window max)"
    scores[det_name] = det_runs[0]

    out = {"n_test": int(len(yte)), "n_test_poisoned": int(yte.sum()),
           "detector_inputs": args.detector, "models": {}, "gb_selection": notes}
    det = scores[det_name]
    print(f"\n{'model':<40}{'AUROC':>8}{'AP':>8}   {'95% CI':>18}   "
          f"{'det - model AUROC [CI]':>30}   {'det - model AP [CI]':>28}")
    for name, s in scores.items():
        lo, hi = boot_ci(yte, s, args.n_boot)
        row = {"auroc": float(roc_auc_score(yte, s)),
               "ap": float(average_precision_score(yte, s)), "ci95": [lo, hi]}
        line = f"{name:<40}{row['auroc']:>8.4f}{row['ap']:>8.4f}   [{lo:.4f}, {hi:.4f}]"
        if name != det_name:
            for m in ("auroc", "ap"):
                d, dlo, dhi = paired_boot(yte, det, s, args.n_boot, metric=m)
                row[f"detector_minus_{m}"] = {"delta": d, "ci95": [dlo, dhi],
                                              "significant": bool(dlo > 0 or dhi < 0)}
                line += f"   {d:+.4f} [{dlo:+.4f}, {dhi:+.4f}]"
        out["models"][name] = row
        print(line)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
