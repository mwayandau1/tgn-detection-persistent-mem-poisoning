"""
Isolation Forest and One-Class SVM on the replacement benchmark.

These are two of the three baselines Chapter 4 publishes. Both were measured on
data/raw/llm_generated, where a per-record content probe reaches AUROC 1.000, so every number
in that table describes a benchmark that has since been withdrawn. This re-runs them on
ap_burst2k so the comparison table in the amended thesis is measured on the benchmark the
detector is actually evaluated on.

TWO LEVELS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS

  Record level -- "are injected records distributional outliers?" This is the claim the thesis
  makes and it is the one worth preserving: both detectors at or below chance means an
  ingestion-time filter with no labelled examples has nothing to fire on. On ap_burst2k the
  claim should hold even more strongly than before, because every record in that benchmark --
  benign or poisoned -- is a genuine unmodified StrategyQA knowledge-base entry, so there is
  no text-level difference to be an outlier in. Confirming that is a control on the benchmark
  as much as a result about the detectors.

  Sequence level -- "is this agent's memory poisoned?" This is the question the detector is
  evaluated on, so an unsupervised baseline has to be scored on it to be comparable. Fitted on
  the benign TRAINING sequences only, which is the realistic deployment: an operator has clean
  history and no labelled attacks. Scored on test, bootstrapped over sequences with the same
  boot_ci used for every other sequence-level number in the project.

Neither detector sees a label at fit time. The labels are used only to score what came out.

Usage:
    python -m src.experiments.unsupervised_baselines --data data/raw/ap_burst2k
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.dataset import MemoryTrajectoryDataset  # noqa: E402
from src.experiments.sequence_eval import (  # noqa: E402
    FINAL_FEATS,
    TEMPORAL_FEATS,
    boot_ci,
    sequence_features,
    to_matrix,
)

OCSVM_CAP = 5000       # RBF One-Class SVM is O(n^2); the notebook capped at 5000 too


def record_features(data: Path, split: str, window: int, emb_dim: int, cap: int):
    """Final-snapshot record embeddings and their labels, one row per record."""
    ds = MemoryTrajectoryDataset(str(data), split=split, temporal_window=window,
                                 embedding_dim=emb_dim)
    idxs = np.linspace(0, len(ds) - 1, min(len(ds.sequence_ids), len(ds))).astype(int)
    X, y = [], []
    for i in idxs:
        item = ds[int(i)]
        X.append(item["graphs"][-1].x[:, :emb_dim].numpy())
        y.append(item["node_labels"].numpy())
    X, y = np.vstack(X), np.concatenate(y)
    if cap and len(X) > cap:
        keep = np.random.default_rng(0).choice(len(X), cap, replace=False)
        X, y = X[keep], y[keep]
    return X, y


def score_both(X_fit, X_score, seed=42):
    """Fit both unsupervised detectors on X_fit, return anomaly scores for X_score.

    Higher is more anomalous in both cases, so the sign convention matches the labels.
    """
    scaler = StandardScaler().fit(X_fit)
    A, B = scaler.transform(X_fit), scaler.transform(X_score)

    iso = IsolationForest(contamination=0.1, random_state=seed, n_jobs=-1).fit(A)
    iso_scores = -iso.score_samples(B)

    fit = A if len(A) <= OCSVM_CAP else A[np.random.default_rng(seed).choice(
        len(A), OCSVM_CAP, replace=False)]
    svm = OneClassSVM(kernel="rbf", nu=0.1).fit(fit)
    svm_scores = -svm.decision_function(B)
    return {"Isolation Forest": iso_scores, "One-Class SVM": svm_scores}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--record-cap", type=int, default=20000,
                    help="Subsample records before fitting; 0 disables.")
    ap.add_argument("--skip-record", action="store_true",
                    help="Sequence level only. The record-level One-Class SVM is the slow "
                         "part and its result does not change when only the sequence-level "
                         "output is being regenerated.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    window = cfg["training"]["temporal_window"]
    emb_dim = cfg["simulation"]["embedding_dim"]
    threshold = cfg["simulation"]["semantic_edge_threshold"]
    data = Path(args.data)

    meta = {m["sequence_id"]: m for m in json.load(open(data / "metadata.json"))}
    splits = json.load(open(data / "splits.json"))
    results = {"data": data.name, "record_level": {}, "sequence_level": {}}

    # --- record level ------------------------------------------------------------------
    if not args.skip_record:
        print(f"\n{data.name} -- RECORD LEVEL")
        print("  'are injected records distributional outliers?'")
        Xte, yte = record_features(data, "test", window, emb_dim, args.record_cap)
        print(f"  {len(Xte):,} test records, {int(yte.sum())} injected ({yte.mean():.2%})")
        print(f"\n  {'detector':<20} {'AUROC':>8} {'AP':>8}")
        for name, s in score_both(Xte, Xte).items():      # unsupervised: fit on what it scores
            auroc, apr = roc_auc_score(yte, s), average_precision_score(yte, s)
            results["record_level"][name] = {"auroc": float(auroc), "ap": float(apr)}
            print(f"  {name:<20} {auroc:>8.4f} {apr:>8.4f}")
        print(f"  {'(base rate)':<20} {0.5:>8.4f} {yte.mean():>8.4f}")

    # --- sequence level ----------------------------------------------------------------
    print(f"\n{data.name} -- SEQUENCE LEVEL")
    print("  'is this agent's memory poisoned?'  fitted on benign training sequences only")
    cache = data / "sequence_features.json"
    if cache.exists():
        cached = json.load(open(cache))
        print(f"  loaded cached sequence features from {cache}")
    else:
        cached = {n: sequence_features(data, splits[n], threshold, emb_dim)
                  for n in ("train", "val", "test")}
        json.dump(cached, open(cache, "w"))

    feats = FINAL_FEATS + TEMPORAL_FEATS
    ytr = np.array([int(meta[r["sequence_id"]]["label"] == "poisoned") for r in cached["train"]])
    y = np.array([int(meta[r["sequence_id"]]["label"] == "poisoned") for r in cached["test"]])
    Xtr, Xs = to_matrix(cached["train"], feats), to_matrix(cached["test"], feats)
    benign = Xtr[ytr == 0]
    print(f"  fitted on {len(benign)} benign train sequences; "
          f"scoring {len(Xs)} test sequences ({int(y.sum())} poisoned)")
    print(f"\n  {'detector':<20} {'AUROC':>8} {'AP':>8}   {'95% CI (AUROC)':>18}")
    dumps = {}
    for name, s in score_both(benign, Xs).items():
        auroc, apr = roc_auc_score(y, s), average_precision_score(y, s)
        lo, hi = boot_ci(y, s, args.n_boot)
        results["sequence_level"][name] = {"auroc": float(auroc), "ap": float(apr),
                                           "ci95": [lo, hi]}
        dumps[name.replace(" ", "_")] = s
        print(f"  {name:<20} {auroc:>8.4f} {apr:>8.4f}   [{lo:.4f}, {hi:.4f}]")
    print(f"  {'(base rate)':<20} {0.5:>8.4f} {y.mean():>8.4f}")

    # Per-sequence scores, so these can be compared to the detector with the same paired
    # bootstrap used everywhere else rather than by reading two confidence intervals off the
    # page -- overlapping intervals are not a test, and non-overlapping ones are conservative.
    order = np.array([r["sequence_id"] for r in cached["test"]])
    np.savez(data / "unsupervised_sequence_scores_test.npz",
             sequence_id=order, y=y, **dumps)

    out = Path(args.output) if args.output else data / "unsupervised_baselines.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")
    print(f"Per-sequence scores -> {data / 'unsupervised_sequence_scores_test.npz'}")


if __name__ == "__main__":
    main()
