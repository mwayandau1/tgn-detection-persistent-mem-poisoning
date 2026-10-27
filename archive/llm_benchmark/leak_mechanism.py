"""
Why does the content leak exist, and can rebalancing fix it?
============================================================

benchmark_gate.py detects a content leak: a logistic regression on ONE record's embedding
scores AUROC 1.000 on data/raw/llm_generated, so the label is recoverable from a single
record's text and no graph model is being tested. This script answers the two questions that
follow from that verdict, because both have obvious-sounding answers that are wrong.

Q1. IS IT CLASS IMBALANCE?  The intuition is that 2.29% positives is a degenerate ratio and
    that a fairer split would bring the score down. It does not, and the reason is structural
    rather than empirical: AUROC is a function of the RANKING alone. If every injected record
    outranks every benign record, the statistic is 1.0 at any prevalence whatever. AP is
    prevalence-sensitive in general, but under perfect separation precision is 1.0 at every
    recall level, so AP is pinned at 1.0 too. This script demonstrates that rather than
    asserting it, by thinning the positive class in the training set (which starves the model)
    and, separately, in the TEST set (which is what actually moves prevalence).

    The distinction matters for the thesis: a leak is a property of the feature space, not of
    the class ratio, so it cannot be removed by resampling. It can only be removed by changing
    what the poisoned text IS.

Q2. THEN WHAT SEPARATES THEM?  Measured three ways.

    (a) Cosine geometry. The naive expectation is that the two classes sit far apart. They do
        not. What distinguishes them is that the injected records are far more homogeneous
        among themselves than the benign records are, because they were generated from a
        narrow set of attack instructions while the benign records came from varied prompts.
        A linear model does not need the classes to be distant -- a tight cluster anywhere in
        the space is enough.

    (b) How many embedding dimensions a linear model needs, over an ANOVA ranking. A leak
        carried by one dimension would be an artefact worth stripping out. One spread across
        the whole embedding is a distributional property of the generated text, and the only
        remedy is a different generator.

    (c) The same measurement on a benchmark that passes the gate, so the contrast is visible
        rather than asserted.

Usage:
    python -m src.experiments.leak_mechanism \
        --data data/raw/llm_generated --contrast data/raw/ap_burst2k
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.dataset import MemoryTrajectoryDataset  # noqa: E402

FRACTIONS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.01)
TOP_K = (1, 2, 5, 10, 25, 50, 384)
COSINE_CAP = 1500          # pairwise cosine is O(n^2); cap it and say so


def collect(data: Path, split: str, window: int, emb_dim: int):
    """One row per record of each sequence's final snapshot: the raw embedding, and its label.

    This is deliberately the weakest possible view of the data -- no edges, no pooling, no
    temporal order. Anything separable here is separable without a graph.
    """
    ds = MemoryTrajectoryDataset(str(data), split=split,
                                 temporal_window=window, embedding_dim=emb_dim)
    idxs = np.linspace(0, len(ds) - 1, min(len(ds.sequence_ids), len(ds))).astype(int)
    X, y = [], []
    for i in idxs:
        item = ds[int(i)]
        X.append(item["graphs"][-1].x[:, :emb_dim].numpy())
        y.append(item["node_labels"].numpy())
    return np.vstack(X), np.concatenate(y)


def fit_probe(Xtr, ytr, Xte, cols=None):
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    return clf.decision_function(Xte)


def mean_offdiag_cosine(A, seed):
    """Mean pairwise cosine within a set of unit vectors, excluding self-similarity."""
    if len(A) > COSINE_CAP:
        A = A[np.random.default_rng(seed).choice(len(A), COSINE_CAP, replace=False)]
    S = A @ A.T
    n = len(A)
    return float((S.sum() - np.trace(S)) / max(n * (n - 1), 1))


def dimension_curve(Xtr, ytr, Xte, yte, ks):
    """AUROC/AP as a function of how many top-ANOVA embedding dimensions are supplied."""
    with warnings.catch_warnings():
        # Dead embedding dimensions are constant and yield 0/0 in the F statistic.
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        F, _ = f_classif(Xtr, ytr)
    order = np.argsort(-np.nan_to_num(F))
    out = []
    for k in ks:
        if k > Xtr.shape[1]:
            continue
        s = fit_probe(Xtr, ytr, Xte, cols=order[:k])
        out.append({"k": int(k),
                    "auroc": float(roc_auc_score(yte, s)),
                    "ap": float(average_precision_score(yte, s))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="benchmark that failed the content control")
    ap.add_argument("--contrast", default=None, help="benchmark that passed it, for comparison")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cfg = yaml.safe_load(open(args.config))
    window = cfg["training"]["temporal_window"]
    emb = cfg["simulation"]["embedding_dim"]
    data = Path(args.data)

    Xtr, ytr = collect(data, "train", window, emb)
    Xte, yte = collect(data, "test", window, emb)
    rng = np.random.default_rng(args.seed)

    print(f"\n{data.name}")
    print(f"  train {Xtr.shape[0]:>6} records, {int(ytr.sum()):>5} injected ({100*ytr.mean():.2f}%)")
    print(f"   test {Xte.shape[0]:>6} records, {int(yte.sum()):>5} injected ({100*yte.mean():.2f}%)")

    # --- Q1 -----------------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("Q1  Does shrinking the positive class remove the leak?")
    print("=" * 74)

    print("\n  Thinning TRAIN positives (the model gets fewer examples to learn from):")
    print(f"    {'kept':>6} {'n_pos':>7} {'AUROC':>8} {'AP':>8}")
    pos_idx, neg_idx = np.flatnonzero(ytr == 1), np.flatnonzero(ytr == 0)
    train_curve = []
    for frac in FRACTIONS:
        keep = rng.choice(pos_idx, size=max(2, int(len(pos_idx) * frac)), replace=False)
        idx = np.concatenate([neg_idx, keep])
        s = fit_probe(Xtr[idx], ytr[idx], Xte)
        row = {"fraction": frac, "n_pos": int(len(keep)),
               "auroc": float(roc_auc_score(yte, s)),
               "ap": float(average_precision_score(yte, s))}
        train_curve.append(row)
        print(f"    {frac:>5.0%} {row['n_pos']:>7} {row['auroc']:>8.4f} {row['ap']:>8.4f}")

    print("\n  Thinning TEST positives (this is what actually moves prevalence):")
    print(f"    {'kept':>6} {'n_pos':>7} {'prev':>8} {'AUROC':>8} {'AP':>8}")
    s_all = fit_probe(Xtr, ytr, Xte)
    te_pos, te_neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    test_curve = []
    for frac in FRACTIONS:
        keep = rng.choice(te_pos, size=max(2, int(len(te_pos) * frac)), replace=False)
        idx = np.concatenate([te_neg, keep])
        row = {"fraction": frac, "n_pos": int(len(keep)),
               "prevalence": float(yte[idx].mean()),
               "auroc": float(roc_auc_score(yte[idx], s_all[idx])),
               "ap": float(average_precision_score(yte[idx], s_all[idx]))}
        test_curve.append(row)
        print(f"    {frac:>5.0%} {row['n_pos']:>7} {row['prevalence']:>7.2%} "
              f"{row['auroc']:>8.4f} {row['ap']:>8.4f}")

    invariant = all(r["auroc"] > 0.999 for r in test_curve)
    print(f"\n  Prevalence fell {test_curve[0]['prevalence'] / test_curve[-1]['prevalence']:.0f}x"
          f" across that table.")
    print("  verdict: " + ("REBALANCING CANNOT FIX THIS -- the classes are separable at every "
                           "prevalence" if invariant else
                           "score moved with prevalence -- inspect before concluding"))

    # --- Q2 -----------------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("Q2  Then what is separating them?")
    print("=" * 74)

    u = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-8)
    P, N = u[yte == 1], u[yte == 0]
    within_p = mean_offdiag_cosine(P, seed=args.seed + 1)
    within_n = mean_offdiag_cosine(N, seed=args.seed + 2)
    Ps = P if len(P) <= COSINE_CAP else P[np.random.default_rng(args.seed + 3).choice(
        len(P), COSINE_CAP, replace=False)]
    Ns = N if len(N) <= COSINE_CAP else N[np.random.default_rng(args.seed + 4).choice(
        len(N), COSINE_CAP, replace=False)]
    between = float((Ps @ Ns.T).mean())
    cp, cn = P.mean(0), N.mean(0)
    centroid = float((cp / np.linalg.norm(cp)) @ (cn / np.linalg.norm(cn)))

    print(f"\n  mean cosine within injected records : {within_p:.4f}   (n={len(P)})")
    print(f"  mean cosine within benign records   : {within_n:.4f}   (n={len(N)})")
    print(f"  mean cosine between the two classes : {between:.4f}")
    print(f"  cosine between the two centroids    : {centroid:.4f}")
    if within_p > within_n and between >= within_n:
        print("\n  The classes are NOT far apart -- an injected record resembles a benign one")
        print("  about as much as two benign records resemble each other. What gives the label")
        print("  away is homogeneity: the injected records form a tight cluster because they")
        print("  were generated from a narrow set of instructions.")

    print("\n  How many embedding dimensions does a linear probe need?")
    print(f"    {'top-k':>6} {'AUROC':>8} {'AP':>8}")
    curve = dimension_curve(Xtr, ytr, Xte, yte, TOP_K)
    for row in curve:
        print(f"    {row['k']:>6} {row['auroc']:>8.4f} {row['ap']:>8.4f}")
    if curve[0]["auroc"] < 0.75:
        print("\n  No single dimension carries it, so this is not an artefact that could be")
        print("  stripped out. It is a distributional property of the generated text.")

    result = {"data": data.name,
              "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
              "train_prevalence": float(ytr.mean()), "test_prevalence": float(yte.mean()),
              "thin_train": train_curve, "thin_test": test_curve,
              "prevalence_invariant": bool(invariant),
              "cosine": {"within_injected": within_p, "within_benign": within_n,
                         "between_classes": between, "between_centroids": centroid},
              "dimension_curve": curve}

    # --- contrast -----------------------------------------------------------------------
    if args.contrast:
        cdata = Path(args.contrast)
        Ctr, cytr = collect(cdata, "train", window, emb)
        Cte, cyte = collect(cdata, "test", window, emb)
        print(f"\n  Same measurement on {cdata.name}, which passes the content control:")
        print(f"    {'top-k':>6} {'AUROC':>8} {'AP':>8}")
        ccurve = dimension_curve(Ctr, cytr, Cte, cyte, (1, 10, 384))
        for row in ccurve:
            print(f"    {row['k']:>6} {row['auroc']:>8.4f} {row['ap']:>8.4f}")
        print(f"    (base rate {cyte.mean():.4f}; AP at chance equals the base rate)")
        result["contrast"] = {"data": cdata.name, "dimension_curve": ccurve,
                              "test_prevalence": float(cyte.mean())}

    out = Path(args.output) if args.output else data / "leak_mechanism.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
