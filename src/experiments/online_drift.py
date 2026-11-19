"""
How the online score depends on how much of the sequence the detector has seen (step 6).
========================================================================================

A detector trained on whole 20-snapshot sequences is read after snapshots 9 to 19
(score_windows --online). For each end point: AUROC of poisoned against clean sequences (by
sequence label, so the population is the same at every end point), AUROC against the window
label (attack present by that snapshot), and the clean sequences' median and 95th percentile
score; mean over the seeds given. Descriptive only: no choice depends on it.

Usage:
    python -m src.experiments.online_drift --runs experiments/seeds_coherence/tgn_s{1..5} \
        --split test --output experiments/seeds_coherence/online_drift_test.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def by_end(npz: Path):
    z = np.load(npz, allow_pickle=True)
    seq, end, y, s = z["seq"], z["end"], z["y"], z["score"]
    poisoned = np.array([q.startswith("poisoned") for q in seq])
    rows = {}
    for e in sorted(set(end.tolist())):
        m = end == e
        rows[str(e)] = {
            "auroc_sequence_label": roc_auc_score(poisoned[m], s[m]),
            "auroc_window_label": (roc_auc_score(y[m], s[m]) if 0 < y[m].sum() < m.sum()
                                   else None),
            "clean_median": float(np.median(s[m & ~poisoned])),
            "clean_p95": float(np.quantile(s[m & ~poisoned], 0.95)),
            "poisoned_median": float(np.median(s[m & poisoned]))}
    return rows, float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    per_run, pooled = {}, []
    for r in args.runs:
        rows, p = by_end(Path(r) / f"online_scores_{args.split}_b1.npz")
        per_run[Path(r).name] = rows
        pooled.append(p)
    ends = list(next(iter(per_run.values())))
    mean = {e: {k: (float(np.mean([per_run[r][e][k] for r in per_run]))
                    if all(per_run[r][e][k] is not None for r in per_run) else None)
                for k in per_run[next(iter(per_run))][e]} for e in ends}
    out = {"split": args.split, "runs": [Path(r).as_posix() for r in args.runs],
           "pooled_online_auroc": {"mean": float(np.mean(pooled)), "per_run": pooled},
           "mean_over_runs": mean, "per_run": per_run}
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
