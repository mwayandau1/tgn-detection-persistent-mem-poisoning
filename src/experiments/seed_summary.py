"""
Summarise the multi-seed study (E2, E3) at sequence level.
==========================================================

Reads experiments/seeds/{model}_s{seed}/window_scores_test_b1.npz (written by
train_variant.py), aggregates each run to one score per test sequence by window max, and
reports for every configuration:

  * AUROC and AP of each seed, their mean and sample standard deviation;
  * a 95% interval for the seed-averaged AUROC, from a bootstrap over sequences in which
    every seed's AUROC is recomputed on the same resample and then averaged, so the interval
    carries sequence sampling and is centred on the multi-seed mean;
  * the paired difference from the detector ("tgn"), matched seed by seed and averaged over
    the seeds both configurations share, with the same bootstrap.

The noise floor is then the detector's seed-to-seed standard deviation, measured, in place of
the 0.02 the thesis asserted from what turned out to be one seed run twice.

Usage:
    python -m src.experiments.seed_summary --root experiments/seeds \
        --output experiments/seeds/summary.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sequence_scores(npz: Path):
    z = np.load(npz, allow_pickle=True)
    seq, s = np.asarray(z["seq"]), np.asarray(z["score"], float)
    ids = sorted(set(seq))
    y = np.array([int(i.startswith("poisoned")) for i in ids])
    return ids, y, np.array([s[seq == i].max() for i in ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="experiments/seeds")
    ap.add_argument("--reference", default="tgn")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default="experiments/seeds/summary.json")
    args = ap.parse_args()

    runs = defaultdict(dict)
    ids_ref = y = None
    for npz in sorted(Path(args.root).glob("*_s*/window_scores_test_b1.npz")):
        m = re.match(r"(.+)_s(\d+)$", npz.parent.name)
        ids, yy, s = sequence_scores(npz)
        if ids_ref is None:
            ids_ref, y = ids, yy
        assert ids == ids_ref, f"{npz}: different test sequences"
        runs[m.group(1)][int(m.group(2))] = s
    if not runs:
        raise SystemExit(f"no completed runs under {args.root}")

    rng = np.random.default_rng(0)
    boots = []
    while len(boots) < args.n_boot:
        i = rng.choice(len(y), size=len(y), replace=True)
        if 0 < y[i].sum() < len(i):
            boots.append(i)

    out = {"n_test_sequences": int(len(y)), "n_poisoned": int(y.sum()), "configs": {}}
    ref = runs.get(args.reference, {})
    for name, by_seed in sorted(runs.items()):
        seeds = sorted(by_seed)
        au = [roc_auc_score(y, by_seed[s]) for s in seeds]
        apv = [average_precision_score(y, by_seed[s]) for s in seeds]
        bmean = [np.mean([roc_auc_score(y[i], by_seed[s][i]) for s in seeds]) for i in boots]
        row = {"seeds": seeds, "auroc": au, "ap": apv,
               "auroc_mean": float(np.mean(au)), "ap_mean": float(np.mean(apv)),
               "auroc_sd": float(np.std(au, ddof=1)) if len(au) > 1 else None,
               "ap_sd": float(np.std(apv, ddof=1)) if len(apv) > 1 else None,
               "auroc_mean_ci95": [float(np.percentile(bmean, 2.5)),
                                   float(np.percentile(bmean, 97.5))]}
        shared = [s for s in seeds if s in ref]
        if name != args.reference and shared:
            d = [np.mean([roc_auc_score(y[i], ref[s][i]) - roc_auc_score(y[i], by_seed[s][i])
                          for s in shared]) for i in boots]
            row["reference_minus_this"] = {
                "seeds": shared,
                "auroc": float(np.mean([roc_auc_score(y, ref[s]) - roc_auc_score(y, by_seed[s])
                                        for s in shared])),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        out["configs"][name] = row
        line = (f"{name:14s} n={len(seeds)}  AUROC {row['auroc_mean']:.4f}"
                + (f" +/- {row['auroc_sd']:.4f}" if row["auroc_sd"] is not None else "")
                + f"  [{row['auroc_mean_ci95'][0]:.4f}, {row['auroc_mean_ci95'][1]:.4f}]"
                + f"  AP {row['ap_mean']:.4f}")
        if "reference_minus_this" in row:
            r = row["reference_minus_this"]
            line += f"   {args.reference} minus this {r['auroc']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        print(line)
    if args.reference in out["configs"] and out["configs"][args.reference]["auroc_sd"]:
        out["noise_floor_auroc_sd"] = out["configs"][args.reference]["auroc_sd"]
        print(f"\nnoise floor (detector seed-to-seed SD): {out['noise_floor_auroc_sd']:.4f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
