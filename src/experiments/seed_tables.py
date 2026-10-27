"""
Table 4.2 (latency) and the attack-size curve, averaged over detector seeds.
============================================================================

Offline: reads each seed's window_scores_{val,test}_b1.npz (written by train_variant.py) and
the nine-feature regression's test scores and calibrated thresholds from the as-specified
latency run (the regression does not depend on the detector's seed). The latency protocol is
detection_latency.py's, unchanged: threshold calibrated on benign validation windows to a
target false-alarm rate, censored non-detections kept, pre-onset alarms counted as false.

Usage:
    python -m src.experiments.seed_tables --seeds-root experiments/seeds --model tgn \
        --lr-scores experiments/audit/latency_window_scores_perwindow.npz \
        --lr-latency experiments/audit/detection_latency_perwindow.json \
        --output experiments/seeds/seed_tables.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

T = 10


def onsets(cache_dir: Path, seq_ids):
    out = {}
    for sid in seq_ids:
        with np.load(cache_dir / f"{sid}.npz") as z:
            g = z["graph_label"]
        out[sid] = int(np.flatnonzero(g)[0]) if g.any() else None
    return out


def latency_rows(seq, start, scores, onset, thr):
    rows = {}
    for sid in np.unique(seq):
        fp = onset[sid]
        if fp is None:
            continue
        m = seq == sid
        idx = start[m] + T - 1
        o = np.argsort(idx)
        idx, sc = idx[o], scores[m][o]
        eligible = idx >= fp
        if not eligible.any():
            continue
        fired = np.flatnonzero(eligible & (sc >= thr))
        rows[sid] = {"latency": int(idx[fired[0]] - fp) if fired.size else None,
                     "pre": int(((~eligible) & (sc >= thr)).sum())}
    return rows


def caught(rows, ids, k):
    return np.mean([rows[i]["latency"] is not None and rows[i]["latency"] <= k for i in ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-root", default="experiments/seeds")
    ap.add_argument("--model", default="tgn")
    ap.add_argument("--data", default="data/raw/ap_burst2k")
    ap.add_argument("--lr-scores", required=True)
    ap.add_argument("--lr-latency", required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    runs = sorted(Path(args.seeds_root).glob(f"{args.model}_s*/window_scores_test_b1.npz"))
    lr = np.load(args.lr_scores, allow_pickle=True)
    lr_lat = json.load(open(args.lr_latency))
    meta = {m["sequence_id"]: m for m in json.load(open(Path(args.data) / "metadata.json"))}
    first = np.load(runs[0], allow_pickle=True)
    seq, start = np.asarray(first["seq"]), np.asarray(first["start"])
    assert (np.asarray(lr["seq"]) == seq).all() and (np.asarray(lr["start"]) == start).all()
    onset = onsets(Path(args.data) / "_seqcache" / "struct", np.unique(seq))
    poisoned = sorted(s for s in np.unique(seq) if onset[s] is not None)
    rng = np.random.default_rng(0)
    boots = [rng.choice(len(poisoned), len(poisoned), replace=True) for _ in range(args.n_boot)]

    out = {"seeds": [str(r.parent.name) for r in runs], "latency": {}, "size_curve": {}}
    for target in args.targets:
        lr_thr = lr_lat["detectors"][f"fpr_{target:g}"]["per_detector"]["lr"]["threshold"]
        lr_rows = latency_rows(seq, start, np.asarray(lr["lr"]), onset, lr_thr)
        per_seed = []
        for r in runs:
            te = np.load(r, allow_pickle=True)
            va = np.load(r.parent / "window_scores_val_b1.npz", allow_pickle=True)
            thr = float(np.quantile(va["score"][va["y"] == 0], 1 - target))
            rows = latency_rows(seq, start, np.asarray(te["score"]), onset, thr)
            lat = [rows[i]["latency"] for i in poisoned if rows[i]["latency"] is not None]
            per_seed.append({"rows": rows, "threshold": thr,
                             "test_benign_fpr": float((te["score"][te["y"] == 0] >= thr).mean()),
                             "detected": len(lat), "median": float(np.median(lat)),
                             "mean": float(np.mean(lat)),
                             "pre_onset": float(np.mean([rows[i]["pre"] for i in poisoned]))})
        res = {"lr": {"detected": sum(lr_rows[i]["latency"] is not None for i in poisoned)},
               "detector": {k: [p[k] for p in per_seed] for k in
                            ("threshold", "test_benign_fpr", "detected", "median", "mean",
                             "pre_onset")},
               "caught_within_difference": {}}
        for k in (0, 2, 5, 10):
            point = float(np.mean([caught(p["rows"], poisoned, k) - caught(lr_rows, poisoned, k)
                                   for p in per_seed]))
            dist = []
            for b in boots:
                ids = [poisoned[i] for i in b]
                dist.append(np.mean([caught(p["rows"], ids, k) - caught(lr_rows, ids, k)
                                     for p in per_seed]))
            res["caught_within_difference"][k] = {
                "detector_mean": float(np.mean([caught(p["rows"], poisoned, k) for p in per_seed])),
                "lr": float(caught(lr_rows, poisoned, k)), "delta": point,
                "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]}
        out["latency"][f"{target:g}"] = res
        d = res["detector"]
        print(f"target {target:.0%}: detector detects {np.mean(d['detected']):.1f}/200 "
              f"(per seed {d['detected']}), realised FPR {np.mean(d['test_benign_fpr']):.2%}, "
              f"median {np.mean(d['median']):.1f}; regression {res['lr']['detected']}/200")
        for k, v in res["caught_within_difference"].items():
            print(f"   within {k:2d}: detector {v['detector_mean']:.3f}  regression {v['lr']:.3f}  "
                  f"difference {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")

    # attack size: sequence AUROC of poisoned-with-k against all clean, averaged over seeds
    ids = sorted(set(seq))
    y = np.array([int(i.startswith("poisoned")) for i in ids])
    k_of = np.array([meta[i]["num_injected"] for i in ids])
    seq_scores = []
    for r in runs:
        z = np.load(r, allow_pickle=True)
        seq_scores.append(np.array([z["score"][z["seq"] == i].max() for i in ids]))
    for lo, hi in [(k, k) for k in range(5, 21)] + [(5, 8), (9, 12), (13, 16), (17, 20)]:
        m = (y == 0) | ((y == 1) & (k_of >= lo) & (k_of <= hi))
        vals = [roc_auc_score(y[m], s[m]) for s in seq_scores]
        key = f"{lo}" if lo == hi else f"{lo}-{hi}"
        out["size_curve"][key] = {"n_poisoned": int(((y == 1) & m).sum()),
                                  "auroc_mean": float(np.mean(vals)),
                                  "auroc_sd": float(np.std(vals, ddof=1))}
    print("size curve (sequence AUROC, mean over seeds):",
          {k: round(v["auroc_mean"], 3) for k, v in out["size_curve"].items()})

    # the same sizes against clean controls of the same cluster size, which is the comparison
    # the matched-control design intends; pooled clean controls include large benign clusters
    # that outscore small attacks
    k_any = np.array([meta[i]["num_injected"] if y[j] else meta[i]["benign_burst"]
                      for j, i in enumerate(ids)])
    out["size_matched"] = {}
    for lo, hi in ((5, 8), (9, 12), (13, 16), (17, 20)):
        m = (k_any >= lo) & (k_any <= hi)
        vals = [roc_auc_score(y[m], s[m]) for s in seq_scores]
        out["size_matched"][f"{lo}-{hi}"] = {
            "n_clean": int((m & (y == 0)).sum()), "n_poisoned": int((m & (y == 1)).sum()),
            "auroc_mean": float(np.mean(vals)), "auroc_sd": float(np.std(vals, ddof=1))}

    # Table 4.1 and the acceptance check, per seed, as specified (batch 1)
    from sklearn.metrics import average_precision_score, f1_score, roc_curve
    t41 = {}
    for split in ("val", "test"):
        for r in runs:
            z = np.load(r.parent / f"window_scores_{split}_b1.npz", allow_pickle=True)
            yy, ss = z["y"], z["score"]
            fpr, tpr, _ = roc_curve(yy, ss)
            sid = sorted(set(z["seq"]))
            ys = np.array([int(i.startswith("poisoned")) for i in sid])
            sm = np.array([ss[z["seq"] == i].max() for i in sid])
            row = {"window_auroc": roc_auc_score(yy, ss), "window_ap": average_precision_score(yy, ss),
                   "sequence_auroc": roc_auc_score(ys, sm), "sequence_ap": average_precision_score(ys, sm)}
            if split == "test":
                row.update({"f1_at_0.5": f1_score(yy, ss >= 0.5),
                            "fpr_at_tpr_0.9": float(fpr[np.argmin(np.abs(tpr - 0.9))]),
                            "node_auroc": roc_auc_score(z["node_label"], z["node_score"]),
                            "node_ap": average_precision_score(z["node_label"], z["node_score"])})
            for key, v in row.items():
                t41.setdefault(split, {}).setdefault(key, []).append(float(v))
    out["table_4_1"] = {split: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                    "per_seed": v} for k, v in d.items()}
                        for split, d in t41.items()}
    for split, d in out["table_4_1"].items():
        print(split, {k: f"{v['mean']:.4f}+/-{v['sd']:.4f}" for k, v in d.items()})
    for t in out["latency"].values():
        t.pop("rows", None)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
