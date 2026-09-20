"""
Calibrated detection latency: how many snapshots after a poison first appears does a
detector actually fire, at a stated false-alarm rate?
==========================================================================================

The existing implementation in generate_eval_results.py cannot answer this. It hard-codes

    threshold = 0.5

which is not an operating point anyone chose -- it is wherever the sigmoid happened to
land -- and it never scores a single benign sequence, so no false-alarm rate is measured
at all. A latency quoted at an unknown FPR is uninterpretable: any detector can be made
to look fast by lowering its threshold until it fires on everything. It also drops
poisoned sequences that never fire, which biases the mean downwards -- a detector that
catches 10% of attacks quickly scores better than one that catches 90% slowly.

This replaces it with four changes:

1.  THE THRESHOLD IS CALIBRATED, NOT ASSUMED.  Each detector gets the threshold that
    produces a target false-positive rate on benign VALIDATION windows. The realised FPR
    on benign TEST windows is reported alongside, so the reader can see whether the
    calibration transferred.

2.  BOTH DETECTORS ARE TREATED IDENTICALLY.  The nine-feature logistic regression and the
    T-GNN are both fit on TRAIN, both calibrated on VAL, both evaluated on TEST. (This
    differs from complementarity.py, which fits its combiner on VAL because it needs the
    T-GNN's scores to be out-of-sample; there is no combiner here, so train-fitting is
    available and gives the stronger baseline.)

3.  NON-DETECTIONS ARE KEPT.  A poisoned sequence that never crosses the threshold is
    censored, not discarded. The headline is therefore a curve -- the fraction of attacks
    caught within k snapshots -- which is defined whether or not a detector ever fires.
    A mean over detections alone is reported only as a secondary number, with its
    denominator stated.

4.  ALARMS BEFORE THE ATTACK EXISTS ARE FALSE ALARMS.  Only windows whose final snapshot
    is at or after the first poisoned snapshot can count as a detection. Windows that
    fire earlier are counted separately as pre-onset false alarms.

Usage:
    python -m src.experiments.detection_latency --data data/raw/ap_sequences \
        --run-dir experiments/run_ap_base
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.dataset import MemoryTrajectoryDataset
from src.models.tgn_detector import TGNDetector
from src.experiments.complementarity import structural_features, tgn_scores


def first_poison_snapshot(data_dir: Path, seq_id: str):
    """Index of the first snapshot whose graph_label is 1, or None if never poisoned."""
    snaps = sorted((data_dir / seq_id).glob("snapshot_*.npz"))
    for i, p in enumerate(snaps):
        with np.load(p) as z:
            if int(np.asarray(z["graph_label"]).reshape(-1)[0]) == 1:
                return i, len(snaps)
    return None, len(snaps)


def calibrate(scores_benign, target_fpr):
    """Smallest threshold whose false-alarm rate on these benign scores is <= target."""
    return float(np.quantile(scores_benign, 1.0 - target_fpr))


def latency_table(seq_ids, seq_col, start_col, scores, T, onset, thr):
    """Per-sequence outcome at a fixed threshold.

    Returns a list of dicts with latency (None if censored) and the pre-onset alarm count.
    """
    rows = []
    for sid in seq_ids:
        m = seq_col == sid
        if not m.any():
            continue
        fp = onset[sid]
        if fp is None:
            continue
        snap_idx = start_col[m] + T - 1
        sc = scores[m]
        order = np.argsort(snap_idx)
        snap_idx, sc = snap_idx[order], sc[order]

        pre = snap_idx < fp
        eligible = ~pre
        if not eligible.any():
            continue                      # no window ever ends at or after the onset
        fired = np.flatnonzero(eligible & (sc >= thr))
        rows.append({
            "sequence_id": str(sid),
            "first_poison_snapshot": int(fp),
            "latency": int(snap_idx[fired[0]] - fp) if fired.size else None,
            "pre_onset_alarms": int((pre & (sc >= thr)).sum()),
            "n_eligible": int(eligible.sum()),
        })
    return rows


def caught_within(rows, k):
    """Fraction of poisoned sequences detected within k snapshots of onset."""
    if not rows:
        return float("nan")
    hit = sum(1 for r in rows if r["latency"] is not None and r["latency"] <= k)
    return hit / len(rows)


def paired_curve_bootstrap(rows_a, rows_b, ks, n_boot=2000, seed=0):
    """Cluster bootstrap over sequences on caught_within(a) - caught_within(b)."""
    rng = np.random.default_rng(seed)
    by_id_a = {r["sequence_id"]: r for r in rows_a}
    by_id_b = {r["sequence_id"]: r for r in rows_b}
    shared = [s for s in by_id_a if s in by_id_b]
    draws = [rng.choice(len(shared), size=len(shared), replace=True) for _ in range(n_boot)]
    out = {}
    for k in ks:
        d = []
        for pick in draws:
            ra = [by_id_a[shared[i]] for i in pick]
            rb = [by_id_b[shared[i]] for i in pick]
            d.append(caught_within(ra, k) - caught_within(rb, k))
        d = np.array(d)
        out[int(k)] = {
            "delta_mean": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        }
    return out, len(shared)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="phase2_best.pt")
    ap.add_argument("--scale-readout", action="store_true")
    ap.add_argument("--cluster-k", type=int, default=12)
    ap.add_argument("--target-fpr", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    T = cfg["training"]["temporal_window"]
    emb = cfg["simulation"]["embedding_dim"]
    thr_edge = cfg["simulation"]["semantic_edge_threshold"]
    cfg["model"]["node_meta_dim"] = 3 + 4
    cfg["model"]["scale_readout"] = bool(args.scale_readout)

    data_dir = Path(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGNDetector(cfg).to(device)
    model.load_state_dict(torch.load(Path(args.run_dir) / args.checkpoint,
                                     map_location=device, weights_only=True))
    model.eval()

    meta = {m["sequence_id"]: m for m in json.load(open(data_dir / "metadata.json"))}

    splits = {}
    for name in ("train", "val", "test"):
        ds = MemoryTrajectoryDataset(args.data, split=name, temporal_window=T,
                                     embedding_dim=emb, structural_features=True)
        X, y, seq = structural_features(ds, emb, thr_edge, args.cluster_k)
        g = tgn_scores(model, ds, device, cfg["training"]["batch_size"])
        start = np.array([w[1] for w in ds.windows], dtype=int)
        assert len(g) == len(y) == len(start), f"{name}: length mismatch"
        splits[name] = {"X": X, "y": y, "seq": seq, "tgn": g, "start": start}
        print(f"{name:>5}: {len(y)} windows, {int(y.sum())} positive, "
              f"{len(np.unique(seq))} sequences")

    tr, va, te = splits["train"], splits["val"], splits["test"]

    # Both detectors: fit on train, calibrate on val, evaluate on test.
    sc = StandardScaler().fit(tr["X"])
    lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(
        sc.transform(tr["X"]), tr["y"])
    for d in (va, te):
        d["lr"] = lr.decision_function(sc.transform(d["X"]))

    onset = {}
    for sid in np.unique(np.concatenate([va["seq"], te["seq"]])):
        onset[sid], _ = first_poison_snapshot(data_dir, sid)

    test_seqs = list(np.unique(te["seq"]))
    poisoned = [s for s in test_seqs if onset[s] is not None]
    benign = [s for s in test_seqs if onset[s] is None]
    print(f"\ntest sequences: {len(poisoned)} poisoned, {len(benign)} never-poisoned")

    results = {"n_test_poisoned": len(poisoned), "n_test_benign": len(benign),
               "temporal_window": T, "detectors": {}}
    max_k = T
    ks = list(range(0, max_k + 1))

    for target in args.target_fpr:
        print("\n" + "=" * 78)
        print(f"target false-positive rate: {target:.0%} "
              f"(calibrated on benign VAL windows)")
        print("=" * 78)
        per_det = {}
        for key, label in (("lr", "9 features (LR)"), ("tgn", "T-GNN")):
            benign_val = va[key][va["y"] == 0]
            thr = calibrate(benign_val, target)
            realised = float((te[key][te["y"] == 0] >= thr).mean())
            rows = latency_table(poisoned, te["seq"], te["start"], te[key], T, onset, thr)
            det = [r for r in rows if r["latency"] is not None]
            lat = np.array([r["latency"] for r in det], float)
            per_det[key] = {
                "label": label, "threshold": thr,
                "val_benign_fpr_window": float((benign_val >= thr).mean()),
                "test_benign_fpr_window": realised,
                "n_poisoned_evaluated": len(rows),
                "n_detected": len(det),
                "detection_rate": len(det) / max(len(rows), 1),
                "mean_latency_detected_only": float(lat.mean()) if lat.size else None,
                "median_latency_detected_only": float(np.median(lat)) if lat.size else None,
                "mean_pre_onset_alarms": float(np.mean([r["pre_onset_alarms"] for r in rows]))
                                         if rows else None,
                "caught_within": {int(k): caught_within(rows, k) for k in ks},
                "rows": rows,
            }
            r = per_det[key]
            print(f"\n  {label}")
            print(f"    threshold {thr:+.4f}   val benign FPR {r['val_benign_fpr_window']:.3f}"
                  f"   test benign FPR {r['test_benign_fpr_window']:.3f}")
            print(f"    detected {len(det)}/{len(rows)} poisoned sequences "
                  f"({r['detection_rate']:.1%})")
            if lat.size:
                print(f"    latency among detected: median {np.median(lat):.1f}, "
                      f"mean {lat.mean():.2f} snapshots (n={lat.size})")
            print(f"    pre-onset false alarms per sequence: "
                  f"{r['mean_pre_onset_alarms']:.2f}")
            print("    caught within k snapshots of onset:")
            print("      k      " + "".join(f"{k:>7}" for k in ks[:8]))
            print("      frac   " + "".join(f"{r['caught_within'][k]:>7.3f}" for k in ks[:8]))

        boot, n_shared = paired_curve_bootstrap(per_det["tgn"]["rows"], per_det["lr"]["rows"],
                                                [0, 2, 5, max_k], n_boot=args.n_boot)
        print(f"\n  paired cluster bootstrap, T-GNN - 9features "
              f"({args.n_boot} resamples over {n_shared} poisoned sequences)")
        for k, v in boot.items():
            sig = "significant" if (v["ci95"][0] > 0 or v["ci95"][1] < 0) else "NOT significant"
            print(f"    caught within k={k:<3} delta={v['delta_mean']:+.4f}  "
                  f"95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {sig}")

        for v in per_det.values():
            v.pop("rows")
        results["detectors"][f"fpr_{target:g}"] = {"per_detector": per_det,
                                                   "tgn_minus_lr": boot}

    out = Path(args.output) if args.output else Path(args.run_dir) / "detection_latency.json"
    json.dump(results, open(out, "w"), indent=2)

    np.savez(Path(args.run_dir) / "window_scores_test.npz",
             seq=te["seq"], start=te["start"], y=te["y"],
             tgn=te["tgn"], lr=te["lr"],
             num_injected=np.array([meta[s]["num_injected"] for s in te["seq"]]),
             attack_type=np.array([meta[s]["attack_type"] or "clean" for s in te["seq"]]))
    print(f"\nSaved to {out}")
    print(f"Per-window test scores: {Path(args.run_dir) / 'window_scores_test.npz'}")


if __name__ == "__main__":
    main()
