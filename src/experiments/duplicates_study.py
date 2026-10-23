"""
R1: is the detector reading the rate a cluster arrives, or duplicated records?
=============================================================================

On ap_burst2k the attacker (and the matched benign control) draws k records from the 8 nearest
neighbours of one entry, with replacement once k > 8, so every cluster of 9 or more holds exact
copies (cosine 1.0). The E4 grid showed the highest detection scores need those copies. This
study builds ap_burst2k_nodup with the generator's --no-duplicates switch (same size, split,
seed and poisoning rate) and measures, at sequence level on the 500 test sequences:

  1. the four train/test cells for the full detector: trained with or without duplicates,
     tested with or without (trained-with rows: the reported checkpoint and the five seeds in
     experiments/seeds; trained-without rows: seeds 1 to 3 trained here);
  2. the temporal gap (detector minus static GAT) on each benchmark;
  3. logistic regressions on a duplicate feature set alone, the 13 sequence features alone,
     and both together, on each benchmark.

Duplicate features, per sequence, from the final snapshot and from every 10-snapshot window:
  dup_final               records in the final snapshot with at least one other record at
                          cosine similarity >= 0.99
  dup_window_growth_max   the largest increase in that count within any window (end minus
                          start snapshot), i.e. how many duplicated records arrived in one window
  dup_max_jump            the largest increase between two consecutive snapshots, i.e. a burst
                          of duplicated records; on ap_burst2k this alone scores 0.707 while the
                          two counts above sit at chance (the matched controls carry as many
                          copies), so leaving it out would understate the duplicate explanation

Subcommands:
  compare     what differs between the two benchmarks (sizes, arrival, duplication, background)
  features    duplicate features for a benchmark, cached as duplicate_features.json
  run         the whole R1 pipeline, resumable, keeping the laptop awake, with STATUS.md
  summarize   SUMMARY.md and summary.json from whatever has finished

Usage:
    python -m src.experiments.duplicates_study run
    python -m src.experiments.duplicates_study summarize
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DUP = ROOT / "data" / "raw" / "ap_burst2k"
NODUP = ROOT / "data" / "raw" / "ap_burst2k_nodup"
OUT = ROOT / "experiments" / "duplicates"
RUNS = ROOT / "experiments" / "seeds_nodup"   # main-benchmark runs (moved from duplicates/runs, 19 Oct)
SCORES = OUT / "scores"
SEEDS_DUP = ROOT / "experiments" / "seeds"
REPORTED = ROOT / "experiments" / "run_burst2k_base"
NEAR = 0.99
T = 10
PY = sys.executable


# --------------------------------------------------------------------------- helpers
def status(lines):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = f"{dt.datetime.now():%a %d %b %H:%M}"
    (OUT / "STATUS.md").write_text(f"# R1 duplicates study: status (updated {stamp})\n\n"
                                   + "\n".join(lines) + "\n", encoding="utf-8")


def keep_awake(on):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 if on else 0))


def sh(cmd, log=None):
    print(f"[{dt.datetime.now():%H:%M}] " + " ".join(str(c) for c in cmd), flush=True)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, stdout=f,
                           stderr=subprocess.STDOUT)
    else:
        subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def cache_file(bench: Path, sid: str) -> Path:
    return bench / "_seqcache" / "struct" / f"{sid}.npz"


def dup_counts(emb: np.ndarray, n_nodes: np.ndarray) -> np.ndarray:
    """Per snapshot: number of present records with a present partner at cosine >= NEAR."""
    u = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    s = u @ u.T
    np.fill_diagonal(s, -1.0)
    near = s >= NEAR
    big = len(emb) + 1
    first_partner = np.where(near.any(axis=1), np.argmax(near, axis=1), big)
    # a record counts once both it and its earliest partner are present
    appears = np.maximum(np.arange(len(emb)), first_partner)
    return np.array([(appears < n).sum() for n in n_nodes])


def duplicate_features(bench: Path, recompute=False):
    path = bench / "duplicate_features.json"
    if path.exists() and not recompute:
        return json.load(open(path))
    splits = json.load(open(bench / "splits.json"))
    out = {}
    for part, ids in splits.items():
        rows = []
        for sid in ids:
            with np.load(cache_file(bench, sid)) as z:
                c = dup_counts(z["emb"], z["n_nodes"])
            growth = [int(c[e] - c[e - T + 1]) for e in range(T - 1, len(c))]
            rows.append({"sequence_id": sid, "dup_final": int(c[-1]),
                         "dup_window_growth_max": int(max(growth)),
                         "dup_max_jump": int(np.diff(c).max()) if len(c) > 1 else 0})
        out[part] = rows
    json.dump(out, open(path, "w"))
    return out


def replay_slots(sid: str, seed=42, n=200):
    """Interaction slots of a sequence's cluster, from the generator's per-sequence RNG.

    Validated against stored labels in the audit (0 mismatches). The per-sequence RNG drives
    size, start and slots, so they do not depend on --no-duplicates.
    """
    i = int(sid.split("_")[-1])
    clean = sid.startswith("clean")
    rng = np.random.default_rng(seed + i if clean else seed + 10000 + i)
    k = int(rng.integers(5, 21))
    start = int(rng.integers(40, max(41, n - k - 10)))
    width = min(n - start, max(120, k)) if clean else min(n - start, k)
    return k, np.sort(start + rng.choice(width, size=min(k, width), replace=False))


# --------------------------------------------------------------------------- compare
def compare():
    md, mn = (json.load(open(b / "metadata.json")) for b in (DUP, NODUP))
    sd, sn = (json.load(open(b / "splits.json")) for b in (DUP, NODUP))
    res = {"same_ids": [m["sequence_id"] for m in md] == [m["sequence_id"] for m in mn],
           "same_splits": sd == sn,
           "same_sizes": all(a["num_injected"] == b["num_injected"]
                             and a["benign_burst"] == b["benign_burst"] for a, b in zip(md, mn))}
    sizes = [m["num_injected"] for m in mn if m["label"] == "poisoned"]
    res["size_range_poisoned"] = [min(sizes), max(sizes)]
    same_slots, dup_clusters = 0, {"dup": 0, "nodup": 0}
    bg_identical, bg_total, n_checked = 0, 0, 0
    for m in md:
        sid = m["sequence_id"]
        k, slots = replay_slots(sid)
        with np.load(cache_file(DUP, sid)) as a, np.load(cache_file(NODUP, sid)) as b:
            if m["label"] == "poisoned":
                same_slots += int(np.array_equal(np.flatnonzero(a["labels"]),
                                                 np.flatnonzero(b["labels"])))
            for name, z in (("dup", a), ("nodup", b)):
                cl = z["emb"][slots]
                dup_clusters[name] += int(len(np.unique(cl.round(6), axis=0)) < len(cl))
            mask = np.ones(len(a["emb"]), bool)
            mask[slots] = False
            bg_identical += int((a["emb"][mask] == b["emb"][mask]).all(axis=1).sum())
            bg_total += int(mask.sum())
        n_checked += 1
    n_pois = sum(m["label"] == "poisoned" for m in md)
    res.update({"poisoned_with_identical_slots": f"{same_slots}/{n_pois}",
                "clusters_containing_duplicates": {k: f"{v}/{n_checked}"
                                                   for k, v in dup_clusters.items()},
                "background_records_identical": f"{bg_identical}/{bg_total}"})
    json.dump(res, open(OUT / "benchmark_comparison.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------- metrics
def seq_scores(npz: Path, order):
    z = np.load(npz, allow_pickle=True)
    seq = np.asarray(z["seq"] if "seq" in z.files else z["window_sequence_id"])
    s = np.asarray(z["score"] if "score" in z.files else z["tgn"], float)
    return np.array([s[seq == i].max() for i in order])


def boots(y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_boot:
        i = rng.choice(len(y), len(y), replace=True)
        if 0 < y[i].sum() < len(i):
            out.append(i)
    return out


def group_stats(y, runs, B):
    from sklearn.metrics import average_precision_score, roc_auc_score
    au = [roc_auc_score(y, s) for s in runs]
    ap = [average_precision_score(y, s) for s in runs]
    dist = [np.mean([roc_auc_score(y[i], s[i]) for s in runs]) for i in B]
    return {"n_runs": len(runs), "auroc_per_run": au, "auroc_mean": float(np.mean(au)),
            "auroc_sd": float(np.std(au, ddof=1)) if len(au) > 1 else None,
            "ap_mean": float(np.mean(ap)),
            "auroc_ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]}


def group_diff(y, a_runs, b_runs, B):
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = {}
    for m, fn in (("auroc", roc_auc_score), ("ap", average_precision_score)):
        point = np.mean([fn(y, s) for s in a_runs]) - np.mean([fn(y, s) for s in b_runs])
        dist = [np.mean([fn(y[i], s[i]) for s in a_runs]) - np.mean([fn(y[i], s[i]) for s in b_runs])
                for i in B]
        lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
        out[m] = {"delta": float(point), "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}
    return out


def oracle_scores(bench: Path):
    """Sequence-level logistic regressions fitted on train, scored on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from src.experiments.sequence_eval import FINAL_FEATS, TEMPORAL_FEATS
    meta = {m["sequence_id"]: m for m in json.load(open(bench / "metadata.json"))}
    feats13 = json.load(open(bench / "sequence_features.json"))
    dupf = duplicate_features(bench)
    dkeys = ["dup_final", "dup_window_growth_max", "dup_max_jump"]

    def table(part):
        d = {r["sequence_id"]: r for r in dupf[part]}
        rows = feats13[part]
        ids = [r["sequence_id"] for r in rows]
        X13 = np.array([[r[f] for f in FINAL_FEATS + TEMPORAL_FEATS] for r in rows], float)
        Xd = np.array([[d[i][k] for k in dkeys] for i in ids], float)
        y = np.array([int(meta[i]["label"] == "poisoned") for i in ids])
        return ids, X13, Xd, y

    ids_tr, X13tr, Xdtr, ytr = table("train")
    ids_te, X13te, Xdte, yte = table("test")
    out = {}
    for name, (A, Bm) in {"duplicate features": (Xdtr, Xdte),
                          "13 features": (X13tr, X13te),
                          "13 + duplicate features": (np.hstack([X13tr, Xdtr]),
                                                      np.hstack([X13te, Xdte]))}.items():
        sc = StandardScaler().fit(A)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(A), ytr)
        s = clf.decision_function(sc.transform(Bm))
        out[name] = dict(zip(ids_te, s))
    return out


# --------------------------------------------------------------------------- summarize
def summarize():
    meta = {m["sequence_id"]: m for m in json.load(open(DUP / "metadata.json"))}
    order = sorted(json.load(open(DUP / "splits.json"))["test"])
    y = np.array([int(meta[i]["label"] == "poisoned") for i in order])
    B = boots(y)
    have = lambda p: Path(p).exists()

    sets = {
        ("dup", "dup"): [SEEDS_DUP / f"tgn_s{s}" / "window_scores_test_b1.npz" for s in range(1, 6)],
        ("dup", "nodup"): [SCORES / f"tgn_s{s}_dup_on_nodup.npz" for s in range(1, 6)],
        ("nodup", "nodup"): [RUNS / f"tgn_s{s}" / "window_scores_test_b1.npz" for s in range(1, 6)],
        ("nodup", "dup"): [SCORES / f"tgn_s{s}_nodup_on_dup.npz" for s in range(1, 6)],
    }
    static = {"dup": [SEEDS_DUP / f"static_attn_s{s}" / "window_scores_test_b1.npz" for s in range(1, 6)],
              "nodup": [RUNS / f"static_attn_s{s}" / "window_scores_test_b1.npz" for s in range(1, 6)]}
    reported = {"dup": ROOT / "experiments" / "audit" / "base_test_b1.npz",
                "nodup": SCORES / "reported_dup_on_nodup.npz"}

    runs = {k: [seq_scores(p, order) for p in v if have(p)] for k, v in sets.items()}
    sruns = {k: [seq_scores(p, order) for p in v if have(p)] for k, v in static.items()}
    res = {"test_sequences": len(y), "poisoned": int(y.sum()), "cells": {}, "reported": {},
           "temporal_gap": {}, "oracles": {}, "train_effect": {}}
    for (tr, te), r in runs.items():
        if r:
            res["cells"][f"train {tr} / test {te}"] = group_stats(y, r, B)
    for te, p in reported.items():
        if have(p):
            res["reported"][f"test {te}"] = group_stats(y, [seq_scores(p, order)], B)
    for te in ("dup", "nodup"):
        if runs[("dup", te)] and runs[("nodup", te)]:
            res["train_effect"][f"test {te}: trained dup minus trained nodup"] = group_diff(
                y, runs[("dup", te)], runs[("nodup", te)], B)
    for bench in ("dup", "nodup"):
        det, st = runs[(bench, bench)], sruns[bench]
        if det and st:
            res["temporal_gap"][bench] = {"static": group_stats(y, st, B),
                                          "detector_minus_static": group_diff(y, det, st, B)}
    for bench, path in (("dup", DUP), ("nodup", NODUP)):
        if not (path / "sequence_features.json").exists() or not (path / "_seqcache").exists():
            continue
        orc = oracle_scores(path)
        det = runs[(bench, bench)]
        rows = {}
        for name, d in orc.items():
            s = np.array([d[i] for i in order])
            row = group_stats(y, [s], B)
            if det:
                row["detector_minus_this"] = group_diff(y, det, [s], B)
            rows[name] = row
        res["oracles"][bench] = rows
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    write_summary_md(res)
    print((OUT / "SUMMARY.md").read_text(encoding="utf-8"))


def fmt_stats(s):
    sd = f" ({s['auroc_sd']:.4f})" if s.get("auroc_sd") is not None else ""
    return (f"{s['auroc_mean']:.4f}{sd} | {s['ap_mean']:.4f} | "
            f"[{s['auroc_ci95'][0]:.4f}, {s['auroc_ci95'][1]:.4f}] | {s['n_runs']}")


def fmt_diff(d, m="auroc"):
    x = d[m]
    return f"{x['delta']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"


def write_summary_md(res):
    L = ["# R1 duplicates study: results", "",
         f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by "
         "`python -m src.experiments.duplicates_study summarize` from `summary.json`.",
         f"Test partition: {res['test_sequences']} sequences ({res['poisoned']} poisoned), "
         "sequence level (window max). AUROC is the mean over runs (SD across runs); the 95% "
         "interval is a bootstrap over sequences of the run-averaged AUROC; differences are "
         "run-averaged and paired over the same test sequences.", ""]
    if (OUT / "benchmark_comparison.json").exists():
        c = json.load(open(OUT / "benchmark_comparison.json"))
        L += ["## The two benchmarks", "",
              f"* Same sequence ids: {c['same_ids']}; same split: {c['same_splits']}; same "
              f"cluster sizes: {c['same_sizes']} (poisoned range {c['size_range_poisoned']}).",
              f"* Poisoned sequences with identical injection slots: "
              f"{c['poisoned_with_identical_slots']}.",
              f"* Clusters containing duplicate entries: with duplicates "
              f"{c['clusters_containing_duplicates']['dup']}, without "
              f"{c['clusters_containing_duplicates']['nodup']}.",
              f"* Background (non-cluster) records identical: "
              f"{c['background_records_identical']}.", ""]
    L += ["## Full detector, four cells", "",
          "| Trained on / tested on | AUROC mean (SD) | AP | 95% CI | Runs |", "|---|---|---|---|---|"]
    for k, s in res["cells"].items():
        L.append(f"| {k} | {fmt_stats(s)} |")
    for k, s in res["reported"].items():
        L.append(f"| reported checkpoint (seed 42, trained dup) / {k} | {fmt_stats(s)} |")
    if res["train_effect"]:
        L += ["", "| Same test set | AUROC difference [95% CI] | AP difference [95% CI] |",
              "|---|---|---|"]
        for k, d in res["train_effect"].items():
            L.append(f"| {k} | {fmt_diff(d)} | {fmt_diff(d, 'ap')} |")
    L += ["", "## Temporal gap (detector minus static GAT, both trained and tested on the same benchmark)", "",
          "| Benchmark | Static GAT AUROC (SD) | Detector minus static, AUROC [95% CI] | AP [95% CI] |",
          "|---|---|---|---|"]
    for b, g in res["temporal_gap"].items():
        st = g["static"]
        sd = f" ({st['auroc_sd']:.4f})" if st["auroc_sd"] is not None else ""
        L.append(f"| {b} | {st['auroc_mean']:.4f}{sd}, {st['n_runs']} runs | "
                 f"{fmt_diff(g['detector_minus_static'])} | "
                 f"{fmt_diff(g['detector_minus_static'], 'ap')} |")
    L += ["", "## Logistic regressions with a duplicate feature", "",
          "Duplicate features: records in the final snapshot with a partner at cosine >= 0.99, "
          "the largest growth in that count within any 10-snapshot window, and the largest "
          "increase between two consecutive snapshots.", "",
          "| Benchmark | Features | AUROC | AP | 95% CI | Detector minus this, AUROC | AP |",
          "|---|---|---|---|---|---|---|"]
    for b, rows in res["oracles"].items():
        for name, r in rows.items():
            dm = r.get("detector_minus_this")
            L.append(f"| {b} | {name} | {r['auroc_mean']:.4f} | {r['ap_mean']:.4f} | "
                     f"[{r['auroc_ci95'][0]:.4f}, {r['auroc_ci95'][1]:.4f}] | "
                     + (f"{fmt_diff(dm)} | {fmt_diff(dm, 'ap')} |" if dm else "not yet | |"))
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- run
def run(n_seeds: int):
    steps = ["1 Generate ap_burst2k_nodup", "2 Cache and features", "3 Train on nodup",
             "4 Cross-score", "5 Summarize"]

    def mark(i, note=""):
        lines = []
        for j, s in enumerate(steps):
            state = "done" if j < i else ("running" if j == i else "queued")
            lines.append(f"- {s}: **{state}**" + (f" ({note})" if j == i and note else ""))
        lines += ["", "Training progress, when step 3 runs: `experiments/seeds_nodup/STATUS.md`."]
        status(lines)

    keep_awake(True)
    try:
        mark(0)
        if not (NODUP / "splits.json").exists():
            sh([PY, "-u", "-m", "src.data_generation.agentpoison_sequences", "--output", NODUP,
                "--num-sequences", "2000", "--matched-control", "1.0", "--benign-spread", "120",
                "--attack-spread-mult", "1.0", "--seed", "42", "--no-duplicates"],
               log=OUT / "generate.log")
        mark(1, "cache, 13 features, duplicate features, benchmark comparison")
        if not (NODUP / "_seqcache" / "struct").exists() or \
                len(list((NODUP / "_seqcache" / "struct").glob("*.npz"))) < 2000:
            sh([PY, "-u", "-m", "src.data_generation.cached_dataset", "build", "--data", NODUP],
               log=OUT / "cache.log")
        if not (NODUP / "sequence_features.json").exists():
            sh([PY, "-u", "-m", "src.experiments.sequence_eval", "--data", NODUP, "--skip-model"],
               log=OUT / "sequence_features.log")
        for b in (DUP, NODUP):
            duplicate_features(b)
        compare()
        mark(2, f"detector and static GAT, seeds 1 to {n_seeds}, about "
                f"{n_seeds * 4.7:.0f} h")
        sh([PY, "-u", "-m", "src.experiments.seed_queue", "--data", NODUP, "--root", RUNS,
            "--seeds", *map(str, range(1, n_seeds + 1)), "--models", "tgn", "static_attn"],
           log=OUT / "train_queue.log")
        mark(3, "every model on the other benchmark's test split")
        SCORES.mkdir(parents=True, exist_ok=True)
        jobs = [(REPORTED, "tgn", NODUP, SCORES / "reported_dup_on_nodup.npz")]
        jobs += [(SEEDS_DUP / f"tgn_s{s}", "tgn", NODUP, SCORES / f"tgn_s{s}_dup_on_nodup.npz")
                 for s in range(1, 6)]
        jobs += [(RUNS / f"tgn_s{s}", "tgn", DUP, SCORES / f"tgn_s{s}_nodup_on_dup.npz")
                 for s in range(1, n_seeds + 1)]
        for run_dir, model, data, out in jobs:
            if not out.exists():
                sh([PY, "-u", "-m", "src.experiments.score_windows", "--data", data,
                    "--run-dir", run_dir, "--model", model, "--cached", "--split", "test",
                    "--batch-size", "4", "--memory-slots", "per_window", "--output", out],
                   log=out.with_suffix(".log"))
        mark(4)
        summarize()
        mark(5)
    finally:
        keep_awake(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["compare", "features", "run", "summarize"])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "compare":
        compare()
    elif args.command == "features":
        for b in (DUP, NODUP):
            duplicate_features(b, recompute=True)
    elif args.command == "run":
        run(args.seeds)
    else:
        summarize()


if __name__ == "__main__":
    main()
