"""
The freeze's full process, rebuilt on the step 5 configuration (experiments/IMPROVEMENT.md, step 6).
====================================================================================================

Same phases and scripts as main_benchmark_plan.py, on detectors trained on whole sequences
(one window of 20 snapshots) with arrival edges (W = 10) and arrival-coherence features:

  A  detector and static GAT to five seeds (seed 1 of each is the step 5 pilot, copied from
     experiments/option5_coherence/ and scored on test here)
  B  online scores, then the R2 rebuild (static-GAT and record-level node metrics, probes,
     Figures 4.1 to 4.3 and the missed example, sweeps with embedding_noise_full added,
     Figures 4.4 to 4.7)
  D1 the headline summaries: detector and static GAT, and the configuration the step 5 rule
     chose (detector + static GAT, rank mean, per seed pair), each run twice and compared byte
     for byte
  C  ablations x 5 seeds: memory only, aggregator only, widened static, mean pooling, no
     structural features; then no coherence features (seed 1 is the step 4 pilot)
  D2 every summary again over all configurations (twice each), the oracle comparison with the
     coherence-feature oracles, the inference profile, and the detector cross-scored on the
     duplicates benchmark

The nine-feature regression's latency scores and thresholds are the freeze's (same data and
windows; they do not depend on the detector). Resumable: every step skips what exists.
Progress: experiments/seeds_coherence/PLAN.md, plus each phase's STATUS.md. Test figures are
written to files and never printed by this driver or the runs it starts.

Usage:
    python -m src.experiments.coherence_plan
    python -m src.experiments.coherence_plan cross-duplicates --output <json>   (D2 summary)
    python -m src.experiments.coherence_plan ensemble --output <json>           (D1 summary)
"""

import datetime as dt
import filecmp
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = "data/raw/ap_burst2k_nodup"
DUP_DATA = "data/raw/ap_burst2k"
S = ROOT / "experiments" / "seeds_coherence"
R = ROOT / "experiments" / "rebuild_coherence"
FROZEN = ROOT / "experiments" / "seeds_nodup"
PY = sys.executable
SEEDS = ["1", "2", "3", "4", "5"]
CONFIG = ["--arrival-window", "10", "--temporal-window", "20", "--epochs-supervised", "66",
          "--score-splits", "val", "test"]
ABLATIONS = ["no_memory", "no_aggregator", "static_wide", "meanpool", "nostruct"]
PILOTS = {"tgn_s1": ROOT / "experiments" / "option5_coherence" / "tgn_s1",
          "static_attn_s1": ROOT / "experiments" / "option5_coherence" / "static_attn_s1",
          "tgn_nocoherence_s1": ROOT / "experiments" / "option3_sequence" / "tgn_s1"}
PHASES = [("A", "Detector and static GAT to five seeds", 21.0),
          ("B", "Online scores and R2 rebuild (records, probes, figures, sweeps)", 18.0),
          ("D1", "Headline summaries (detector, static GAT), each run twice", 0.5),
          ("C", "Ablations x 5 seeds (memory only, aggregator only, widened static, mean pool, "
                "no structural features, no coherence features)", 80.0),
          ("D2", "All summaries twice, oracle comparison, profile, duplicates cross-score", 3.0)]


def keep_awake(on):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 if on else 0))


def plan(current, done, note=""):
    left = sum(h for p, _, h in PHASES if p not in done)
    finish = dt.datetime.now() + dt.timedelta(hours=left)
    lines = [f"# Step 6 plan (updated {dt.datetime.now():%a %d %b %H:%M})", "",
             f"Benchmark: `{DATA}`; configuration: step 5 (see `experiments/IMPROVEMENT.md`). "
             f"Estimated finish: **{finish:%a %d %b, %H:%M}** (about {left:.0f} h left).", "",
             "| Phase | What | Estimate | State |", "|---|---|---|---|"]
    for p, what, h in PHASES:
        state = "done" if p in done else ("running" if p == current else "queued")
        lines.append(f"| {p} | {what} | {h:.1f} h | {state} |")
    lines += ["", "Detail: training `experiments/seeds_coherence/STATUS.md`; rebuild "
              "`experiments/rebuild_coherence/STATUS.md`."]
    if note:
        lines += ["", note]
    (S / "PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sh(*cmd, log=None):
    cmd = [str(c) for c in cmd]
    print(f"[{dt.datetime.now():%H:%M}] " + " ".join(cmd), flush=True)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def twice(cmd, output: Path):
    """Run a summary twice from fresh processes and require identical output files. Console
    output goes to <output>.log, not to this driver's log, since it carries test figures."""
    log = output.with_suffix(".log")
    sh(*cmd, log=log)
    first = output.with_suffix(output.suffix + ".first")
    shutil.copy(output, first)
    sh(*cmd, log=log)
    same = filecmp.cmp(output, first, shallow=False)
    first.unlink()
    if not same:
        raise SystemExit(f"{output} differs between two runs")
    return f"{output.relative_to(ROOT)}: identical on two runs"


def copy_pilots(*names):
    for name in names:
        if not (S / name).exists():
            shutil.copytree(PILOTS[name], S / name)


def score_test(run: Path, model: str):
    """Test scores for a run trained with validation scoring only (the step 5 pilots)."""
    from src.experiments.score_windows import variant_flags
    if not (run / "window_scores_test_b1.npz").exists():
        sh(PY, "-u", "-m", "src.experiments.score_windows", "--data", DATA, "--run-dir", run,
           "--model", model, "--split", "test", "--batch-size", "1", "--node-scores",
           "--cached", "--no-metrics", *variant_flags(run), log=run / "score_test.log")


def cross_score_duplicates():
    """The detector seeds on the duplicates benchmark's test split (freeze section 6's
    'trained without, tested with duplicates' row), and their summary."""
    from src.experiments.score_windows import variant_flags
    out_dir = S / "cross_duplicates"
    out_dir.mkdir(exist_ok=True)
    for s in SEEDS:
        run, out = S / f"tgn_s{s}", out_dir / f"tgn_s{s}_on_dup.npz"
        if not out.exists():
            sh(PY, "-u", "-m", "src.experiments.score_windows", "--data", DUP_DATA,
               "--run-dir", run, "--model", "tgn", "--split", "test", "--batch-size", "1",
               "--cached", "--no-metrics", "--output", out, *variant_flags(run),
               log=out.with_suffix(".log"))


def cross_duplicates_summary(output: Path):
    """Seed-averaged AUROC and AP of the detector on the duplicates benchmark's test split,
    with the freeze's bootstrap (duplicates_study.group_stats)."""
    import numpy as np
    from src.experiments.duplicates_study import boots, group_stats, seq_scores
    files = [S / "cross_duplicates" / f"tgn_s{s}_on_dup.npz" for s in SEEDS]
    order = sorted(set(np.load(files[0], allow_pickle=True)["seq"]))
    y = np.array([int(i.startswith("poisoned")) for i in order])
    out = {"data": DUP_DATA, "split": "test", "n_sequences": len(order),
           "trained_on": DATA, "inputs": [f.relative_to(ROOT).as_posix() for f in files],
           "detector": group_stats(y, [seq_scores(f, order) for f in files], boots(y))}
    json.dump(out, open(output, "w"), indent=2)


def ensemble_summary(output: Path):
    """The configuration chosen by the step 5 rule, (C) detector + static GAT rank mean, over
    the five seed pairs (detector seed s with static seed s), beside the detector (B) and the
    static GAT alone, with the freeze's bootstrap. Primary: ranks within the scored split, as
    the rule was measured. Secondary (deployable form): each score replaced by its position in
    the same model's validation scores."""
    import numpy as np
    from scipy.stats import rankdata
    from src.experiments.duplicates_study import boots, group_diff, group_stats, seq_scores

    def scores(model, s, split, order):
        return seq_scores(S / f"{model}_s{s}" / f"window_scores_{split}_b1.npz", order)

    out = {"data": DATA, "pairs": [f"tgn_s{s} + static_attn_s{s}" for s in SEEDS]}
    for split in ("val", "test"):
        z = np.load(S / "tgn_s1" / f"window_scores_{split}_b1.npz", allow_pickle=True)
        order = sorted(set(z["seq"] if "seq" in z.files else z["window_sequence_id"]))
        y = np.array([int(i.startswith("poisoned")) for i in order])
        B = boots(y)
        det = [scores("tgn", s, split, order) for s in SEEDS]
        sta = [scores("static_attn", s, split, order) for s in SEEDS]
        ens = [(rankdata(d) + rankdata(t)) / (2 * len(d)) for d, t in zip(det, sta)]
        res = {"n_sequences": len(order), "detector_plus_static": group_stats(y, ens, B),
               "detector": group_stats(y, det, B), "static": group_stats(y, sta, B),
               "ensemble_minus_detector": group_diff(y, ens, det, B),
               "ensemble_minus_static": group_diff(y, ens, sta, B),
               "detector_minus_static": group_diff(y, det, sta, B)}
        if split == "test":
            vorder = sorted(set(np.load(S / "tgn_s1" / "window_scores_val_b1.npz",
                                        allow_pickle=True)["seq"]))

            def ecdf(model, s, x):
                v = np.sort(scores(model, s, "val", vorder))
                return np.searchsorted(v, x, side="right") / len(v)
            dep = [(ecdf("tgn", s, d) + ecdf("static_attn", s, t)) / 2
                   for s, d, t in zip(SEEDS, det, sta)]
            res["detector_plus_static_validation_ecdf"] = group_stats(y, dep, B)
        out[split] = res
    json.dump(out, open(output, "w"), indent=2)


def main():
    sys.path.insert(0, str(ROOT))
    if sys.argv[1:2] == ["cross-duplicates"]:
        cross_duplicates_summary(Path(sys.argv[sys.argv.index("--output") + 1]))
        return
    if sys.argv[1:2] == ["ensemble"]:
        ensemble_summary(Path(sys.argv[sys.argv.index("--output") + 1]))
        return
    S.mkdir(parents=True, exist_ok=True)
    done, checks = [], []
    keep_awake(True)
    try:
        plan("A", done)
        copy_pilots("tgn_s1", "static_attn_s1")
        for model in ("tgn", "static_attn"):
            score_test(S / f"{model}_s1", model)
        sh(PY, "-u", "-m", "src.experiments.seed_queue", "--data", DATA, "--root", S,
           "--seeds", *SEEDS, "--models", "tgn", "static_attn", *CONFIG,
           "--coherence-features", log=S / "phase_A_driver.log")
        done.append("A")

        plan("B", done)
        R.mkdir(parents=True, exist_ok=True)
        if not (R / "probes_raw.json").exists():      # data only; identical to the freeze's
            shutil.copy(ROOT / "experiments" / "rebuild" / "probes_raw.json", R)
        rebuild = [PY, "-u", "-m", "src.experiments.rebuild"]
        common = ["--data", DATA, "--seeds-root", S.relative_to(ROOT), "--out",
                  R.relative_to(ROOT), "--online"]
        sh(*rebuild, "run", *common, "--extra-sweeps", "embedding_noise_full",
           log=S / "phase_B.log")
        sh(*rebuild, "figures", *common, "--sequence", "poisoned_placement_0463",
           "--fig-dir", "figures_missed_example", log=S / "phase_B_missed_example.log")
        done.append("B")

        plan("D1", done)
        tables = [PY, "-m", "src.experiments.seed_tables", "--seeds-root", S, "--model", "tgn",
                  "--data", DATA, "--online",
                  "--lr-scores", FROZEN / "latency_window_scores_nodup.npz",
                  "--lr-latency", FROZEN / "latency_nodup.json"]
        checks.append(twice([PY, "-m", "src.experiments.seed_summary", "--root", S,
                             "--output", S / "summary_headline.json"],
                            S / "summary_headline.json"))
        checks.append(twice(tables + ["--output", S / "seed_tables.json"], S / "seed_tables.json"))
        checks.append(twice([PY, "-m", "src.experiments.coherence_plan", "ensemble",
                             "--output", S / "ensemble_headline.json"],
                            S / "ensemble_headline.json"))
        done.append("D1")

        plan("C", done, "\n".join(f"- {c}" for c in checks))
        copy_pilots("tgn_nocoherence_s1")
        sh(PY, "-u", "-m", "src.experiments.seed_queue", "--data", DATA, "--root", S,
           "--seeds", *SEEDS, "--models", *ABLATIONS, *CONFIG, "--coherence-features",
           log=S / "phase_C.log")
        sh(PY, "-u", "-m", "src.experiments.seed_queue", "--data", DATA, "--root", S,
           "--seeds", *SEEDS, "--models", "tgn", "--suffix", "_nocoherence", *CONFIG,
           log=S / "phase_C_nocoherence.log")
        done.append("C")

        plan("D2", done, "\n".join(f"- {c}" for c in checks))
        checks.append(twice([PY, "-m", "src.experiments.seed_summary", "--root", S,
                             "--output", S / "summary.json"], S / "summary.json"))
        checks.append(twice(
            [PY, "-m", "src.experiments.oracle_comparison", "--data", DATA,
             "--detector", *[S / f"tgn_s{s}" / "window_scores_test_b1.npz" for s in SEEDS],
             "--window-lr", FROZEN / "latency_window_scores_nodup.npz", "--coherence",
             "--output", S / "oracle_comparison_5seeds.json"],
            S / "oracle_comparison_5seeds.json"))
        cross_score_duplicates()
        checks.append(twice([PY, "-m", "src.experiments.coherence_plan", "cross-duplicates",
                             "--output", S / "cross_duplicates.json"],
                            S / "cross_duplicates.json"))
        profile = R / "table_4_11_profile.json"
        if not profile.exists():
            sh(PY, "-u", "-m", "src.experiments.compute_profile", "--data", DATA,
               "--run-dir", S / "tgn_s1", "--output", profile, log=profile.with_suffix(".log"))
        done.append("D2")
        plan(None, done, "All phases finished.\n\n" + "\n".join(f"- {c}" for c in checks))
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
