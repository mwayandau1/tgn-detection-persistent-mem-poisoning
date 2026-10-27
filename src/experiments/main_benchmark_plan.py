"""
Brief v3, after R1: everything on the main benchmark (ap_burst2k_nodup), in order.
==================================================================================

  A  detector and static GAT to five seeds (seeds 4 and 5; 1 to 3 exist)
  B  R2 rebuild: static-GAT record metrics, attribution probes, Figures 4.1 to 4.3,
     robustness sweeps over five seeds (src.experiments.rebuild)
  C  R3 and the E3 rerun, five seeds each: memory only, aggregator only, size-matched static,
     mean pooling, no structural features
  D  nine-feature latency baseline on this benchmark, then every summary twice from fresh
     processes, compared byte for byte

Resumable: each phase skips what is already done. Progress: experiments/seeds_nodup/PLAN.md,
plus each phase's own STATUS.md. The laptop is kept awake while this runs.

Usage:
    python -m src.experiments.main_benchmark_plan
"""

import datetime as dt
import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = "data/raw/ap_burst2k_nodup"
SEEDS = ROOT / "experiments" / "seeds_nodup"
PY = sys.executable
PHASES = [("A", "Detector and static GAT to five seeds", 7.5),
          ("B", "R2 rebuild (records, probes, figures, sweeps)", 16.0),
          ("C", "Ablations x 5 seeds (memory only, aggregator only, size-matched static, "
                "mean pool, no structural features)", 55.5),
          ("D", "Latency baseline and summaries, each run twice", 1.0)]


def keep_awake(on):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x1 if on else 0))


def plan(current, done, note=""):
    left = sum(h for p, _, h in PHASES if p not in done)
    finish = dt.datetime.now() + dt.timedelta(hours=left)
    lines = [f"# Main-benchmark plan (updated {dt.datetime.now():%a %d %b %H:%M})", "",
             f"Benchmark: `{DATA}`. Estimated finish: **{finish:%a %d %b, %H:%M}** "
             f"(about {left:.0f} h left).", "",
             "| Phase | What | Estimate | State |", "|---|---|---|---|"]
    for p, what, h in PHASES:
        state = "done" if p in done else ("running" if p == current else "queued")
        lines.append(f"| {p} | {what} | {h:.1f} h | {state} |")
    lines += ["", "Detail: training `experiments/seeds_nodup/STATUS.md`; rebuild "
              "`experiments/rebuild/STATUS.md`."]
    if note:
        lines += ["", note]
    (SEEDS / "PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sh(*cmd, log=None):
    cmd = [str(c) for c in cmd]
    print(f"[{dt.datetime.now():%H:%M}] " + " ".join(cmd), flush=True)
    if log:
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, cwd=ROOT)


def twice(cmd, output: Path):
    """Run a summary twice from fresh processes and require identical output files."""
    sh(*cmd)
    first = output.with_suffix(output.suffix + ".first")
    shutil.copy(output, first)
    sh(*cmd)
    same = filecmp.cmp(output, first, shallow=False)
    first.unlink()
    if not same:
        raise SystemExit(f"{output} differs between two runs")
    return f"{output.relative_to(ROOT)}: identical on two runs"


def main():
    SEEDS.mkdir(parents=True, exist_ok=True)
    done = []
    keep_awake(True)
    try:
        plan("A", done)
        sh(PY, "-u", "-m", "src.experiments.seed_queue", "--data", DATA, "--root", SEEDS,
           "--seeds", "1", "2", "3", "4", "5", "--models", "tgn", "static_attn",
           log=SEEDS / "phase_A.log")
        done.append("A")

        plan("B", done)
        sh(PY, "-u", "-m", "src.experiments.rebuild", "run", "--data", DATA,
           "--seeds-root", "experiments/seeds_nodup", log=SEEDS / "phase_B.log")
        done.append("B")

        plan("C", done)
        sh(PY, "-u", "-m", "src.experiments.seed_queue", "--data", DATA, "--root", SEEDS,
           "--seeds", "1", "2", "3", "4", "5", "--models", "no_memory", "no_aggregator",
           "static_wide", "meanpool", "nostruct", log=SEEDS / "phase_C.log")
        done.append("C")

        plan("D", done)
        lat = SEEDS / "latency_nodup.json"
        if not lat.exists():
            sh(PY, "-u", "-m", "src.experiments.detection_latency", "--data", DATA,
               "--run-dir", SEEDS / "tgn_s1", "--target-fpr", "0.05", "0.10",
               "--memory-slots", "per_window", "--cached", "--output", lat,
               "--window-scores-name", "../latency_window_scores_nodup.npz",
               log=SEEDS / "phase_D_latency.log")
        checks = [
            twice([PY, "-m", "src.experiments.seed_summary", "--root", SEEDS,
                   "--output", SEEDS / "summary.json"], SEEDS / "summary.json"),
            twice([PY, "-m", "src.experiments.seed_tables", "--seeds-root", SEEDS,
                   "--model", "tgn", "--data", DATA,
                   "--lr-scores", SEEDS / "latency_window_scores_nodup.npz",
                   "--lr-latency", lat, "--output", SEEDS / "seed_tables.json"],
                  SEEDS / "seed_tables.json"),
            twice([PY, "-m", "src.experiments.oracle_comparison", "--data", DATA,
                   "--detector", *[SEEDS / f"tgn_s{s}" / "window_scores_test_b1.npz"
                                   for s in range(1, 6)],
                   "--window-lr", SEEDS / "latency_window_scores_nodup.npz",
                   "--output", SEEDS / "oracle_comparison_5seeds.json"],
                  SEEDS / "oracle_comparison_5seeds.json"),
        ]
        done.append("D")
        plan(None, done, "All phases finished.\n\n" + "\n".join(f"- {c}" for c in checks))
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
