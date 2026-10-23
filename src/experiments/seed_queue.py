"""
The multi-seed study (E2) and component ablations (E3) as one resumable queue.
==============================================================================

Seed-major order: every configuration at seed 1, then every configuration at seed 2, and so
on. Stopping the queue at any point therefore leaves all configurations with the same number
of seeds, which is what a paired comparison across configurations needs. A run whose
window_scores_test_b1.npz exists is skipped, so the queue can be killed and restarted freely.

Progress is written where it can be read without the terminal:
  <root>/STATUS.md            what is running, what has finished (with test AUROC), an ETA
  <root>/<model>_s<seed>/train.log   the full output of each run, epoch by epoch

--keep-awake asks Windows not to sleep while the queue runs (SetThreadExecutionState). The
request belongs to this process and lapses when it exits; no power setting is changed.

Usage:
    python -m src.experiments.seed_queue --data data/raw/ap_burst2k --seeds 1 2 3 4 5 \
        --models core --keep-awake
    python -m src.experiments.seed_queue ... --dry-run
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = ["tgn", "static_attn", "no_memory", "no_aggregator", "static_wide",
           "meanpool", "nostruct", "static_mean"]
CORE = CONFIGS[:5]
# Hours per run on the Quadro P2000, used only for the ETA until real durations exist.
GUESS_HOURS = {"tgn": 2.8, "meanpool": 2.8, "nostruct": 2.8, "no_memory": 2.5,
               "no_aggregator": 1.5, "static_attn": 1.0, "static_mean": 1.0,
               "static_wide": 1.3}


def keep_awake(on: bool):
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def write_status(root: Path, plan, done, current, durations):
    hours = {m: (sum(d) / len(d) / 3600 if d else GUESS_HOURS.get(m, 2.0))
             for m, d in durations.items()}
    remaining = [(m, s) for m, s in plan if (m, s) not in done and (m, s) != current]
    eta_h = sum(hours[m] for m, _ in remaining) + (hours[current[0]] if current else 0)
    finish = dt.datetime.now() + dt.timedelta(hours=eta_h)
    lines = [f"# Seed queue status (updated {dt.datetime.now():%Y-%m-%d %H:%M})", "",
             f"**{len(done)} of {len(plan)} runs finished.** "
             f"Estimated finish: {finish:%a %d %b, %H:%M} (about {eta_h:.1f} h of GPU time left).",
             ""]
    if current:
        lines += [f"Running now: `{current[0]}` seed {current[1]} "
                  f"(log: `{current[0]}_s{current[1]}/train.log`)", ""]
    lines += ["| Run | Status | Test window AUROC (batch 1) | Hours |", "|---|---|---|---|"]
    for m, s in plan:
        run = root / f"{m}_s{s}"
        rec = run / "variant.json"
        if rec.exists():
            r = json.load(open(rec))
            lines.append(f"| {m} s{s} | done | {r.get('test_window_auroc_b1', float('nan')):.4f} "
                         f"| {r.get('train_seconds', 0) / 3600:.1f} |")
        elif (m, s) == current:
            lines.append(f"| {m} s{s} | running | | |")
        else:
            lines.append(f"| {m} s{s} | queued | | |")
    (root / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", default="experiments/seeds")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--models", nargs="+", default=CORE,
                    help="Model kinds, or 'core' (the first five) or 'all'.")
    ap.add_argument("--memory-slots", choices=["shared", "per_window"], default="per_window")
    ap.add_argument("--keep-awake", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = CORE if args.models == ["core"] else CONFIGS if args.models == ["all"] \
        else args.models
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    plan = [(m, s) for s in args.seeds for m in models]
    done, durations = set(), {m: [] for m in models}
    for m, s in plan:
        if (root / f"{m}_s{s}" / "window_scores_test_b1.npz").exists():
            done.add((m, s))
            rec = root / f"{m}_s{s}" / "variant.json"
            if rec.exists():
                durations[m].append(json.load(open(rec)).get("train_seconds", 0))

    if args.keep_awake and not args.dry_run:
        keep_awake(True)
    try:
        for m, s in plan:
            if (m, s) in done:
                continue
            out = root / f"{m}_s{s}"
            cmd = [sys.executable, "-u", "-m", "src.experiments.train_variant",
                   "--data", args.data, "--model", m, "--seed", str(s),
                   "--output", str(out), "--memory-slots", args.memory_slots, "--cached"]
            print(f"[{dt.datetime.now():%H:%M}] {'would run' if args.dry_run else 'running'} "
                  f"{m} seed {s}", flush=True)
            if args.dry_run:
                continue
            write_status(root, plan, done, (m, s), durations)
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(out / "train.log", "w", encoding="utf-8") as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            durations[m].append(time.time() - t0)
            done.add((m, s))
            write_status(root, plan, done, None, durations)
            rec = json.load(open(out / "variant.json"))
            print(f"[{dt.datetime.now():%H:%M}] done {m} seed {s}: test window AUROC "
                  f"{rec.get('test_window_auroc_b1', float('nan')):.4f} "
                  f"({(time.time() - t0) / 3600:.1f} h)", flush=True)
    finally:
        if args.keep_awake and not args.dry_run:
            keep_awake(False)
        if not args.dry_run:
            write_status(root, plan, done, None, durations)


if __name__ == "__main__":
    main()
