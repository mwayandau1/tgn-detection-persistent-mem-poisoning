"""
Attack size, duplication and burst length, varied one at a time (E4).
=====================================================================

On ap_burst2k three properties of an attack move together:

  size          k is drawn from 5..20
  duplication   the attacker draws from a pool of 8 neighbours, with replacement once k > 8,
                so every cluster of 9 or more contains exact copies and none of 8 or fewer does
  burst length  the attack occupies exactly k consecutive interactions

so the size curve of Section 4.5 cannot say which of them the detector responds to. Measured
on the test partition, sequence-level AUROC steps from 0.56 at k=8 to 0.78 at k=9, exactly
where duplication begins.

This builds evaluation-only sets in which the three are crossed -- duplication allowed or not,
k in {5, 8, 9, 12, 16, 20}, burst length k (as trained), 40 or 80 interactions -- and scores a
FIXED trained detector on each. Every poisoned cell is compared with matched clean controls of
the same k and the same duplication rule, generated exactly as ap_burst2k's controls are
(benign cluster spread over up to 120 interactions). Nothing is retrained, so the grid measures
what the reported detector responds to, not what a detector could learn.

Usage:
    python -m src.experiments.attack_size_grid --run-dir experiments/run_burst2k_base \
        --root data/raw/e4_grid --output experiments/audit/attack_size_grid.json
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SIZES = [5, 8, 9, 12, 16, 20]
SPREADS = [None, 40, 80]          # None = k consecutive interactions, as in ap_burst2k


def generate(out: Path, k: int, poisoned: bool, n: int, seed: int, no_dup: bool,
             spread):
    if (out / "splits.json").exists():
        return
    cmd = [sys.executable, "-m", "src.data_generation.agentpoison_sequences",
           "--output", str(out), "--num-sequences", str(n),
           "--attack-ratio", "1.0" if poisoned else "0.0",
           "--matched-control", "1.0", "--benign-spread", "120",
           "--attack-spread-mult", "1.0", "--injection-range", str(k), str(k),
           "--train-ratio", "0", "--val-ratio", "0", "--seed", str(seed)]
    if spread:
        cmd += ["--attack-spread", str(spread)]
    if no_dup:
        cmd += ["--no-duplicates"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def score(run_dir: str, data: Path, out_npz: Path, model: str):
    if out_npz.exists():
        return
    subprocess.run([sys.executable, "-m", "src.experiments.score_windows", "--data", str(data),
                    "--run-dir", run_dir, "--model", model, "--split", "test",
                    "--batch-size", "4", "--memory-slots", "per_window",
                    "--output", str(out_npz)], check=True, stdout=subprocess.DEVNULL)


def seq_max(npz: Path):
    z = np.load(npz, allow_pickle=True)
    ids = sorted(set(z["seq"]))
    return np.array([z["score"][z["seq"] == i].max() for i in ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", default="tgn")
    ap.add_argument("--root", default="data/raw/e4_grid")
    ap.add_argument("--n-poisoned", type=int, default=50)
    ap.add_argument("--n-clean", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--output", default="experiments/audit/attack_size_grid.json")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    root = Path(args.root)
    tag = Path(args.run_dir).name
    results = {"run_dir": args.run_dir, "n_poisoned": args.n_poisoned,
               "n_clean": args.n_clean, "cells": []}
    seed = 1000
    for no_dup, k in itertools.product((False, True), SIZES):
        rule = "nodup" if no_dup else "dup"
        clean_dir = root / f"{rule}_k{k}_clean"
        seed += 1
        generate(clean_dir, k, False, args.n_clean, seed, no_dup, None)
        clean_npz = clean_dir / f"scores_{tag}.npz"
        score(args.run_dir, clean_dir, clean_npz, args.model)
        sc = seq_max(clean_npz)
        for spread in SPREADS:
            seed += 1
            pdir = root / f"{rule}_k{k}_spread{spread or 'k'}"
            generate(pdir, k, True, args.n_poisoned, seed, no_dup, spread)
            pnpz = pdir / f"scores_{tag}.npz"
            score(args.run_dir, pdir, pnpz, args.model)
            sp = seq_max(pnpz)
            y = np.r_[np.zeros(len(sc)), np.ones(len(sp))]
            s = np.r_[sc, sp]
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(args.n_boot):
                i = np.r_[rng.choice(len(sc), len(sc)), len(sc) + rng.choice(len(sp), len(sp))]
                boots.append(roc_auc_score(y[i], s[i]))
            cell = {"duplicates": not no_dup, "k": k, "burst_interactions": spread or k,
                    "auroc": float(roc_auc_score(y, s)),
                    "ci95": [float(np.percentile(boots, 2.5)),
                             float(np.percentile(boots, 97.5))],
                    "mean_score_poisoned": float(sp.mean()),
                    "mean_score_clean": float(sc.mean())}
            results["cells"].append(cell)
            print(f"{rule:5s} k={k:2d} burst={str(spread or k):>3s}  AUROC {cell['auroc']:.3f} "
                  f"[{cell['ci95'][0]:.3f}, {cell['ci95'][1]:.3f}]", flush=True)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            json.dump(results, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
