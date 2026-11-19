"""
How much of the attack is detectable at all on a benchmark? A ceiling check.
==========================================================================

The detector scores 0.66 sequence AUROC on ap_burst2k_nodup and the thirteen engineered
features 0.69. Neither says whether 0.75 is reachable: both could be far below what the data
allows, or near it. This scores two oracles that know more than any learned model is given,
on the same 500 test sequences, so the gap between the detector and the information actually
present can be read off.

  A. Arrival-run scan, embeddings only. Every record's position in a snapshot is its arrival
     order (the recency column encodes it exactly) and the attack lands as k consecutive
     records. For each run of L consecutive records (L = 5..20) the scan measures how tight
     the run is: mean pairwise cosine and the fraction of pairs above the edge threshold,
     raw and relative to the same sequence's typical run. Max over runs, per L. This uses
     nothing the detector does not receive, so it estimates what is visible in its input.

  B. Generator oracle. Also knows how the generator draws an attack: k entries from the
     max(8, k) nearest neighbours of one knowledge-base entry. For each run it finds the
     entry whose neighbour list covers the most records of the run. The attack covers its
     own run completely; ordinary sessions draw from 60 neighbours and rarely do. This is
     close to the likelihood-ratio test for this attack model, so it approximates the ceiling.

Each oracle is fitted on TRAIN, its hyperparameters chosen on VALIDATION, and scored once on
TEST, the protocol the detector had. Paired differences from the detector are averaged over
its five seeds, as in oracle_comparison.py. Results are also split by attack size, against
clean controls with a cluster of the same size.

Usage:
    python -m src.experiments.ceiling_check --data data/raw/ap_burst2k_nodup \
        --detector experiments/seeds_nodup/tgn_s{1,2,3,4,5}/window_scores_test_b1.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.oracle_comparison import (fit_gb, fit_lr, multi_seed_comparison,  # noqa: E402
                                               seq_max)
from src.experiments.sequence_eval import boot_ci  # noqa: E402

RUN_LENGTHS = list(range(5, 21))
SIZE_BANDS = {"5-8": (5, 8), "9-12": (9, 12), "13-16": (13, 16), "17-20": (17, 20)}


def final_snapshot(data: Path, sid: str, emb_dim: int):
    last = sorted((data / sid).glob("snapshot_*.npz"))[-1]
    with np.load(last) as z:
        return np.asarray(z["node_features"][:, :emb_dim], np.float64)


def block_means(M: np.ndarray, L: int) -> np.ndarray:
    """Mean of M over the off-diagonal entries of every L x L diagonal block."""
    P = np.zeros((M.shape[0] + 1, M.shape[1] + 1))
    P[1:, 1:] = M.cumsum(0).cumsum(1)
    s = np.arange(M.shape[0] - L + 1)
    total = P[s + L, s + L] - P[s, s + L] - P[s + L, s] + P[s, s]
    return total / (L * (L - 1))


def scan_features(emb: np.ndarray, threshold: float) -> np.ndarray:
    """Oracle A: per run length, the tightest run of consecutive arrivals."""
    u = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    S = u @ u.T
    np.fill_diagonal(S, 0.0)
    E = (S > threshold).astype(np.float64)
    np.fill_diagonal(E, 0.0)
    row = []
    for L in RUN_LENGTHS:
        for stat in (block_means(S, L), block_means(E, L)):
            row += [stat.max(), stat.max() - np.median(stat)]
    return np.array(row)


class GeneratorOracle:
    """Oracle B: the knowledge-base entry whose neighbour list best covers a run."""

    def __init__(self, kb_emb: np.ndarray, benign_pool: int, poison_pool: int):
        self.unit = kb_emb / (np.linalg.norm(kb_emb, axis=1, keepdims=True) + 1e-8)
        sim = self.unit @ self.unit.T
        np.fill_diagonal(sim, -np.inf)          # as the generator ranks neighbours
        nb = np.argsort(-sim, axis=1)
        n = len(kb_emb)
        self.widths = sorted({max(poison_pool, L) for L in RUN_LENGTHS})
        # cover[w][r, c] is 1 when entry r is among the w nearest neighbours of entry c
        self.cover = {}
        for w in self.widths:
            m = np.zeros((n, n), np.int16)
            m[nb[:, :w].ravel(), np.repeat(np.arange(n), w)] = 1
            self.cover[w] = m
        self.poison_pool = poison_pool

    def kb_index(self, emb: np.ndarray) -> np.ndarray:
        u = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return (u @ self.unit.T).argmax(1)

    def features(self, emb: np.ndarray) -> np.ndarray:
        idx = self.kb_index(emb)
        cover_frac, longest = [], 0
        for w in self.widths:
            C = np.zeros((len(idx) + 1, self.cover[w].shape[1]), np.int32)
            C[1:] = self.cover[w][idx].cumsum(0)
            for L in [L for L in RUN_LENGTHS if max(self.poison_pool, L) == w]:
                best = (C[L:] - C[:-L]).max() / L
                cover_frac.append(best)
                if best >= 1.0:
                    longest = max(longest, L)
        return np.array(cover_frac + [longest])


def size_band_auroc(y, score, size, bands):
    out = {}
    for name, (lo, hi) in bands.items():
        keep = (size >= lo) & (size <= hi)
        yy = y[keep]
        if 0 < yy.sum() < len(yy):
            out[name] = {"n_clean": int((yy == 0).sum()), "n_poisoned": int(yy.sum()),
                         "auroc": float(roc_auc_score(yy, score[keep]))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--detector", nargs="+", required=True,
                    help="Per-window test scores of each detector seed (score_windows output).")
    ap.add_argument("--kb-embeddings", default="data/raw/agentpoison/kb_embeddings.npy")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--emb-dim", type=int, default=384)
    ap.add_argument("--benign-pool", type=int, default=60)
    ap.add_argument("--poison-pool", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default="experiments/ceiling/ceiling_check.json")
    args = ap.parse_args()

    data = Path(args.data)
    meta = {m["sequence_id"]: m for m in json.load(open(data / "metadata.json"))}
    splits = json.load(open(data / "splits.json"))
    oracle_b = GeneratorOracle(np.load(args.kb_embeddings), args.benign_pool, args.poison_pool)

    X, y, size = {}, {}, {}
    t0 = time.time()
    for split in ("train", "val", "test"):
        rows_a, rows_b = [], []
        for i, sid in enumerate(splits[split]):
            emb = final_snapshot(data, sid, args.emb_dim)
            rows_a.append(scan_features(emb, args.threshold))
            rows_b.append(oracle_b.features(emb))
            if (i + 1) % 200 == 0:
                print(f"  {split}: {i + 1}/{len(splits[split])} sequences "
                      f"({time.time() - t0:.0f} s)", flush=True)
        X[split] = {"A": np.array(rows_a), "B": np.array(rows_b)}
        y[split] = np.array([int(meta[s]["label"] == "poisoned") for s in splits[split]])
        size[split] = np.array([meta[s]["num_injected"] or meta[s]["benign_burst"]
                                for s in splits[split]])

    scores, notes = {}, {}
    for key, name in (("A", "A. arrival-run scan, embeddings only"),
                      ("B", "B. generator oracle")):
        Xtr, Xva, Xte = X["train"][key], X["val"][key], X["test"][key]
        scores[f"{name}, LR"] = fit_lr(Xtr, y["train"], Xte)
        s, hp, va = fit_gb(Xtr, y["train"], Xva, y["val"], Xte)
        scores[f"{name}, GB"] = s
        notes[f"{name}, GB"] = {"selected": hp, "val_auroc": va}
    # The generator oracle's single unfitted statistic: the longest run of consecutive
    # arrivals that one neighbour list covers completely.
    scores["B. generator oracle, longest covered run (no fitting)"] = X["test"]["B"][:, -1]

    yte = y["test"]
    det_runs = []
    for p in args.detector:
        z = np.load(p, allow_pickle=True)
        det_runs.append(seq_max(Path(p), "score" if "score" in z.files else "tgn",
                                splits["test"]))
    out = multi_seed_comparison(yte, det_runs, scores, args.n_boot)
    out["detector_inputs"] = [Path(p).as_posix() for p in args.detector]
    out["gb_selection"] = notes
    for name, s in scores.items():
        out["oracles"][name]["ci95"] = list(boot_ci(yte, s, args.n_boot))
        out["oracles"][name]["by_size"] = size_band_auroc(yte, s, size["test"], SIZE_BANDS)
    out["detector"]["by_size_mean_over_seeds"] = {
        b: float(np.mean([size_band_auroc(yte, d, size["test"], {b: r})[b]["auroc"]
                          for d in det_runs]))
        for b, r in SIZE_BANDS.items()}

    print("\nby attack size (AUROC, poisoned against clean controls of the same size)")
    print(f"{'':<58}" + "".join(f"{b:>8}" for b in SIZE_BANDS))
    print(f"{'detector (mean over seeds)':<58}" + "".join(
        f"{out['detector']['by_size_mean_over_seeds'][b]:>8.3f}" for b in SIZE_BANDS))
    for name in scores:
        bs = out["oracles"][name]["by_size"]
        print(f"{name:<58}" + "".join(f"{bs[b]['auroc']:>8.3f}" for b in SIZE_BANDS))

    out["settings"] = vars(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    np.savez_compressed(Path(args.output).with_suffix(".features.npz"),
                        **{f"{s}_{k}": X[s][k] for s in X for k in X[s]},
                        **{f"{s}_y": y[s] for s in y}, **{f"{s}_size": size[s] for s in size})
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
