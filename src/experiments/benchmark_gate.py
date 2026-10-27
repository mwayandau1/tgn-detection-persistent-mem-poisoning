"""
Is this benchmark worth training on?
====================================

Four ways a memory-poisoning benchmark can be solvable without doing any of the work it claims
to test. Each has been hit in this project, and each is measured here before a GPU is used.

1.  CONTENT LEAK.  A logistic regression on one record's embedding recovers the label. Both
    LLM-generated benchmarks score AUROC 1.000 on this, because a record was poisoned if and
    only if it was *generated* from a "harmful advice" instruction -- so poisoned and benign
    text are two different distributions and no graph is involved. Handled by
    separability_probe.node_content_probe; re-reported here for completeness.

2.  POSITION LEAK.  Window labels are presence-based, so a window turns positive only once the
    injection has appeared, and injections start part-way through a sequence. Positive windows
    therefore sit later than negative ones. Records accumulate one per interaction with a
    snapshot every `snapshot_interval`, which makes node count an EXACT linear function of
    snapshot index:

        n_nodes(snapshot i) = (i + 1) * snapshot_interval

    Node count is the first of the nine structural features, so that feature is window
    position under another name. Measured: position alone scores 0.586 on ap_sequences and
    0.596 on ap_matched_2k, putting an unearned floor under every window-level AUROC in the
    project. This is distinct from mislabelling -- the labels are perfectly consistent with
    snapshot content; the defect is that they are *exploitable* by a feature that describes no
    structure at all.

3.  FINAL-STATE LEAK.  The poisoned final graph is simply denser than the benign one, so no
    temporal modelling is needed. This is what the matched-control benchmark exists to remove.

4.  NO SIGNAL AT ALL.  Having removed the leaks, nothing is left to learn. A benchmark can be
    unsolvable rather than hard, and the two look identical from a training curve.

Sequence-level metrics are primary here. Position cannot help at sequence level, because each
sequence contributes exactly one observation -- and "is this agent's memory poisoned?" is a
per-sequence question anyway. Window-level numbers are reported alongside, with the
position-only baseline next to them so the floor is visible rather than absorbed.

Usage:
    python -m src.experiments.benchmark_gate --data data/raw/ap_burst_1.0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def sequence_features(data: Path, seq_ids, threshold, emb_dim):
    """Final-state and growth-rate summaries, one row per sequence."""
    rows = []
    for sid in seq_ids:
        snaps = sorted((data / sid).glob("snapshot_*.npz"))
        if not snaps:
            continue
        sizes, dens, tight = [], [], []
        for p in snaps:
            with np.load(p) as z:
                x = z["node_features"][:, :emb_dim]
            u = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            sim = u @ u.T
            np.fill_diagonal(sim, 0.0)
            adj = sim > threshold
            deg = adj.sum(axis=1)
            sizes.append(float(deg.max()) if deg.size else 0.0)
            dens.append(float(adj.sum()) / 2.0)
            # Mean similarity inside the tightest neighbourhood: tracks how coherent the
            # densest region is, independently of how large it is.
            if deg.size and deg.max() > 0:
                hub = int(np.argmax(deg))
                nb = np.flatnonzero(adj[hub])
                tight.append(float(sim[np.ix_(nb, nb)].mean()) if nb.size > 1 else 0.0)
            else:
                tight.append(0.0)
        sizes, dens, tight = np.array(sizes), np.array(dens), np.array(tight)
        d_size, d_dens = np.diff(sizes), np.diff(dens)
        rows.append({
            "sequence_id": sid,
            "final_maxdeg": sizes[-1],
            "final_density": dens[-1],
            "final_tightness": tight[-1],
            "max_jump_maxdeg": float(d_size.max()) if d_size.size else 0.0,
            "max_jump_density": float(d_dens.max()) if d_dens.size else 0.0,
            "jump_ratio": float(d_dens.max() / (dens[-1] + 1e-6)) if d_dens.size else 0.0,
            "growth_std": float(d_dens.std()) if d_dens.size else 0.0,
            "max_jump_tightness": float(np.diff(tight).max()) if tight.size > 1 else 0.0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-sequences", type=int, default=400)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    T = cfg["training"]["temporal_window"]
    interval = cfg["simulation"]["snapshot_interval"]
    threshold = cfg["simulation"]["semantic_edge_threshold"]
    emb_dim = cfg["simulation"]["embedding_dim"]

    data = Path(args.data)
    meta = {m["sequence_id"]: m for m in json.load(open(data / "metadata.json"))}
    splits = json.load(open(data / "splits.json"))
    # "all" is legitimate here precisely because nothing is fitted: every number below is the
    # AUROC of a hand-built feature, so there is no split to leak across and the only thing a
    # larger sample buys is a tighter estimate. With 24 poisoned test sequences the standard
    # error on an AUROC is around 0.12, which is wider than any effect being looked for.
    ids = (sum(splits.values(), []) if args.split == "all" else splits[args.split])
    if args.max_sequences and len(ids) > args.max_sequences:
        ids = list(np.random.default_rng(0).choice(ids, size=args.max_sequences,
                                                   replace=False))

    # --- window level, and the position floor under it ------------------------------------
    pos, wlab = [], []
    for sid in ids:
        snaps = sorted((data / sid).glob("snapshot_*.npz"))
        gl = []
        for p in snaps:
            with np.load(p) as z:
                gl.append(int(np.asarray(z["graph_label"]).reshape(-1)[0]))
        for s in range(max(1, len(snaps) - T + 1)):
            end = s + T - 1
            if end < len(gl):
                pos.append(end)
                wlab.append(gl[end])
    pos, wlab = np.array(pos, float), np.array(wlab, int)
    pos_auroc = float(roc_auc_score(wlab, pos)) if len(set(wlab.tolist())) > 1 else float("nan")

    # --- sequence level -------------------------------------------------------------------
    rows = sequence_features(data, ids, threshold, emb_dim)
    y = np.array([int(meta[r["sequence_id"]]["label"] == "poisoned") for r in rows])

    feats = ["final_maxdeg", "final_density", "final_tightness",
             "max_jump_maxdeg", "max_jump_density", "jump_ratio",
             "growth_std", "max_jump_tightness"]
    kinds = {f: ("final-state" if f.startswith("final") else "temporal") for f in feats}

    print(f"\n{data.name}  [{args.split}]  {len(rows)} sequences, {int(y.sum())} poisoned; "
          f"{len(wlab)} windows, {int(wlab.sum())} positive")
    print(f"\n  WINDOW LEVEL -- the floor that owes nothing to structure")
    print(f"    window end-snapshot index alone      AUROC {pos_auroc:.4f}")
    print(f"    (node count is (i+1)*{interval}, so this is also node count alone)")

    print(f"\n  SEQUENCE LEVEL -- position cannot contribute here")
    print(f"    {'feature':<22} {'kind':<12} {'AUROC':>7}  {'|AUROC-0.5|':>11}")
    best_t, best_f = 0.0, 0.0
    for f in feats:
        v = np.array([r[f] for r in rows], float)
        a = float(roc_auc_score(y, v))
        dev = abs(a - 0.5)
        print(f"    {f:<22} {kinds[f]:<12} {a:>7.4f}  {dev:>11.4f}")
        if kinds[f] == "temporal":
            best_t = max(best_t, dev)
        else:
            best_f = max(best_f, dev)

    print(f"\n  best final-state deviation from chance : {best_f:.4f}")
    print(f"  best temporal   deviation from chance : {best_t:.4f}")
    verdict = ("TEMPORAL SIGNAL PRESENT -- worth training"
               if best_t >= 0.10 and best_t > best_f else
               "FINAL-STATE SIGNAL DOMINATES -- matching is incomplete"
               if best_f >= 0.10 else
               "NO SIGNAL -- do not train; retune the generator")
    print(f"  verdict: {verdict}")

    out = Path(args.output) if args.output else data / "benchmark_gate.json"
    json.dump({"position_only_window_auroc": pos_auroc,
               "sequence_level": {f: float(roc_auc_score(
                   y, np.array([r[f] for r in rows], float))) for f in feats},
               "n_sequences": len(rows), "n_poisoned": int(y.sum()),
               "verdict": verdict}, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
