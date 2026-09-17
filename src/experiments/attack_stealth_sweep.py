"""
Attack Stealth Sweep (Benchmark Hardening, Fixes #2/#3 - cheap variant)
=======================================================================
Implements the `semantic_coherence` knob that `configs/default.yaml` declares but
that NO code in `src/` has ever read, and sweeps it to produce a
coherence-vs-detectability curve.

Why this exists
---------------
Measured on the shipped dataset, poisoned records sit at mean pairwise cosine
0.885 with an internal edge density of 0.978 - a near-perfect clique - because
every injection in a sequence is generated from ONE `target_topic` and one of
only THREE hardcoded `POISON_TEMPLATES`. They are near-paraphrases of each other.
Benign records, even after being re-grouped into topic sessions by
`rebuild_session_structure.py`, only reach ~0.54. That residual gap, not topic
concentration, is what makes the benchmark near-trivial.

What this does
--------------
Simulates an attacker who varies their phrasing (and mimics the host memory's
distribution) by blending each poisoned embedding toward a randomly chosen
benign record from the same sequence:

    p_i' = normalize( alpha * p_i + (1 - alpha) * b_j )     b_j ~ benign records

Because each poisoned record blends toward a DIFFERENT benign record, lowering
alpha both (a) decorrelates the poisoned records from each other, collapsing the
clique, and (b) pulls them toward the benign manifold. alpha = 1.0 reproduces the
original attack exactly.

IMPORTANT - scope and honesty
-----------------------------
This is an embedding-space *simulation* of a stealthier attacker, run as a
controlled sensitivity analysis. It is NOT equivalent to regenerating the dataset
with genuinely more diverse poison text, because it never produces new sentences -
it interpolates existing ones. Report it as a sensitivity curve that motivates
regeneration, not as a substitute for it. The corresponding real fixes are:
expanding POISON_TEMPLATES (3 -> 30+), paraphrasing each injection, and making
the attacker choose a target topic already frequent in the victim's memory.

Only the TEST-split sequences are rebuilt (evaluation never touches train/val),
which keeps each sweep point to a few seconds instead of minutes.

Usage
-----
    python -m src.experiments.attack_stealth_sweep \
        --data data/raw/llm_generated_sessions \
        --run-dir experiments/run_current_arch
"""

import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.experiments.rebuild_session_structure import recover_records, build_snapshot
from src.experiments.generate_eval_results import Evaluator
from src.models.tgn_detector import TGNDetector


def blend_poison(embeddings: np.ndarray, labels: np.ndarray, alpha: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Blend each poisoned embedding toward a random benign one by `alpha`."""
    if alpha >= 1.0:
        return embeddings

    out = embeddings.copy()
    benign_idx = np.where(labels == 0)[0]
    poison_idx = np.where(labels == 1)[0]
    if len(benign_idx) == 0 or len(poison_idx) == 0:
        return out

    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    for p in poison_idx:
        b = benign_idx[rng.integers(0, len(benign_idx))]
        mixed = alpha * unit(embeddings[p]) + (1.0 - alpha) * unit(embeddings[b])
        # Preserve the original vector norm so the metadata/feature scale is unchanged.
        out[p] = unit(mixed) * np.linalg.norm(embeddings[p])
    return out


def poison_coherence(embeddings: np.ndarray, labels: np.ndarray, threshold: float):
    """Mean pairwise cosine and internal edge density among poisoned records."""
    P = embeddings[labels == 1]
    if len(P) < 2:
        return None, None
    N = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    s = N @ N.T
    iu = np.triu_indices(len(P), k=1)
    vals = s[iu]
    return float(vals.mean()), float((vals > threshold).mean())


def rebuild_test_split(src_dir: Path, dst_dir: Path, alpha: float, seed: int,
                       embedding_dim: int, threshold: float, snapshot_interval: int):
    """Re-emit only the test-split sequences with poison blended by `alpha`."""
    metadata = json.load(open(src_dir / "metadata.json"))
    splits = json.load(open(src_dir / "splits.json"))
    test_ids = set(splits["test"])

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_dir / "metadata.json", dst_dir / "metadata.json")
    shutil.copy(src_dir / "splits.json", dst_dir / "splits.json")

    coh_vals, dens_vals = [], []

    for m in metadata:
        seq_id = m["sequence_id"]
        if seq_id not in test_ids:
            continue
        rec = recover_records(src_dir / seq_id, embedding_dim)
        if rec is None:
            continue

        rng = np.random.default_rng(seed + abs(hash(seq_id)) % (2 ** 31))
        emb = blend_poison(rec["embeddings"], rec["labels"], alpha, rng)
        labels, ts = rec["labels"], rec["timestamps"]

        if (labels == 1).any():
            c, d = poison_coherence(emb, labels, threshold)
            if c is not None:
                coh_vals.append(c)
                dens_vals.append(d)

        # Re-emit the cumulative snapshot series in the original record order.
        out_seq = dst_dir / seq_id
        out_seq.mkdir(parents=True, exist_ok=True)
        n_total = len(labels)
        retrieval_count = np.zeros(n_total)
        last_retrieved = np.zeros(n_total)
        written = 0
        final_ts = rec["final_timestamp"]

        for t in range(final_ts + 1):
            upto = int(np.searchsorted(ts, t, side="right"))
            if upto > 5 and rng.random() < 0.3:
                k = int(rng.integers(0, upto))
                retrieval_count[k] += 1
                last_retrieved[k] = t
            if (t + 1) % snapshot_interval == 0 and upto > 0:
                snap = build_snapshot(
                    emb[:upto], labels[:upto], ts[:upto],
                    retrieval_count[:upto], last_retrieved[:upto],
                    ts_now=t, threshold=threshold, embedding_dim=embedding_dim,
                )
                np.savez_compressed(
                    out_seq / f"snapshot_{written:04d}.npz",
                    node_features=snap["node_features"],
                    edge_index=snap["edge_index"],
                    edge_weights=snap["edge_weights"],
                    node_labels=snap["node_labels"],
                    graph_label=np.array([snap["graph_label"]]),
                    timestamp=np.array([snap["timestamp"]]),
                )
                written += 1

    return {
        "mean_poison_coherence": float(np.mean(coh_vals)) if coh_vals else 0.0,
        "mean_poison_internal_density": float(np.mean(dens_vals)) if dens_vals else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/raw/llm_generated_sessions",
                        help="Source dataset (ideally the session-structured one).")
    parser.add_argument("--run-dir", default="experiments/run_current_arch")
    parser.add_argument("--checkpoint", default="phase2_best.pt")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/attack_stealth_sweep.json")
    parser.add_argument("--keep-datasets", action="store_true",
                        help="Keep each rebuilt sweep dataset instead of deleting it.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    embedding_dim = config["simulation"]["embedding_dim"]
    threshold = config["simulation"]["semantic_edge_threshold"]
    snapshot_interval = config["simulation"]["snapshot_interval"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.run_dir) / args.checkpoint
    src_dir = Path(args.data)

    print(f"Using device: {device}")
    print(f"Source dataset: {src_dir}")
    print(f"Checkpoint:     {checkpoint_path}\n")

    results = {}
    work_root = Path(tempfile.mkdtemp(prefix="stealth_sweep_")) if not args.keep_datasets \
        else Path("data/raw/_stealth_sweep")
    work_root.mkdir(parents=True, exist_ok=True)

    for alpha in args.alphas:
        print(f"--- alpha = {alpha:.2f} ---")
        dst = work_root / f"alpha_{alpha:.2f}"
        stats = rebuild_test_split(src_dir, dst, alpha, args.seed,
                                   embedding_dim, threshold, snapshot_interval)
        print(f"  poison coherence: {stats['mean_poison_coherence']:.4f}   "
              f"internal density: {stats['mean_poison_internal_density']:.4f}")

        evaluator = Evaluator(config, str(checkpoint_path), str(dst), device=device,
                              model_cls=TGNDetector, batch_size=1)
        graph_res = evaluator.evaluate_graph_level()
        node_res = evaluator.evaluate_node_level()

        results[f"{alpha:.2f}"] = {**stats, **graph_res, **node_res}
        print(f"  graph AUROC: {graph_res['auroc']:.4f}   AP: {graph_res['average_precision']:.4f}   "
              f"node AUROC: {node_res['node_auroc']:.4f}   node AP: {node_res['node_average_precision']:.4f}\n")

        if not args.keep_datasets:
            shutil.rmtree(dst, ignore_errors=True)

    if not args.keep_datasets:
        shutil.rmtree(work_root, ignore_errors=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 78)
    print(f"{'alpha':>6} {'coherence':>10} {'density':>9} {'g_AUROC':>9} {'g_AP':>8} "
          f"{'n_AUROC':>9} {'n_AP':>8}")
    print("-" * 78)
    for a, r in results.items():
        print(f"{a:>6} {r['mean_poison_coherence']:>10.4f} {r['mean_poison_internal_density']:>9.4f} "
              f"{r['auroc']:>9.4f} {r['average_precision']:>8.4f} "
              f"{r['node_auroc']:>9.4f} {r['node_average_precision']:>8.4f}")
    print("=" * 78)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
