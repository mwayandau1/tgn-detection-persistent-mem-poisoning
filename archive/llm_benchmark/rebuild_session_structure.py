"""
Session-Structure Dataset Re-assembly (Benchmark Hardening, Fix #1)
===================================================================
Rebuilds an existing generated dataset so that **benign** memory has realistic
session/topic structure, WITHOUT any new LLM API calls.

Motivation
----------
In the original generator (`src/data_generation/llm_generator.py`) each benign
record independently samples one of 12 topics (line ~210), while every poisoned
record in a sequence shares ONE `target_topic` and is generated from one of only
THREE hardcoded `POISON_TEMPLATES`. Measured on the shipped dataset:

    mean cosine sim, benign <-> benign : 0.414
    mean cosine sim, poison <-> poison : 0.885
    internal edge density of poison set: 0.978  (near-perfect clique)

So "a dense, mutually-similar cluster" is by construction unique to attacks, and
a GNN can separate the classes on that shortcut alone. Real agent memory is not
topically uniform: users have sessions, so benign memory contains dense
same-topic clusters too.

What this script does
---------------------
Records are append-only and snapshots are cumulative, so the final snapshot of a
sequence contains every record, and the 3 metadata columns invert exactly back to
(timestamp, retrieval_count, last_retrieved). We therefore reconstruct every
record, recover the latent topics by clustering benign embeddings, re-order the
benign records into topic-locked sessions, re-insert the poisoned records at
their ORIGINAL timestamps, and re-emit the snapshot series.

The attack itself is left completely untouched: same poisoned embeddings, same
injection schedule, same labels. Only the benign background changes. That makes
the resulting AUROC delta a clean measurement of how much of the detector's
performance rested on the topic-concentration shortcut.

Usage
-----
    python -m src.experiments.rebuild_session_structure \
        --data data/raw/llm_generated \
        --output data/raw/llm_generated_sessions
"""

import sys
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def recover_records(seq_dir: Path, embedding_dim: int):
    """Reconstruct every MemoryRecord in a sequence from its final snapshot.

    Snapshots are cumulative and append-only, so the last one holds all records.
    The 3 metadata columns are invertible:
        col[emb]   = (ts_snap - rec.timestamp)     / max(ts_snap, 1)
        col[emb+1] = rec.retrieval_count           / 50.0
        col[emb+2] = (ts_snap - rec.last_retrieved)/ max(ts_snap, 1)
    """
    snaps = sorted(seq_dir.glob("snapshot_*.npz"))
    if not snaps:
        return None
    last = np.load(snaps[-1])
    feats = last["node_features"]
    labels = last["node_labels"]
    ts_snap = int(last["timestamp"][0])
    denom = max(ts_snap, 1)

    embeddings = feats[:, :embedding_dim]
    timestamps = np.rint(ts_snap - feats[:, embedding_dim] * denom).astype(int)

    return {
        "embeddings": embeddings,
        "labels": labels.astype(int),
        "timestamps": timestamps,
        "n_snapshots": len(snaps),
        "final_timestamp": ts_snap,
    }


def build_session_order(topic_ids: np.ndarray, rng: np.random.Generator,
                        session_min: int, session_max: int) -> np.ndarray:
    """Re-order benign record indices into topic-locked runs (sessions).

    Returns an index permutation that groups records of the same topic into
    consecutive runs of length ~U(session_min, session_max), emulating a user
    working through one subject for a while before moving on.
    """
    buckets = {}
    for idx, t in enumerate(topic_ids):
        buckets.setdefault(int(t), []).append(idx)
    for t in buckets:
        rng.shuffle(buckets[t])

    order = []
    while buckets:
        topics = list(buckets.keys())
        # Weight topic choice by how many records remain, so large topics do not
        # all pile up at the end of the sequence.
        weights = np.array([len(buckets[t]) for t in topics], dtype=float)
        topic = topics[rng.choice(len(topics), p=weights / weights.sum())]

        run_len = int(rng.integers(session_min, session_max + 1))
        take = buckets[topic][:run_len]
        order.extend(take)
        buckets[topic] = buckets[topic][run_len:]
        if not buckets[topic]:
            del buckets[topic]

    return np.array(order, dtype=int)


def build_snapshot(embeddings: np.ndarray, labels: np.ndarray, timestamps: np.ndarray,
                   retrieval_count: np.ndarray, last_retrieved: np.ndarray,
                   ts_now: int, threshold: float, embedding_dim: int) -> dict:
    """Mirror of LLMMemoryGenerator._build_snapshot, vectorized for edge building."""
    n = embeddings.shape[0]
    denom = max(ts_now, 1)

    node_features = np.zeros((n, embedding_dim + 3))
    node_features[:, :embedding_dim] = embeddings
    node_features[:, embedding_dim] = (ts_now - timestamps) / denom
    node_features[:, embedding_dim + 1] = retrieval_count / 50.0
    node_features[:, embedding_dim + 2] = (ts_now - last_retrieved) / denom

    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim = normed @ normed.T

    iu, ju = np.triu_indices(n, k=1)
    keep = sim[iu, ju] > threshold
    src, dst = iu[keep], ju[keep]
    w = sim[src, dst]

    if src.size:
        edge_index = np.vstack([np.concatenate([src, dst]),
                                np.concatenate([dst, src])]).astype(int)
        edge_weights = np.concatenate([w, w])
    else:
        edge_index = np.zeros((2, 0), dtype=int)
        edge_weights = np.zeros(0)

    return {
        "timestamp": ts_now,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_weights": edge_weights,
        "node_labels": labels.astype(int),
        "graph_label": int(labels.any()),
    }


def rebuild_sequence(rec: dict, topic_ids: np.ndarray, out_dir: Path,
                     rng: np.random.Generator, threshold: float, embedding_dim: int,
                     snapshot_interval: int, session_min: int, session_max: int) -> dict:
    """Re-emit one sequence with benign records re-ordered into sessions."""
    emb, labels, ts = rec["embeddings"], rec["labels"], rec["timestamps"]
    benign_idx = np.where(labels == 0)[0]
    poison_idx = np.where(labels == 1)[0]

    # Re-order ONLY the benign background; the attack is left untouched.
    order = build_session_order(topic_ids, rng, session_min, session_max)
    benign_sorted = benign_idx[order]

    n_benign = len(benign_sorted)
    # Poison records keep their original timestamps (preserving injection spread);
    # map each to the interaction slot it was injected at.
    poison_at = {}
    for p in poison_idx:
        slot = int(np.clip(ts[p], 0, n_benign - 1))
        poison_at.setdefault(slot, []).append(p)

    emb_out, lab_out, ts_out = [], [], []
    retrieval_count, last_retrieved = [], []
    snapshots_written = 0

    for t in range(n_benign):
        src_i = benign_sorted[t]
        emb_out.append(emb[src_i])
        lab_out.append(0)
        ts_out.append(t)
        retrieval_count.append(0)
        last_retrieved.append(0)

        for p in poison_at.get(t, []):
            emb_out.append(emb[p])
            lab_out.append(1)
            ts_out.append(t)
            retrieval_count.append(0)
            last_retrieved.append(0)

        # Re-simulate retrieval with the generator's own rule so the metadata
        # columns stay internally consistent with the new ordering.
        if len(emb_out) > 5 and rng.random() < 0.3:
            k = int(rng.integers(0, len(emb_out)))
            retrieval_count[k] += 1
            last_retrieved[k] = t

        if (t + 1) % snapshot_interval == 0:
            snap = build_snapshot(
                np.array(emb_out), np.array(lab_out), np.array(ts_out),
                np.array(retrieval_count, dtype=float), np.array(last_retrieved, dtype=float),
                ts_now=t, threshold=threshold, embedding_dim=embedding_dim,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / f"snapshot_{snapshots_written:04d}.npz",
                node_features=snap["node_features"],
                edge_index=snap["edge_index"],
                edge_weights=snap["edge_weights"],
                node_labels=snap["node_labels"],
                graph_label=np.array([snap["graph_label"]]),
                timestamp=np.array([snap["timestamp"]]),
            )
            snapshots_written += 1

    return {"n_snapshots": snapshots_written, "n_nodes": len(emb_out)}


def coherence_stats(embeddings: np.ndarray, labels: np.ndarray, order: np.ndarray,
                    window: int, threshold: float) -> dict:
    """Measure whether re-ordered benign records now form clusters as tight as
    the poison clique does. This is the number that says whether the fix bit."""
    benign = embeddings[labels == 0][order]
    normed = benign / (np.linalg.norm(benign, axis=1, keepdims=True) + 1e-8)
    sims = []
    for start in range(0, len(normed) - window, window):
        block = normed[start:start + window]
        s = block @ block.T
        iu = np.triu_indices(len(block), k=1)
        sims.append(s[iu].mean())
    return {"mean_within_session_sim": float(np.mean(sims)) if sims else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/raw/llm_generated")
    parser.add_argument("--output", default="data/raw/llm_generated_sessions")
    parser.add_argument("--num-topics", type=int, default=12,
                        help="k for topic recovery; matches the 12 DOMAIN_PROMPTS topics.")
    parser.add_argument("--session-min", type=int, default=8)
    parser.add_argument("--session-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    embedding_dim = config["simulation"]["embedding_dim"]
    threshold = config["simulation"]["semantic_edge_threshold"]
    snapshot_interval = config["simulation"]["snapshot_interval"]

    data_dir, out_dir = Path(args.data), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.load(open(data_dir / "metadata.json"))
    print(f"Reconstructing {len(metadata)} sequences from {data_dir}...")

    recovered = {}
    for m in tqdm(metadata, desc="Recovering records"):
        rec = recover_records(data_dir / m["sequence_id"], embedding_dim)
        if rec is not None:
            recovered[m["sequence_id"]] = rec

    # --- Recover latent topics by clustering the benign embeddings globally ---
    print(f"\nFitting k-means (k={args.num_topics}) to recover latent topics...")
    pool = np.vstack([r["embeddings"][r["labels"] == 0] for r in recovered.values()])
    rng_global = np.random.default_rng(args.seed)
    sample = pool[rng_global.choice(len(pool), size=min(20000, len(pool)), replace=False)]

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.num_topics, random_state=args.seed, n_init=10).fit(sample)
    sizes = np.bincount(km.predict(sample), minlength=args.num_topics)
    print(f"  cluster sizes (sampled): {sorted(sizes.tolist(), reverse=True)}")

    # --- Rebuild every sequence ---
    print(f"\nRebuilding into {out_dir} (benign re-ordered into sessions; attacks untouched)...")
    before, after = [], []
    for m in tqdm(metadata, desc="Rebuilding"):
        seq_id = m["sequence_id"]
        if seq_id not in recovered:
            continue
        rec = recovered[seq_id]
        benign_emb = rec["embeddings"][rec["labels"] == 0]
        topic_ids = km.predict(benign_emb)

        rng = np.random.default_rng(args.seed + abs(hash(seq_id)) % (2 ** 31))

        # Diagnostic: within-window similarity before vs after re-ordering.
        identity = np.arange(len(benign_emb))
        before.append(coherence_stats(rec["embeddings"], rec["labels"], identity,
                                      window=args.session_max, threshold=threshold)["mean_within_session_sim"])
        order_preview = build_session_order(topic_ids, np.random.default_rng(args.seed), args.session_min, args.session_max)
        after.append(coherence_stats(rec["embeddings"], rec["labels"], order_preview,
                                     window=args.session_max, threshold=threshold)["mean_within_session_sim"])

        rebuild_sequence(rec, topic_ids, out_dir / seq_id, rng, threshold,
                         embedding_dim, snapshot_interval, args.session_min, args.session_max)

    # Splits and metadata carry over unchanged so the test partition is identical.
    shutil.copy(data_dir / "metadata.json", out_dir / "metadata.json")
    shutil.copy(data_dir / "splits.json", out_dir / "splits.json")

    diag = {
        "mean_within_window_sim_before": float(np.mean(before)),
        "mean_within_window_sim_after": float(np.mean(after)),
        "num_topics": args.num_topics,
        "session_min": args.session_min,
        "session_max": args.session_max,
    }
    with open(out_dir / "session_rebuild_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    print("\n=== Benign local coherence (mean pairwise cosine within a "
          f"{args.session_max}-record window) ===")
    print(f"  before re-ordering (i.i.d. topics): {diag['mean_within_window_sim_before']:.4f}")
    print(f"  after  re-ordering (sessions)     : {diag['mean_within_window_sim_after']:.4f}")
    print(f"  poison clique coherence (target)  : ~0.885")
    print(f"\nDone. Rebuilt dataset at {out_dir}")


if __name__ == "__main__":
    main()
