"""
Temporal memory sequences built from AgentPoison's real knowledge base
=====================================================================

Both LLM benchmarks in this repo failed the same control: a logistic regression on a single
record's embedding scores AUROC 1.000, so the label is recoverable from one record's text and
no relational or temporal modelling is being tested (docs/benchmark_difficulty_analysis.md
section 4f). The cause is the labelling rule, not the representation -- a record was poisoned
if and only if it was *generated* from a "harmful advice" instruction, which makes poisoned and
benign text two different distributions.

This constructor removes that by construction. Every record, benign or poisoned, is a genuine
entry from AgentPoison's StrategyQA knowledge base (NeurIPS 2024). The same entry appears as
benign in one sequence and as poisoned in another. What makes a sequence poisoned is not what
any record says but how many mutually-similar records the attacker has placed: a flooding or
Sybil attack on one semantic neighbourhood, visible only in aggregate.

Two poison modes, and the contrast between them is the point:

  placement (default)
      The attacker injects genuine entries drawn from a much tighter neighbourhood than
      ordinary traffic produces. No text is authored or altered. A per-record content probe
      should sit near chance, because the records themselves are ordinary.

  trigger (control)
      Injected entries carry AgentPoison's optimised trigger appended verbatim. This is what
      the published attack actually does, and it reintroduces the shortcut: the trigger is a
      fixed token sequence that exact-match finds with zero false positives (measured: 18/100
      adversarial trajectories carry it, 0/100 benign, substring match AUROC 1.000). Kept so
      the difference between a content-detectable and a structure-only attack can be measured
      rather than asserted.

Output matches the existing snapshot format, so `MemoryTrajectoryDataset`, `train.py` and
`generate_eval_results.py` consume it unchanged.

Usage:
    python -m src.data_generation.agentpoison_sequences --output data/raw/ap_sequences \
        --num-sequences 300 --poison-mode placement
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# AgentPoison's optimised trigger for the DPR retriever on ReAct-StrategyQA, recovered by
# diffing their published adversarial trajectories against the benign ones.
AGENTPOISON_TRIGGER = "turns concurrently luther gates 1911"

KB_FILENAME = "strategyqa_train_filtered.json"


def load_kb(kb_dir: Path) -> List[Dict]:
    path = kb_dir / KB_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with:\n"
            f"  curl -L -o {path} https://raw.githubusercontent.com/AI-secure/"
            f"AgentPoison/master/ReAct/database/{KB_FILENAME}")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    # `description` gives the entry topical body, `question` the retrievable surface form.
    # Together they read like a stored agent memory rather than a bare query.
    for r in rows:
        r["text"] = f"{r['question']} {r.get('description', '')}".strip()
    return rows


class AgentPoisonSequenceBuilder:
    def __init__(self, config: dict, kb_dir: Path, seed: int = 42,
                 poison_mode: str = "placement",
                 benign_pool: int = 60, poison_pool: int = 8,
                 session_length: int = 12):
        self.cfg = config["simulation"]
        self.rng = np.random.default_rng(seed)
        self.poison_mode = poison_mode
        # Ordinary traffic wanders within a broad neighbourhood; the attacker concentrates on a
        # narrow one. The gap between these two pool sizes IS the attack signal, and it is a
        # placement choice rather than anything about the text.
        self.benign_pool = benign_pool
        self.poison_pool = poison_pool
        self.session_length = session_length
        self.threshold = self.cfg["semantic_edge_threshold"]
        self.emb_dim = self.cfg["embedding_dim"]

        self.kb = load_kb(kb_dir)
        self._embed_kb(kb_dir)

    def _embed_kb(self, kb_dir: Path):
        cache = kb_dir / "kb_embeddings.npy"
        if cache.exists():
            self.emb = np.load(cache)
            if self.emb.shape[0] == len(self.kb):
                print(f"Loaded cached embeddings {self.emb.shape}")
                self._finalise_embeddings()
                return
        from sentence_transformers import SentenceTransformer
        print("Encoding knowledge base...")
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        self.emb = enc.encode([r["text"] for r in self.kb], batch_size=64,
                              show_progress_bar=True)
        np.save(cache, self.emb)
        self._finalise_embeddings()

    def _finalise_embeddings(self):
        self.unit = self.emb / (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-8)
        sim = self.unit @ self.unit.T
        np.fill_diagonal(sim, -np.inf)
        # Neighbour ranking is computed once; sequences are assembled by indexing into it.
        self.neighbours = np.argsort(-sim, axis=1)
        self.trigger_emb = None

    def _trigger_embedding(self):
        """Embedding of an entry with the trigger appended, for the control mode."""
        if self.trigger_emb is None:
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [f"{r['text']} {AGENTPOISON_TRIGGER}" for r in self.kb]
            self.trigger_emb = enc.encode(texts, batch_size=64, show_progress_bar=False)
        return self.trigger_emb

    def _benign_indices(self, n: int) -> List[int]:
        """Records arrive in topical sessions, as real agent traffic does."""
        out = []
        while len(out) < n:
            seed_idx = int(self.rng.integers(0, len(self.kb)))
            pool = self.neighbours[seed_idx, :self.benign_pool]
            take = min(self.session_length, n - len(out))
            out.extend(int(i) for i in self.rng.choice(pool, size=take, replace=False))
        return out[:n]

    def _poison_indices(self, k: int, drift_step: int = 0) -> List[int]:
        """Entries drawn from one narrow neighbourhood.

        With drift_step = 0 the neighbourhood is fixed: this is the attacker, whose planted
        cluster has a single centre. With drift_step > 0 the neighbourhood re-centres on the
        last entry drawn every drift_step records, so the cluster is equally tight at any
        instant but wanders -- which is what a benign user pursuing one deepening interest
        looks like.
        """
        cur = int(self.rng.integers(0, len(self.kb)))
        if drift_step <= 0:
            pool = self.neighbours[cur, :self.poison_pool]
            return [int(i) for i in self.rng.choice(pool, size=k, replace=k > len(pool))]
        out: List[int] = []
        while len(out) < k:
            pool = self.neighbours[cur, :self.poison_pool]
            take = min(drift_step, k - len(out), len(pool))
            picked = [int(i) for i in self.rng.choice(pool, size=take, replace=False)]
            out.extend(picked)
            cur = picked[-1]
        return out[:k]

    def _place(self, idxs: List[int], rng: np.random.Generator, n_interactions: int,
               k: int, spread: Optional[int], drift_step: int) -> np.ndarray:
        """Overwrite k slots with entries from a narrow neighbourhood; return the slot indices.

        Placement REPLACES benign interactions rather than appending, so the record count
        cannot leak the label -- appending is what made the original benchmark solvable by
        counting alone (section 2.3 of the analysis document).

        `spread` is the window of interactions the k records land in. A small spread is a
        burst (the attack); a spread covering the whole sequence is the same cluster arriving
        gradually (the matched benign control). Burstiness is therefore the only thing that
        separates them when drift_step and k are also matched.
        """
        if spread is None:
            start, width = 0, n_interactions
        else:
            start = int(rng.integers(40, max(41, n_interactions - k - 10)))
            width = min(n_interactions - start, max(spread, k))
        slots = start + rng.choice(width, size=min(k, width), replace=False)
        picks = self._poison_indices(len(slots), drift_step=drift_step)
        for slot, p in zip(sorted(slots), picks):
            idxs[int(slot)] = p
        return np.sort(slots).astype(int)

    def build_sequence(self, n_interactions: int, n_injected: int,
                       seq_id: str, out_dir: Path, rng: np.random.Generator,
                       benign_burst: int = 0, benign_drift_step: int = 0,
                       benign_spread: Optional[int] = None,
                       attack_spread_mult: float = 3.0,
                       poison_mode: Optional[str] = None) -> Dict:
        poisoned = n_injected > 0
        mode = poison_mode or self.poison_mode
        idxs = self._benign_indices(n_interactions)
        labels = np.zeros(n_interactions, dtype=int)

        if poisoned:
            # The attack's burst width, as a multiple of its size. At 3.0 the attack spreads k
            # records over 3k interactions -- roughly six snapshots -- against a benign cluster
            # spread over 120. Measured on ap_matched_2k, that contrast is too weak to survive
            # the backdrop: sequence-level growth-rate features scored 0.47-0.50, chance. The
            # backdrop adds ten records per snapshot, so a one-or-two record per snapshot
            # difference in cluster growth is buried. A multiplier near 1.0 lands the whole
            # attack inside one or two snapshots, which is what makes burstiness detectable.
            spread = max(int(round(n_injected * attack_spread_mult)), n_injected)
            slots = self._place(idxs, rng, n_interactions, n_injected,
                                spread=spread, drift_step=0)
            labels[slots] = 1
        elif benign_burst > 0:
            # A clean sequence carrying a tight cluster of the same size. Its labels stay 0,
            # so the final snapshot looks like an attack to any density statistic while the
            # sequence is genuinely benign.
            self._place(idxs, rng, n_interactions, benign_burst,
                        spread=benign_spread, drift_step=benign_drift_step)

        trig = self._trigger_embedding() if (poisoned and mode == "trigger") else None

        records, snaps = [], 0
        interval = self.cfg["snapshot_interval"]
        retrieval_count = np.zeros(n_interactions)
        last_retrieved = np.zeros(n_interactions)

        for t in range(n_interactions):
            records.append(t)
            # Retrieval is label-independent: biasing it toward poisoned records would put the
            # label straight into the metadata columns.
            if len(records) > 5 and rng.random() < 0.3:
                j = int(rng.integers(0, len(records)))
                retrieval_count[j] += 1
                last_retrieved[j] = t

            if (t + 1) % interval == 0:
                n = len(records)
                feats = np.zeros((n, self.emb_dim + 3))
                for i in range(n):
                    src = trig if (trig is not None and labels[i] == 1) else self.emb
                    feats[i, :self.emb_dim] = src[idxs[i]]
                    feats[i, self.emb_dim] = (t - i) / max(t, 1)
                    feats[i, self.emb_dim + 1] = retrieval_count[i] / 50.0
                    feats[i, self.emb_dim + 2] = (t - last_retrieved[i]) / max(t, 1)

                e = feats[:, :self.emb_dim]
                u = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
                sim = u @ u.T
                iu = np.triu_indices(n, k=1)
                mask = sim[iu] > self.threshold
                si, sj = iu[0][mask], iu[1][mask]
                w = sim[si, sj]
                edge_index = (np.concatenate([np.stack([si, sj]), np.stack([sj, si])], axis=1)
                              if si.size else np.zeros((2, 0), dtype=int))
                weights = np.concatenate([w, w]) if si.size else np.zeros(0)

                seq_dir = out_dir / seq_id
                seq_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    seq_dir / f"snapshot_{snaps:04d}.npz",
                    node_features=feats,
                    edge_index=edge_index.astype(int),
                    edge_weights=weights,
                    node_labels=labels[:n].copy(),
                    graph_label=np.array([int(labels[:n].any())]),
                    timestamp=np.array([t]),
                )
                snaps += 1

        return {"sequence_id": seq_id,
                "label": "poisoned" if poisoned else "clean",
                "attack_type": (f"flood_{mode}" if poisoned
                                else ("benign_burst" if benign_burst > 0 else None)),
                "num_snapshots": snaps,
                "num_injected": int(n_injected),
                "benign_burst": int(benign_burst)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--kb-dir", default="data/raw/agentpoison")
    ap.add_argument("--output", default="data/raw/ap_sequences")
    ap.add_argument("--num-sequences", type=int, default=300)
    ap.add_argument("--interactions-per-sequence", type=int, default=None)
    ap.add_argument("--poison-mode", choices=["placement", "trigger"], default="placement")
    ap.add_argument("--injection-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                    default=[5, 20],
                    help="Attack size drawn per sequence, so one dataset spans a difficulty "
                         "range and detection can be reported binned by attack size.")
    ap.add_argument("--benign-pool", type=int, default=60,
                    help="Neighbourhood width ordinary sessions draw from. Larger = looser "
                         "benign clusters = easier to detect the attack.")
    ap.add_argument("--poison-pool", type=int, default=8,
                    help="Neighbourhood width the attacker floods. Smaller = tighter planted "
                         "cluster = easier to detect. This pair is the difficulty dial.")
    ap.add_argument("--session-length", type=int, default=12)
    ap.add_argument("--attack-ratio", type=float, default=0.4)
    ap.add_argument("--trigger-fraction", type=float, default=0.0,
                    help="Share of poisoned sequences using the trigger mode instead of "
                         "placement, so one dataset carries both attack types and the "
                         "per-attack-type breakdown has more than one row.")
    ap.add_argument("--matched-control", type=float, default=0.0,
                    help="Share of CLEAN sequences given a benign tight cluster of the same "
                         "size as an attack. Its final snapshot carries the same density a "
                         "flood produces, so final-state features lose their grip and only "
                         "the arrival pattern separates the classes. This is the "
                         "final-state-matched benchmark.")
    ap.add_argument("--benign-drift-step", type=int, default=0,
                    help="0 keeps the benign cluster's centre fixed, matching the attack "
                         "exactly. >0 lets it re-centre every N records, so the cluster is "
                         "as tight at any instant but wanders over the sequence.")
    ap.add_argument("--attack-spread-mult", type=float, default=3.0,
                    help="Attack burst width as a multiple of attack size. 1.0 lands the whole "
                         "injection inside one or two snapshots; 3.0 spreads it over about six "
                         "and, measured on ap_matched_2k, leaves no detectable growth-rate "
                         "signal once the benign backdrop is accounted for.")
    ap.add_argument("--benign-spread", type=int, default=None,
                    help="Interactions the benign cluster arrives over. Default (unset) is "
                         "the whole sequence, against the attack's burst of 3x its size -- "
                         "so burstiness alone distinguishes them.")
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.interactions_per_sequence:
        config["simulation"]["interactions_per_sequence"] = args.interactions_per_sequence
    n_inter = config["simulation"]["interactions_per_sequence"]

    builder = AgentPoisonSequenceBuilder(
        config, Path(args.kb_dir), seed=args.seed, poison_mode=args.poison_mode,
        benign_pool=args.benign_pool, poison_pool=args.poison_pool,
        session_length=args.session_length)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    n_pois = int(args.num_sequences * args.attack_ratio)
    n_clean = args.num_sequences - n_pois
    n_trig = int(round(n_pois * args.trigger_fraction))
    n_matched = int(round(n_clean * args.matched_control))
    print(f"Building {n_clean} clean and {n_pois} poisoned sequences "
          f"({n_inter} interactions each)")
    print(f"  poisoned: {n_pois - n_trig} placement, {n_trig} trigger")
    print(f"  clean:    {n_clean - n_matched} ordinary, {n_matched} with a matched benign "
          f"burst (drift step {args.benign_drift_step}, spread "
          f"{args.benign_spread if args.benign_spread else 'whole sequence'})")

    meta = []
    for i in tqdm(range(n_clean), desc="clean"):
        rng = np.random.default_rng(args.seed + i)
        burst = 0
        if i < n_matched:
            # Drawn from the same range the attacks are, so the two clusters are matched in
            # size as well as in tightness.
            burst = int(rng.integers(args.injection_range[0], args.injection_range[1] + 1))
        meta.append(builder.build_sequence(
            n_inter, 0, f"clean_{i:04d}", out, rng, benign_burst=burst,
            benign_drift_step=args.benign_drift_step, benign_spread=args.benign_spread))
    for i in tqdm(range(n_pois), desc="poisoned"):
        rng = np.random.default_rng(args.seed + 10000 + i)
        k = int(rng.integers(args.injection_range[0], args.injection_range[1] + 1))
        mode = "trigger" if i < n_trig else "placement"
        meta.append(builder.build_sequence(
            n_inter, k, f"poisoned_{mode}_{i:04d}", out, rng, poison_mode=mode,
            attack_spread_mult=args.attack_spread_mult))

    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    idx = np.arange(len(meta))
    np.random.default_rng(args.seed).shuffle(idx)
    tr = int(len(meta) * args.train_ratio)
    va = tr + int(len(meta) * args.val_ratio)
    splits = {"train": [meta[int(i)]["sequence_id"] for i in idx[:tr]],
              "val": [meta[int(i)]["sequence_id"] for i in idx[tr:va]],
              "test": [meta[int(i)]["sequence_id"] for i in idx[va:]]}
    with open(out / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nDone. {len(meta)} sequences -> {out}")
    print(f"  Splits: {len(splits['train'])} train, {len(splits['val'])} val, "
          f"{len(splits['test'])} test")
    print("\nNow run the gate before training on this:")
    print(f"  python -m src.experiments.separability_probe --data {out}")
    print("  node_content near 0.5 means the label is NOT in the text and a graph model is "
          "actually being tested.")


if __name__ == "__main__":
    main()
