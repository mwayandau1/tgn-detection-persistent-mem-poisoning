"""
Synthetic Memory Graph Dataset Generator (Stage A) — hardened
==============================================================
Generates labeled memory graph trajectories WITHOUT requiring an LLM or any API
calls. Uses synthetic embeddings to simulate memory records and poisoning attacks.

This is a restored and substantially hardened version of the Stage A generator
removed in commit a2b0ad4. The original produced a near-trivially separable
benchmark for three independent reasons, all of which are now controllable:

1. **The poison cluster was far tighter than the benign background.** Benign
   records were drawn at sigma=0.30 around a topic centre, poisoned ones at
   sigma=(1-coherence)*0.5 — i.e. 0.05 at the configured coherence of 0.90, six
   times tighter. That guarantees a dense clique no benign region can imitate.
   Poison spread is now expressed *relative to* benign spread, so the attacker can
   be made as diffuse as ordinary traffic.

2. **Benign topics were i.i.d. per record.** A topically coherent cluster was
   therefore unique to attacks by construction. Benign records are now optionally
   drawn in topic-locked sessions, so clean memory contains dense same-topic
   clusters too — the hard negatives the benchmark previously lacked.

3. **Retrieval bias leaked the label.** The original preferentially retrieved
   poisoned records (`rng.random() < 0.4`), writing the ground-truth label
   straight into the `retrieval_count` and `retrieval_recency` node features. A
   detector could read poisoning off metadata without looking at the graph at all.
   This is now OFF by default and must be opted into explicitly.

Difficulty is set with `--difficulty {easy,medium,hard}` (or by overriding the
individual knobs). `easy` reproduces the original removed generator's behaviour,
including the metadata leak, so the two can be compared directly.

Usage:
    python -m src.data_generation.synthetic_generator \
        --config configs/default.yaml --output data/raw/synthetic_hard --difficulty hard
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from tqdm import tqdm

from .types import MemoryRecord, GraphSnapshot


# --- Embedding model -------------------------------------------------------
#
# The original generator drew records as unit(topic_centre + N(0, 0.3^2 I)) in 384
# dimensions. That is broken: the noise norm is 0.3*sqrt(384) ~= 5.9 against a
# unit-length centre, so the topic signal is swamped and every record is
# effectively a random direction. Measured on its output: mean pairwise cosine
# 0.003 and ZERO edges above the 0.6 threshold — the graphs had no structure at all.
#
# Records are therefore built from three orthogonal parts,
#
#     raw = g * u_global + tau * u_topic + sigma * noise,   record = raw / ||raw||
#
# and the coefficients are solved from *target cosine similarities*, which is both
# dimension-safe and directly comparable to the real LLM-generated dataset:
#
#     cross-topic similarity  = g^2                     (the shared-domain baseline)
#     within-topic similarity = g^2 + tau^2
#     noise budget            = sigma^2 * d = 1 - within_topic
#
# Reference values measured on data/raw/llm_generated: benign-benign 0.414,
# poison-poison 0.885, poison internal edge density 0.978.

# What actually sets difficulty is the gap between how tightly the ATTACKER clusters
# and how tightly ordinary SAME-TOPIC traffic clusters, measured against the cosine
# edge threshold (`semantic_edge_threshold`, 0.6 by default):
#
#   * topic_similarity below the threshold -> benign sessions never form edges, so any
#     dense component is an attack by construction. This is the regime the LLM-generated
#     dataset is in, and it is why that benchmark is near-trivial.
#   * topic_similarity above the threshold -> clean memory grows dense same-topic
#     components of its own, and "dense component" stops being diagnostic. The detector
#     must then separate the attacker's cluster from ordinary ones by degree only.
#
# So `hard` is not merely "lower coherence" — it is a small coherence-minus-topic gap
# with both sides above the edge threshold.
# DIFFICULTY IS THE COHERENCE GAP: `coherence` minus `topic_similarity`, i.e. how much
# more similar the attacker's planted records are to each other than ordinary same-topic
# records are. This is a CONTINUOUS separation and it is what a detector actually reads,
# because edge weights carry the raw cosine value.
#
# An earlier calibration used the *thresholded* density gap (the fraction of pairs
# clearing `semantic_edge_threshold`) and got this badly wrong. At topic_similarity 0.59
# against coherence 0.75, most pairs on BOTH sides clear the 0.6 threshold, so the
# density gap reads a deceptively hard 0.148 — while the underlying weights, 0.59 versus
# 0.75, separate cleanly. A detector given edge-weight statistics reached graph AUROC
# 0.997 and node AP 0.847 on that supposedly "hard" setting, beating the real
# LLM-generated benchmark. Thresholding discards exactly the magnitude information the
# task turns on, so never calibrate on density alone.
#
# Both quantities are reported by the separability report; `coherence_gap` is the one to
# tune. Known reference points, all with session structure and no leaks:
#
#     coherence_gap   ~0.44   -> trivial
#     coherence_gap   ~0.16   -> graph AUROC 0.997 (still trivial, 40 epochs)
#     coherence_gap    0.00   -> unlearnable (graph AUROC 0.475, F1 0.0)
#
# The usable band therefore lies somewhere below 0.16 and above 0.00, and the settings
# below are a first bisection of it rather than validated calibrations. Always read the
# separability report, and treat any preset as provisional until a training run confirms
# it.
DIFFICULTY_PRESETS = {
    # Reproduces the real LLM dataset's geometry (benign ~0.41 / poison ~0.885) plus the
    # original generator's two shortcuts: i.i.d. benign topics and the retrieval leak.
    "easy": {"session_length": 0, "retrieval_bias": 0.4,
             "topic_similarity": 0.45, "coherence": 0.885},   # gap 0.435
    # Attacker clearly tighter than a benign session, but far less so than `easy`.
    "medium": {"session_length": 12, "retrieval_bias": 0.0,
               "topic_similarity": 0.59, "coherence": 0.69},  # gap 0.10
    # Attacker only marginally tighter than ordinary same-topic traffic.
    "hard": {"session_length": 12, "retrieval_bias": 0.0,
             "topic_similarity": 0.59, "coherence": 0.63},    # gap 0.04
    # The calibrated reporting point: the coherence gap at which the detector lands in the
    # 0.89-0.96 graph-AUROC band. Named separately because the three presets above were
    # fixed before the difficulty curve was measured and their names no longer track the
    # measured difficulty ("hard" is in fact close to unlearnable). See REPRODUCE.md.
    "benchmark": {"session_length": 12, "retrieval_bias": 0.0,
                  "topic_similarity": 0.59, "coherence": 0.715},  # gap 0.125
    # Retained for reference: proved unlearnable (graph AUROC 0.475). Not a benchmark.
    "degenerate": {"session_length": 12, "retrieval_bias": 0.0,
                   "topic_similarity": 0.62, "coherence": 0.62},
}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


class SyntheticMemorySimulator:
    """Simulates an LLM agent's long-term memory with poisoning attacks."""

    def __init__(self, config: dict, seed: int = 42,
                 num_clusters: int = 10,
                 global_similarity: float = 0.35,
                 topic_similarity: float = 0.55,
                 session_length: int = 12,
                 retrieval_bias: float = 0.0,
                 coherence_override: Optional[float] = None,
                 poison_offset: float = 0.0,
                 constant_length: bool = True):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.sim_cfg = config["simulation"]
        self.atk_cfg = config["attacks"]
        self.embedding_dim = self.sim_cfg["embedding_dim"]

        if not 0.0 <= global_similarity < topic_similarity < 1.0:
            raise ValueError(
                "require 0 <= global_similarity < topic_similarity < 1, got "
                f"{global_similarity} and {topic_similarity}")

        self.num_clusters = num_clusters
        self.global_similarity = global_similarity
        self.topic_similarity = topic_similarity
        self.session_length = session_length
        self.retrieval_bias = retrieval_bias
        self.coherence_override = coherence_override
        self.poison_offset = poison_offset
        self.constant_length = constant_length

        # Coefficients solved from the target similarities (see module header).
        self.g = float(np.sqrt(global_similarity))
        self.tau = float(np.sqrt(topic_similarity - global_similarity))
        self.sigma = float(np.sqrt((1.0 - topic_similarity) / self.embedding_dim))

        self.basis = self._generate_basis()
        self.u_global = self.basis[0]

    # ---------------------------------------------------------------- embeddings

    def _generate_basis(self) -> np.ndarray:
        """Orthonormal directions: one shared 'domain' axis plus one per topic.

        Random directions in 384-d are already near-orthogonal, but taking an exact
        orthonormal basis (via QR) makes the similarity algebra in the module header
        hold precisely rather than approximately.
        """
        a = self.rng.standard_normal((self.embedding_dim, self.num_clusters + 1))
        q, _ = np.linalg.qr(a)
        return q.T[: self.num_clusters + 1]

    def _generate_record_embedding(self, cluster_id: int, sigma: Optional[float] = None) -> np.ndarray:
        """Generate a record embedding near a topic centre."""
        s = self.sigma if sigma is None else sigma
        raw = (self.g * self.u_global
               + self.tau * self.basis[1 + cluster_id]
               + s * self.rng.standard_normal(self.embedding_dim))
        return _unit(raw)

    def _poison_spread(self, coherence: float) -> float:
        """Noise scale giving planted records a mean pairwise cosine of `coherence`.

        Inverting  cos = (g^2 + tau^2) / (g^2 + tau^2 + sigma_p^2 * d):

            sigma_p = sqrt( (g^2 + tau^2) * (1/cos - 1) / d )

        coherence -> 1.0 : a tight clique, trivially separable
        coherence -> topic_similarity : as diffuse as ordinary same-topic traffic
        """
        coherence = float(np.clip(coherence, 1e-3, 0.999))
        structured = self.g ** 2 + self.tau ** 2
        return float(np.sqrt(structured * (1.0 / coherence - 1.0) / self.embedding_dim))

    def _poison_centroid(self, target_cluster: int) -> np.ndarray:
        """The structured (un-normalised) vector the attacker clusters around.

        Returned un-normalised, with norm sqrt(g^2 + tau^2), so that adding
        `sigma_p * noise` reproduces the similarity algebra in the module header
        exactly.

        With `poison_offset` = 0 the attacker sits on the topic centre, i.e. in the
        densest part of the benign distribution (maximum camouflage). A positive
        offset displaces it into sparser space, making it easier to spot.
        """
        base = self.g * self.u_global + self.tau * self.basis[1 + target_cluster]
        if self.poison_offset <= 0.0:
            return base
        u = _unit(base)
        d = self.rng.standard_normal(self.embedding_dim)
        d = d - np.dot(d, u) * u          # component orthogonal to the centre
        return base + self.poison_offset * np.linalg.norm(base) * _unit(d)

    # ------------------------------------------------------------------- graphs

    def _build_graph_snapshot(self, records: List[MemoryRecord], timestamp: int) -> GraphSnapshot:
        """Build a graph snapshot from current memory records."""
        n = len(records)
        if n == 0:
            return GraphSnapshot(
                timestamp=timestamp, node_ids=[],
                node_features=np.zeros((0, self.embedding_dim + 3)),
                edge_index=np.zeros((2, 0), dtype=int),
                edge_weights=np.zeros(0),
                node_labels=np.zeros(0, dtype=int),
                graph_label=0,
            )

        # Node features: [embedding, recency, retrieval_freq, retrieval_recency].
        # NOTE: provenance_flag was removed — it leaked the label directly.
        emb_dim = self.embedding_dim
        node_features = np.zeros((n, emb_dim + 3))
        node_labels = np.zeros(n, dtype=int)
        node_ids = []
        denom = max(timestamp, 1)

        for i, rec in enumerate(records):
            node_features[i, :emb_dim] = rec.embedding
            node_features[i, emb_dim] = (timestamp - rec.timestamp) / denom
            node_features[i, emb_dim + 1] = rec.retrieval_count / 50.0
            node_features[i, emb_dim + 2] = (timestamp - rec.last_retrieved) / denom
            node_labels[i] = 1 if rec.is_poisoned else 0
            node_ids.append(rec.record_id)

        # Edges by semantic similarity (vectorized; the original used a Python
        # double loop, which is O(n^2) interpreted and dominates generation time).
        threshold = self.sim_cfg["semantic_edge_threshold"]
        embeddings = node_features[:, :emb_dim]
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

        return GraphSnapshot(
            timestamp=timestamp, node_ids=node_ids,
            node_features=node_features, edge_index=edge_index,
            edge_weights=edge_weights, node_labels=node_labels,
            graph_label=1 if any(r.is_poisoned for r in records) else 0,
        )

    # ---------------------------------------------------------------- behaviour

    def _next_benign_cluster(self, state: dict) -> int:
        """Pick the topic for the next benign record.

        With `session_length` > 0, benign records arrive in topic-locked runs, so
        clean memory develops dense same-topic clusters of its own. With 0, topics
        are i.i.d. per record (the original behaviour), which makes any coherent
        cluster a giveaway for an attack.
        """
        if self.session_length <= 0:
            return int(self.rng.integers(0, self.num_clusters))

        if state["run_left"] <= 0:
            state["topic"] = int(self.rng.integers(0, self.num_clusters))
            lo = max(1, self.session_length // 2)
            hi = max(lo + 1, self.session_length * 2)
            state["run_left"] = int(self.rng.integers(lo, hi))
        state["run_left"] -= 1
        return state["topic"]

    def _simulate_retrieval(self, records: List[MemoryRecord], t: int):
        """Simulate a retrieval event.

        `retrieval_bias` > 0 preferentially retrieves poisoned records. The original
        generator hardcoded this at 0.4, which wrote the ground-truth label straight
        into the retrieval metadata features. It defaults to 0.0 here; raise it only
        to reproduce that leak deliberately as an ablation.
        """
        if len(records) <= 5 or self.rng.random() >= 0.3:
            return
        idx = None
        if self.retrieval_bias > 0.0:
            poisoned = [i for i, r in enumerate(records) if r.is_poisoned]
            if poisoned and self.rng.random() < self.retrieval_bias:
                idx = int(self.rng.choice(poisoned))
        if idx is None:
            idx = int(self.rng.integers(0, len(records)))
        records[idx].retrieval_count += 1
        records[idx].last_retrieved = t

    # ---------------------------------------------------------------- sequences

    def generate_clean_sequence(self, num_interactions: int) -> List[GraphSnapshot]:
        """Generate a clean (no attack) interaction sequence."""
        records: List[MemoryRecord] = []
        snapshots: List[GraphSnapshot] = []
        snapshot_interval = self.sim_cfg["snapshot_interval"]
        session = {"topic": 0, "run_left": 0}

        for t in range(num_interactions):
            cluster = self._next_benign_cluster(session)
            records.append(MemoryRecord(
                record_id=len(records),
                embedding=self._generate_record_embedding(cluster),
                timestamp=t, provenance="user", is_poisoned=False,
                semantic_cluster=cluster,
            ))

            self._simulate_retrieval(records, t)

            if (t + 1) % snapshot_interval == 0:
                snapshots.append(self._build_graph_snapshot(records, t))

        return snapshots

    def generate_poisoned_sequence(self, num_interactions: int, attack_type: dict) -> List[GraphSnapshot]:
        """Generate a sequence with a poisoning attack injected."""
        records: List[MemoryRecord] = []
        snapshots: List[GraphSnapshot] = []
        snapshot_interval = self.sim_cfg["snapshot_interval"]
        session = {"topic": 0, "run_left": 0}

        target_cluster = int(self.rng.integers(0, self.num_clusters))
        num_injected = attack_type["num_injected_records"]
        injection_spread = attack_type["injection_spread"]

        coherence = (self.coherence_override if self.coherence_override is not None
                     else attack_type.get("semantic_coherence", 0.85))
        sigma_p = self._poison_spread(coherence)
        centroid = self._poison_centroid(target_cluster)

        injection_start = int(self.rng.integers(50, max(51, num_interactions - injection_spread - 20)))

        # Which interactions carry a poisoned write.
        #
        # With `constant_length` (the default) a poisoned interaction REPLACES its
        # benign one, so a poisoned sequence holds exactly as many records as a clean
        # one. The alternative — appending poisoned records on top of an unchanged
        # benign stream — makes the record count itself a perfect giveaway: clean
        # sequences end at exactly `num_interactions`, poisoned ones at
        # `num_interactions + num_injected`, which separates the classes at AUROC 1.0
        # on node count alone. It also fixes benign growth at exactly
        # `snapshot_interval` records per snapshot, so any snapshot that grows faster
        # is under attack by construction — a mechanical tell visible inside a single
        # temporal window, with nothing semantic about it.
        if injection_spread <= 1:
            slots = [injection_start + i for i in range(num_injected)]
        else:
            slots = sorted(self.rng.integers(
                injection_start, injection_start + injection_spread, size=num_injected).tolist())
        if self.constant_length:
            slots = sorted(set(s for s in slots if s < num_interactions))
        poisoned_slots = set(slots)

        for t in range(num_interactions):
            if t in poisoned_slots:
                noise = self.rng.standard_normal(self.embedding_dim) * sigma_p
                records.append(MemoryRecord(
                    record_id=len(records),
                    embedding=_unit(centroid + noise),
                    timestamp=t, provenance="attacker", is_poisoned=True,
                    semantic_cluster=target_cluster,
                ))
            if t not in poisoned_slots or not self.constant_length:
                cluster = self._next_benign_cluster(session)
                records.append(MemoryRecord(
                    record_id=len(records),
                    embedding=self._generate_record_embedding(cluster),
                    timestamp=t, provenance="user", is_poisoned=False,
                    semantic_cluster=cluster,
                ))

            self._simulate_retrieval(records, t)

            if (t + 1) % snapshot_interval == 0:
                snapshots.append(self._build_graph_snapshot(records, t))

        return snapshots

    # ------------------------------------------------------------------ dataset

    def generate_dataset(self, output_dir: str) -> dict:
        """Generate the full dataset of labeled trajectories."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_sequences = self.sim_cfg["num_sequences"]
        num_interactions = self.sim_cfg["interactions_per_sequence"]
        attack_ratio = self.atk_cfg["attack_ratio"]
        attack_types = self.atk_cfg["types"]

        num_poisoned = int(num_sequences * attack_ratio)
        num_clean = num_sequences - num_poisoned
        metadata = []
        final_snaps = []

        print(f"Generating {num_clean} clean sequences...")
        for i in tqdm(range(num_clean)):
            snapshots = self.generate_clean_sequence(num_interactions)
            seq_id = f"clean_{i:04d}"
            self._save_sequence(snapshots, output_path / seq_id)
            final_snaps.append(snapshots[-1])
            metadata.append({"sequence_id": seq_id, "label": "clean",
                             "attack_type": None, "num_snapshots": len(snapshots)})

        print(f"Generating {num_poisoned} poisoned sequences...")
        for i in tqdm(range(num_poisoned)):
            attack_type = attack_types[i % len(attack_types)]
            snapshots = self.generate_poisoned_sequence(num_interactions, attack_type)
            seq_id = f"poisoned_{attack_type['name']}_{i:04d}"
            self._save_sequence(snapshots, output_path / seq_id)
            final_snaps.append(snapshots[-1])
            metadata.append({"sequence_id": seq_id, "label": "poisoned",
                             "attack_type": attack_type["name"], "num_snapshots": len(snapshots)})

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self._create_splits(metadata, output_path)

        print(f"\nDataset generated: {len(metadata)} sequences saved to {output_path}")
        print(f"  Clean: {num_clean}, Poisoned: {num_poisoned}")

        report = self._separability_report(final_snaps)
        report["settings"] = {
            "num_clusters": self.num_clusters,
            "global_similarity": self.global_similarity,
            "topic_similarity": self.topic_similarity,
            "session_length": self.session_length,
            "retrieval_bias": self.retrieval_bias,
            "coherence_override": self.coherence_override,
            "poison_offset": self.poison_offset,
            "constant_length": self.constant_length,
        }
        with open(output_path / "generation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

    @staticmethod
    def _best_benign_cluster_density(sim: np.ndarray, benign_mask: np.ndarray,
                                     k: int, threshold: float) -> float:
        """Internal edge density of the tightest k-record benign neighbourhood.

        For each benign node, take its k-1 most similar benign neighbours and measure
        that group's internal density; return the best such group. This is the
        strongest 'hard negative' the clean data actually contains, so the difference
        between it and the poison set's density is the signal the detector has to work
        with.
        """
        idx = np.where(benign_mask)[0]
        if len(idx) < k or k < 2:
            return 0.0
        sub = sim[np.ix_(idx, idx)]
        np.fill_diagonal(sub, -np.inf)

        # Cap the scan for speed; candidates are ranked by how tight their local
        # neighbourhood already is, so the true maximum is almost always included.
        order = np.argsort(-np.sort(sub, axis=1)[:, k - 2])[:64]
        best = 0.0
        iu = np.triu_indices(k, k=1)
        for i in order:
            nb = np.argpartition(-sub[i], k - 2)[:k - 1]
            group = np.concatenate([[i], nb])
            vals = sub[np.ix_(group, group)][iu]
            best = max(best, float((vals > threshold).mean()))
        return best

    def _separability_report(self, final_snaps: List[GraphSnapshot]) -> dict:
        """Measure how separable the generated task is, before any training.

        The structural signal the detector keys on is the poison set forming a dense
        connected component above the cosine edge threshold. Reporting it here gives
        immediate feedback on difficulty without paying for a training run.
        """
        threshold = self.sim_cfg["semantic_edge_threshold"]
        emb_dim = self.embedding_dim
        benign_sims, benign_dens, poison_sims, poison_dens, retr_gap = [], [], [], [], []

        for snap in final_snaps:
            x = snap.node_features[:, :emb_dim]
            labels = snap.node_labels
            n = x.shape[0]
            if n < 5:
                continue
            nx_ = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            sim = nx_ @ nx_.T

            b = labels == 0
            p = labels == 1
            if b.sum() > 1:
                sb = sim[np.ix_(b, b)]
                iu = np.triu_indices(int(b.sum()), k=1)
                vals_b = sb[iu]
                benign_sims.append(float(vals_b.mean()))

            if p.sum() > 1:
                sp = sim[np.ix_(p, p)]
                iu = np.triu_indices(int(p.sum()), k=1)
                vals = sp[iu]
                poison_sims.append(float(vals.mean()))
                poison_dens.append(float((vals > threshold).mean()))

                # The fair comparison: the DENSEST benign neighbourhood of the same
                # size. Overall benign density is diluted by cross-topic pairs and so
                # flatters the benchmark; what the detector must actually do is tell
                # the attacker's cluster apart from the tightest ordinary one.
                benign_dens.append(self._best_benign_cluster_density(
                    sim, b, int(p.sum()), threshold))

                # Does retrieval metadata leak the label?
                retr = snap.node_features[:, emb_dim + 1]
                retr_gap.append(float(retr[p].mean() - retr[b].mean()))

        def m(v):
            return float(np.mean(v)) if v else 0.0

        coherence_gap = self.topic_similarity_effective(m(poison_sims))
        report = {
            "benign_mean_pairwise_cosine": m(benign_sims),
            "benign_best_cluster_density": m(benign_dens),
            "poison_mean_pairwise_cosine": m(poison_sims),
            "poison_internal_edge_density": m(poison_dens),
            "coherence_gap": coherence_gap,
            "density_gap": m(poison_dens) - m(benign_dens),
            "note": "coherence_gap = poison mean pairwise cosine minus the configured "
                    "same-topic similarity. This CONTINUOUS separation is what governs "
                    "difficulty. density_gap thresholds the same quantity and so discards "
                    "magnitude; it misled an earlier calibration and is reported only for "
                    "reference.",
            "retrieval_metadata_gap": m(retr_gap),
        }

        print("\n=== Separability report (pre-training) ===")
        print(f"  benign  mean pairwise cosine : {report['benign_mean_pairwise_cosine']:.4f}")
        print(f"  benign  same-topic similarity: {self.topic_similarity:.4f}  (configured)")
        print(f"  poison  mean pairwise cosine : {report['poison_mean_pairwise_cosine']:.4f}")
        print(f"  COHERENCE GAP                : {coherence_gap:+.4f}   <-- difficulty dial")
        print(f"  benign  BEST cluster density : {report['benign_best_cluster_density']:.4f}")
        print(f"  poison  internal edge density: {report['poison_internal_edge_density']:.4f}")
        print(f"  density gap (reference only) : {report['density_gap']:+.4f}")
        print(f"  retrieval metadata gap       : {report['retrieval_metadata_gap']:+.4f} "
              f"({'LEAKS the label' if abs(report['retrieval_metadata_gap']) > 0.005 else 'no leak'})")
        print("\n  Reference points (session structure, no leaks):")
        print("    coherence gap ~0.44 -> trivial")
        print("    coherence gap ~0.16 -> graph AUROC 0.997 at 40 epochs (still trivial)")
        print("    coherence gap  0.00 -> unlearnable (graph AUROC 0.475, F1 0.0)")
        print("  Do NOT calibrate on density gap: thresholding hides the magnitude that")
        print("  edge weights expose, and it overstated difficulty badly once already.")
        return report

    def topic_similarity_effective(self, poison_mean: float) -> float:
        """Continuous separation between the attack and ordinary same-topic traffic."""
        return float(poison_mean - self.topic_similarity)

    def _save_sequence(self, snapshots: List[GraphSnapshot], seq_dir: Path):
        """Save a sequence of graph snapshots to disk."""
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, snap in enumerate(snapshots):
            np.savez_compressed(
                seq_dir / f"snapshot_{i:04d}.npz",
                node_features=snap.node_features,
                edge_index=snap.edge_index,
                edge_weights=snap.edge_weights,
                node_labels=snap.node_labels,
                graph_label=np.array([snap.graph_label]),
                timestamp=np.array([snap.timestamp]),
            )

    def _create_splits(self, metadata: list, output_path: Path):
        """Create train/val/test splits at the sequence level."""
        ds_cfg = self.config["dataset"]
        indices = np.arange(len(metadata))
        self.rng.shuffle(indices)

        n = len(metadata)
        train_end = int(n * ds_cfg["train_ratio"])
        val_end = train_end + int(n * ds_cfg["val_ratio"])

        splits = {
            "train": [metadata[i]["sequence_id"] for i in indices[:train_end]],
            "val": [metadata[i]["sequence_id"] for i in indices[train_end:val_end]],
            "test": [metadata[i]["sequence_id"] for i in indices[val_end:]],
        }
        with open(output_path / "splits.json", "w") as f:
            json.dump(splits, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="data/raw/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTY_PRESETS), default="hard",
                        help="easy reproduces the original removed generator (metadata leak "
                             "included); medium and hard progressively remove the shortcuts.")
    parser.add_argument("--num-clusters", type=int, default=10)
    parser.add_argument("--global-similarity", type=float, default=0.35,
                        help="Mean cosine between records of DIFFERENT topics (the shared "
                             "domain baseline). Real LLM data measures ~0.41 overall.")
    parser.add_argument("--topic-similarity", type=float, default=None,
                        help="Mean cosine between records of the SAME topic (overrides the "
                             "preset). Above the edge threshold, benign sessions form real "
                             "clusters and become genuine hard negatives.")
    parser.add_argument("--session-length", type=int, default=None,
                        help="Override the preset. 0 = i.i.d. topics per record.")
    parser.add_argument("--retrieval-bias", type=float, default=None,
                        help="Override the preset. >0 leaks the label into retrieval metadata.")
    parser.add_argument("--coherence", type=float, default=None,
                        help="Override every attack type's semantic_coherence.")
    parser.add_argument("--additive-injection", action="store_true",
                        help="Append poisoned records on top of an unchanged benign stream "
                             "instead of replacing benign ones. This is what the original "
                             "generators did, and it leaks the label through record count "
                             "(AUROC 1.0 on node count alone). Kept only for comparison.")
    parser.add_argument("--poison-offset", type=float, default=0.0,
                        help="Displace the attacker's centroid off the topic centre. "
                             "0 = maximum camouflage; larger = easier to detect.")
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="Override config['dataset']['train_ratio']. The 80/10/10 default "
                             "leaves only 30 val and 30 test sequences at 300 sequences total, "
                             "and windows within a sequence overlap heavily, so effective sample "
                             "size is the SEQUENCE count, not the window count. 70/15/15 gives "
                             "far more reliable checkpoint selection and test estimates.")
    parser.add_argument("--val-ratio", type=float, default=None,
                        help="Override config['dataset']['val_ratio']; test takes the remainder.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.num_sequences:
        config["simulation"]["num_sequences"] = args.num_sequences
    if args.train_ratio is not None:
        config["dataset"]["train_ratio"] = args.train_ratio
    if args.val_ratio is not None:
        config["dataset"]["val_ratio"] = args.val_ratio
    ds = config["dataset"]
    ds["test_ratio"] = max(0.0, 1.0 - ds["train_ratio"] - ds["val_ratio"])
    n_seq = config["simulation"]["num_sequences"]
    print(f"Splits: {ds['train_ratio']:.2f}/{ds['val_ratio']:.2f}/{ds['test_ratio']:.2f} "
          f"-> {int(n_seq * ds['train_ratio'])} train, {int(n_seq * ds['val_ratio'])} val, "
          f"{n_seq - int(n_seq * ds['train_ratio']) - int(n_seq * ds['val_ratio'])} test sequences")

    preset = dict(DIFFICULTY_PRESETS[args.difficulty])
    if args.session_length is not None:
        preset["session_length"] = args.session_length
    if args.retrieval_bias is not None:
        preset["retrieval_bias"] = args.retrieval_bias
    if args.coherence is not None:
        preset["coherence"] = args.coherence
    if args.topic_similarity is not None:
        preset["topic_similarity"] = args.topic_similarity

    print(f"Difficulty preset '{args.difficulty}': {preset}")

    simulator = SyntheticMemorySimulator(
        config, seed=args.seed,
        num_clusters=args.num_clusters,
        global_similarity=args.global_similarity,
        topic_similarity=preset["topic_similarity"],
        session_length=preset["session_length"],
        retrieval_bias=preset["retrieval_bias"],
        coherence_override=preset["coherence"],
        poison_offset=args.poison_offset,
        constant_length=not args.additive_injection,
    )
    simulator.generate_dataset(args.output)


if __name__ == "__main__":
    main()
