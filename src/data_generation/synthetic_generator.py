"""
Synthetic Memory Graph Dataset Generator (Stage A)
===================================================
Generates labeled memory graph trajectories WITHOUT requiring an LLM.
Uses synthetic embeddings to simulate memory records and poisoning attacks.

This lets you validate the TGN architecture before investing in LLM-based generation.

Usage:
    python -m src.data_generation.synthetic_generator --config configs/default.yaml
"""

import os
import json
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from tqdm import tqdm
import yaml


@dataclass
class MemoryRecord:
    """A single memory record (node in the graph)."""
    record_id: int
    embedding: np.ndarray
    timestamp: int              # When it was written
    retrieval_count: int = 0
    last_retrieved: int = 0
    provenance: str = "user"    # user | attacker
    is_poisoned: bool = False
    semantic_cluster: int = 0   # Which topic cluster this belongs to


@dataclass
class GraphSnapshot:
    """A snapshot of the memory graph at time t."""
    timestamp: int
    node_ids: List[int]
    node_features: np.ndarray       # (num_nodes, feature_dim)
    edge_index: np.ndarray          # (2, num_edges)
    edge_weights: np.ndarray        # (num_edges,)
    node_labels: np.ndarray         # (num_nodes,) 0=benign, 1=poisoned
    graph_label: int                # 0=clean, 1=poisoned at this timestep


class SyntheticMemorySimulator:
    """Simulates an LLM agent's long-term memory with poisoning attacks."""

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.sim_cfg = config["simulation"]
        self.atk_cfg = config["attacks"]
        self.embedding_dim = self.sim_cfg["embedding_dim"]

        # Create semantic cluster centers (topics in the memory)
        self.num_clusters = 10
        self.cluster_centers = self._generate_cluster_centers()

    def _generate_cluster_centers(self) -> np.ndarray:
        """Generate random cluster centers representing different topics."""
        centers = self.rng.standard_normal((self.num_clusters, self.embedding_dim))
        # Normalize to unit sphere
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        return centers

    def _generate_record_embedding(self, cluster_id: int, noise_scale: float = 0.3) -> np.ndarray:
        """Generate a record embedding near a cluster center."""
        center = self.cluster_centers[cluster_id]
        noise = self.rng.standard_normal(self.embedding_dim) * noise_scale
        embedding = center + noise
        return embedding / np.linalg.norm(embedding)

    def _generate_poisoned_embedding(self, target_cluster: int, coherence: float) -> np.ndarray:
        """Generate a poisoned record embedding targeting a specific cluster."""
        # Higher coherence = closer to cluster center = harder to distinguish
        noise_scale = 1.0 - coherence  # Low noise = high coherence
        return self._generate_record_embedding(target_cluster, noise_scale=noise_scale * 0.5)

    def _compute_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _build_graph_snapshot(self, records: List[MemoryRecord], timestamp: int) -> GraphSnapshot:
        """Build a graph snapshot from current memory records."""
        n = len(records)
        if n == 0:
            return GraphSnapshot(
                timestamp=timestamp,
                node_ids=[],
                node_features=np.zeros((0, self.embedding_dim + 3)),
                edge_index=np.zeros((2, 0), dtype=int),
                edge_weights=np.zeros(0),
                node_labels=np.zeros(0, dtype=int),
                graph_label=0,
            )

        # Build node features: [embedding, recency, retrieval_freq, retrieval_recency]
        # NOTE: provenance_flag was removed — it leaked the label directly.
        node_features = np.zeros((n, self.embedding_dim + 3))
        node_labels = np.zeros(n, dtype=int)
        node_ids = []

        for i, rec in enumerate(records):
            node_features[i, :self.embedding_dim] = rec.embedding
            node_features[i, self.embedding_dim] = (timestamp - rec.timestamp) / max(timestamp, 1)  # recency
            node_features[i, self.embedding_dim + 1] = rec.retrieval_count / 50.0  # normalized freq
            node_features[i, self.embedding_dim + 2] = (timestamp - rec.last_retrieved) / max(timestamp, 1)
            node_labels[i] = 1 if rec.is_poisoned else 0
            node_ids.append(rec.record_id)

        # Build edges based on semantic similarity
        threshold = self.sim_cfg["semantic_edge_threshold"]
        edges_src, edges_dst, weights = [], [], []

        # Compute pairwise cosine similarities (for small graphs this is fine)
        embeddings = node_features[:, :self.embedding_dim]
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > threshold:
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])
                    weights.extend([sim_matrix[i, j], sim_matrix[i, j]])

        edge_index = np.array([edges_src, edges_dst], dtype=int) if edges_src else np.zeros((2, 0), dtype=int)
        edge_weights = np.array(weights) if weights else np.zeros(0)

        graph_label = 1 if any(r.is_poisoned for r in records) else 0

        return GraphSnapshot(
            timestamp=timestamp,
            node_ids=node_ids,
            node_features=node_features,
            edge_index=edge_index,
            edge_weights=edge_weights,
            node_labels=node_labels,
            graph_label=graph_label,
        )

    def generate_clean_sequence(self, num_interactions: int) -> List[GraphSnapshot]:
        """Generate a clean (no attack) interaction sequence."""
        records: List[MemoryRecord] = []
        snapshots: List[GraphSnapshot] = []
        record_counter = 0
        snapshot_interval = self.sim_cfg["snapshot_interval"]

        for t in range(num_interactions):
            # Add a new benign record
            cluster = self.rng.integers(0, self.num_clusters)
            rec = MemoryRecord(
                record_id=record_counter,
                embedding=self._generate_record_embedding(cluster),
                timestamp=t,
                provenance="user",
                is_poisoned=False,
                semantic_cluster=cluster,
            )
            records.append(rec)
            record_counter += 1

            # Simulate retrieval (some records get retrieved)
            if len(records) > 5 and self.rng.random() < 0.3:
                retrieved_idx = self.rng.integers(0, len(records))
                records[retrieved_idx].retrieval_count += 1
                records[retrieved_idx].last_retrieved = t

            # Take snapshot at intervals
            if (t + 1) % snapshot_interval == 0:
                snapshots.append(self._build_graph_snapshot(records, t))

        return snapshots

    def generate_poisoned_sequence(self, num_interactions: int, attack_type: dict) -> List[GraphSnapshot]:
        """Generate a sequence with a poisoning attack injected."""
        records: List[MemoryRecord] = []
        snapshots: List[GraphSnapshot] = []
        record_counter = 0
        snapshot_interval = self.sim_cfg["snapshot_interval"]

        # Choose target cluster for the attack
        target_cluster = self.rng.integers(0, self.num_clusters)

        # Determine injection schedule
        num_injected = attack_type["num_injected_records"]
        injection_spread = attack_type["injection_spread"]
        coherence = attack_type["semantic_coherence"]

        # Start injection after some baseline interactions
        injection_start = self.rng.integers(50, max(51, num_interactions - injection_spread - 20))
        if injection_spread <= 1:
            injection_times = [injection_start] * num_injected
        else:
            injection_times = sorted(
                self.rng.integers(injection_start, injection_start + injection_spread, size=num_injected)
            )

        injection_idx = 0

        for t in range(num_interactions):
            # Add benign record
            cluster = self.rng.integers(0, self.num_clusters)
            rec = MemoryRecord(
                record_id=record_counter,
                embedding=self._generate_record_embedding(cluster),
                timestamp=t,
                provenance="user",
                is_poisoned=False,
                semantic_cluster=cluster,
            )
            records.append(rec)
            record_counter += 1

            # Inject poisoned records at scheduled times
            while injection_idx < len(injection_times) and injection_times[injection_idx] <= t:
                poisoned_rec = MemoryRecord(
                    record_id=record_counter,
                    embedding=self._generate_poisoned_embedding(target_cluster, coherence),
                    timestamp=t,
                    provenance="attacker",
                    is_poisoned=True,
                    semantic_cluster=target_cluster,
                )
                records.append(poisoned_rec)
                record_counter += 1
                injection_idx += 1

            # Simulate retrieval — poisoned records get retrieved more over time
            if len(records) > 5 and self.rng.random() < 0.3:
                # Bias retrieval toward poisoned records (simulating semantic trigger)
                poisoned_indices = [i for i, r in enumerate(records) if r.is_poisoned]
                if poisoned_indices and self.rng.random() < 0.4:
                    retrieved_idx = self.rng.choice(poisoned_indices)
                else:
                    retrieved_idx = self.rng.integers(0, len(records))
                records[retrieved_idx].retrieval_count += 1
                records[retrieved_idx].last_retrieved = t

            # Take snapshot
            if (t + 1) % snapshot_interval == 0:
                snapshots.append(self._build_graph_snapshot(records, t))

        return snapshots

    def generate_dataset(self, output_dir: str):
        """Generate the full dataset of labeled trajectories."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_sequences = self.sim_cfg["num_sequences"]
        num_interactions = self.sim_cfg["interactions_per_sequence"]
        attack_ratio = self.atk_cfg["attack_ratio"]
        attack_types = self.atk_cfg["types"]

        num_poisoned = int(num_sequences * attack_ratio)
        num_clean = num_sequences - num_poisoned

        all_sequences = []
        metadata = []

        print(f"Generating {num_clean} clean sequences...")
        for i in tqdm(range(num_clean)):
            snapshots = self.generate_clean_sequence(num_interactions)
            seq_id = f"clean_{i:04d}"
            self._save_sequence(snapshots, output_path / seq_id)
            metadata.append({
                "sequence_id": seq_id,
                "label": "clean",
                "attack_type": None,
                "num_snapshots": len(snapshots),
            })

        print(f"Generating {num_poisoned} poisoned sequences...")
        for i in tqdm(range(num_poisoned)):
            # Cycle through attack types
            attack_type = attack_types[i % len(attack_types)]
            snapshots = self.generate_poisoned_sequence(num_interactions, attack_type)
            seq_id = f"poisoned_{attack_type['name']}_{i:04d}"
            self._save_sequence(snapshots, output_path / seq_id)
            metadata.append({
                "sequence_id": seq_id,
                "label": "poisoned",
                "attack_type": attack_type["name"],
                "num_snapshots": len(snapshots),
            })

        # Save metadata
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Create train/val/test split
        self._create_splits(metadata, output_path)

        print(f"\nDataset generated: {len(metadata)} sequences saved to {output_path}")
        print(f"  Clean: {num_clean}, Poisoned: {num_poisoned}")

    def _save_sequence(self, snapshots: List[GraphSnapshot], seq_dir: Path):
        """Save a sequence of graph snapshots to disk."""
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, snap in enumerate(snapshots):
            snap_data = {
                "timestamp": snap.timestamp,
                "node_ids": snap.node_ids,
                "node_features": snap.node_features.tolist(),
                "edge_index": snap.edge_index.tolist(),
                "edge_weights": snap.edge_weights.tolist(),
                "node_labels": snap.node_labels.tolist(),
                "graph_label": snap.graph_label,
            }
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
        """Create train/val/test splits."""
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
    parser = argparse.ArgumentParser(description="Generate synthetic memory poisoning dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="data/raw/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    simulator = SyntheticMemorySimulator(config, seed=args.seed)
    simulator.generate_dataset(args.output)


if __name__ == "__main__":
    main()
