"""
Dataset loader for memory graph trajectories.
Loads sequences of graph snapshots and prepares them for TGN training.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import List, Tuple, Optional


# Default feature dimensions: 384 (embedding) + 3 (meta) = 387
_DEFAULT_EMBEDDING_DIM = 384
_DEFAULT_META_FEATURES = 3  # recency, retrieval_freq, retrieval_recency


class MemoryTrajectoryDataset(Dataset):
    """
    Dataset of memory graph trajectories.
    Each item is a sequence of T graph snapshots forming a temporal window.
    """

    def __init__(self, data_dir: str, split: str = "train", temporal_window: int = 10,
                 embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
                 sequence_ids: Optional[List[str]] = None):
        self.data_dir = Path(data_dir)
        self.temporal_window = temporal_window
        self.feature_dim = embedding_dim + _DEFAULT_META_FEATURES

        # Load metadata
        with open(self.data_dir / "metadata.json") as f:
            metadata = json.load(f)
        self.metadata = {m["sequence_id"]: m for m in metadata}

        if sequence_ids is not None:
            # Explicit override (e.g. attack-type holdout filtering) bypasses splits.json.
            self.sequence_ids = sequence_ids
        else:
            with open(self.data_dir / "splits.json") as f:
                splits = json.load(f)
            self.sequence_ids = splits[split]

        # Build index of all valid temporal windows
        self.windows = self._build_window_index()

    def _build_window_index(self) -> List[Tuple[str, int]]:
        """Build index of (sequence_id, start_snapshot_idx) for all valid windows."""
        windows = []
        for seq_id in self.sequence_ids:
            seq_dir = self.data_dir / seq_id
            if not seq_dir.exists():
                continue
            num_snapshots = len(list(seq_dir.glob("snapshot_*.npz")))
            # Each window starts at idx and covers [idx, idx+T)
            for start in range(max(1, num_snapshots - self.temporal_window + 1)):
                windows.append((seq_id, start))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        seq_id, start = self.windows[idx]
        seq_dir = self.data_dir / seq_id

        graphs = []
        for i in range(start, start + self.temporal_window):
            snap_path = seq_dir / f"snapshot_{i:04d}.npz"
            if snap_path.exists():
                graphs.append(self._load_snapshot(snap_path))
            else:
                # Pad with empty graph if not enough snapshots
                graphs.append(self._empty_graph())

        # The label is from the last snapshot in the window
        last_graph = graphs[-1]
        graph_label = last_graph.y.item()
        node_labels = last_graph.node_labels

        return {
            "graphs": graphs,              # List of T PyG Data objects
            "graph_label": graph_label,    # 0 or 1
            "node_labels": node_labels,    # Per-node labels at final timestep
            "sequence_id": seq_id,
            "window_start": start,
        }

    def _load_snapshot(self, path: Path) -> Data:
        """Load a single graph snapshot as a PyG Data object."""
        snap = np.load(path)

        node_features = torch.tensor(snap["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(snap["edge_index"], dtype=torch.long)
        edge_weights = torch.tensor(snap["edge_weights"], dtype=torch.float32)
        node_labels = torch.tensor(snap["node_labels"], dtype=torch.long)
        graph_label = torch.tensor(snap["graph_label"], dtype=torch.long).squeeze()

        # Always provide edge_attr (empty tensor if no edges) for consistent batching
        if edge_weights.numel() > 0:
            edge_attr = edge_weights.unsqueeze(-1)
        else:
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)

        # Generate sequential node IDs for persistent memory tracking.
        # These IDs are unique within a sequence (record_id from the generator).
        # Since records are appended over time, node i at snapshot t keeps the
        # same position (and thus same ID) across snapshots.
        num_nodes = node_features.size(0)
        node_ids = torch.arange(num_nodes, dtype=torch.long)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=graph_label,
            node_labels=node_labels,
            node_ids=node_ids,
        )
        return data

    def _empty_graph(self) -> Data:
        """Return an empty graph as padding."""
        return Data(
            x=torch.zeros((1, self.feature_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1), dtype=torch.float32),
            y=torch.tensor(0, dtype=torch.long),
            node_labels=torch.zeros(1, dtype=torch.long),
            node_ids=torch.zeros(1, dtype=torch.long),
        )


def collate_trajectories(batch: List[dict]) -> dict:
    """Custom collate function for batching trajectories."""
    T = len(batch[0]["graphs"])

    # Batch graphs at each timestep
    batched_graphs = []
    for t in range(T):
        graphs_at_t = [item["graphs"][t] for item in batch]
        batched_graphs.append(Batch.from_data_list(graphs_at_t))

    graph_labels = torch.tensor([item["graph_label"] for item in batch], dtype=torch.float32)

    return {
        "graphs": batched_graphs,
        "graph_labels": graph_labels,
        "sequence_ids": [item["sequence_id"] for item in batch],
    }
