"""
Dataset loader for memory graph trajectories.
Loads sequences of graph snapshots and prepares them for TGN training.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import List, Tuple, Optional


# Default feature dimensions: 384 (embedding) + 3 (meta) = 387
_DEFAULT_EMBEDDING_DIM = 384
_DEFAULT_META_FEATURES = 3  # recency, retrieval_freq, retrieval_recency

# Optional per-node structural features: degree (two scalings), and the mean and max
# weight of a node's incident edges.
#
# These exist because GATConv normalises its attention coefficients with a softmax over
# each node's neighbourhood, so its aggregation is a convex combination — a weighted
# average. Averaging is largely degree-invariant, which is the standard expressiveness
# limitation that motivates sum aggregation (GIN): a node with 40 neighbours and one
# with 8 both yield an average, so the encoder cannot readily *count* neighbours.
#
# On a benchmark whose only signal is local density — poisoned records sitting in a
# tighter neighbourhood than a benign session — that blindness is fatal. Measured on
# data/raw/synthetic_hard: logistic regression on explicit degree features reaches
# AUROC 0.89 while the TGN reaches 0.56. Supplying degree and edge-weight statistics
# as node features hands the encoder what its aggregator cannot derive for itself.
_STRUCTURAL_FEATURES = 4


def window_is_stale(graphs: List[Data]) -> bool:
    """A positive window in which no injected record arrives.

    A window is labelled positive once any injected record is present, but node memory is
    reset at every window, so when the injection was already complete at the window's first
    snapshot the window holds no evidence of its arrival: the cluster is there, exactly as
    the matched benign cluster is in a clean sequence. On ap_burst2k_nodup 17% of positive
    windows are like this. --window-target arrival leaves them out of the window loss.
    """
    first, last = graphs[0].node_labels, graphs[-1].node_labels
    return bool(int(last.sum()) > 0 and int(first.sum()) == int(last.sum()))


def with_arrival_edges(data: Data, emb_dim: int, window: int) -> Data:
    """Join every two records that arrived at most `window` interactions apart.

    Similarity edges exist only above cosine 0.6. On ap_burst2k_nodup only 25% of the pairs
    inside an attack burst clear that threshold (mean pairwise cosine 0.43, against 0.20 for
    an ordinary run of records), so the graph shows three quarters of a burst as unconnected
    and attention cannot measure how tight it is. Arrival edges connect records that arrived
    close together whatever their similarity, carrying their cosine as the weight. Node i
    arrived at interaction i, so arrival distance is index distance.

    Edge attributes become [weight, is_arrival_edge] for every edge (model edge_dim 2). The
    structural node features are unchanged: they are computed from the similarity edges.
    """
    n = data.x.size(0)
    sim_attr = torch.cat([data.edge_attr, torch.zeros((data.edge_attr.size(0), 1))], dim=1)
    if n < 2:
        data.edge_attr = sim_attr
        return data
    u = F.normalize(data.x[:, :emb_dim].float(), dim=1)
    src, dst = [], []
    for d in range(1, min(window, n - 1) + 1):
        i = torch.arange(n - d)
        src += [i, i + d]
        dst += [i + d, i]
    src, dst = torch.cat(src), torch.cat(dst)
    cos = (u[src] * u[dst]).sum(dim=1, keepdim=True)
    data.edge_index = torch.cat([data.edge_index, torch.stack([src, dst])], dim=1)
    data.edge_attr = torch.cat([sim_attr, torch.cat([cos, torch.ones_like(cos)], dim=1)])
    return data


COHERENCE_BLOCKS = (5, 10, 15, 20)
COHERENCE_NEIGHBOURS = (5, 10)
COHERENCE_DIM = 2 * len(COHERENCE_BLOCKS) + len(COHERENCE_NEIGHBOURS)


def with_coherence_features(data: Data, emb_dim: int) -> Data:
    """Append per-record arrival-coherence columns: how alike a record is to those that arrived
    around it.

    An attack is k consecutive records drawn from one small neighbourhood, so a run of records
    that are unusually alike is the signal; the ceiling check (src/experiments/ceiling_check.py)
    reaches test AUROC 0.83 from block statistics of exactly this kind, against 0.66 for the
    detector. These columns hand the per-record version to the encoder. Computed from the
    embeddings and arrival order only (node i arrived at interaction i), as the detector sees them.

    For each block length L in COHERENCE_BLOCKS: the highest mean pairwise cosine over the runs of
    L consecutive records that contain the record (the whole snapshot when it holds fewer than L),
    and the same minus its median over the snapshot, since ordinary sessions are alike too.
    For each k in COHERENCE_NEIGHBOURS: the mean cosine to the records that arrived at most k
    interactions before or after it.
    """
    data.x = torch.cat([data.x, coherence_columns(data.x, emb_dim)], dim=1)
    return data


def coherence_columns(x: torch.Tensor, emb_dim: int) -> torch.Tensor:
    """The COHERENCE_DIM columns with_coherence_features appends, for node features x."""
    n = x.size(0)
    cols = torch.zeros((n, COHERENCE_DIM))
    if n >= 2:
        u = F.normalize(x[:, :emb_dim].float(), dim=1)
        S = u @ u.T
        S.fill_diagonal_(0.0)
        P = F.pad(S.cumsum(0).cumsum(1), (1, 0, 1, 0))
        idx = torch.arange(n)
        c = 0
        for L in COHERENCE_BLOCKS:
            L = min(L, n)
            s = torch.arange(n - L + 1)
            block = (P[s + L, s + L] - P[s, s + L] - P[s + L, s] + P[s, s]) / (L * (L - 1))
            inside = (s[None, :] <= idx[:, None]) & (idx[:, None] <= s[None, :] + L - 1)
            best = block[None, :].expand(n, -1).masked_fill(~inside, -2.0).max(dim=1).values
            cols[:, c], cols[:, c + 1] = best, best - best.median()
            c += 2
        dist = (idx[:, None] - idx[None, :]).abs()
        for k in COHERENCE_NEIGHBOURS:
            band = ((dist >= 1) & (dist <= k)).float()
            cols[:, c] = (S * band).sum(1) / band.sum(1).clamp(min=1.0)
            c += 1
    return cols


class MemoryTrajectoryDataset(Dataset):
    """
    Dataset of memory graph trajectories.
    Each item is a sequence of T graph snapshots forming a temporal window.
    """

    def __init__(self, data_dir: str, split: str = "train", temporal_window: int = 10,
                 embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
                 sequence_ids: Optional[List[str]] = None,
                 structural_features: bool = False, arrival_window: int = 0,
                 coherence_features: bool = False):
        self.data_dir = Path(data_dir)
        self.temporal_window = temporal_window
        self.structural_features = structural_features
        self.embedding_dim = embedding_dim
        self.arrival_window = arrival_window
        self.coherence_features = coherence_features
        self.feature_dim = embedding_dim + _DEFAULT_META_FEATURES
        if structural_features:
            self.feature_dim += _STRUCTURAL_FEATURES

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
        if self.arrival_window > 0:
            graphs = [with_arrival_edges(g, self.embedding_dim, self.arrival_window)
                      for g in graphs]
        if self.coherence_features:
            graphs = [with_coherence_features(g, self.embedding_dim) for g in graphs]

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
            "stale": window_is_stale(graphs),
        }

    def _load_snapshot(self, path: Path) -> Data:
        """Load a single graph snapshot as a PyG Data object."""
        snap = np.load(path)

        raw_features = snap["node_features"]
        if self.structural_features:
            raw_features = np.concatenate(
                [raw_features, self._structural_features(
                    raw_features.shape[0], snap["edge_index"], snap["edge_weights"])],
                axis=1,
            )

        node_features = torch.tensor(raw_features, dtype=torch.float32)
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

    @staticmethod
    def _structural_features(num_nodes: int, edge_index: np.ndarray,
                             edge_weights: np.ndarray) -> np.ndarray:
        """Per-node degree and incident-edge-weight statistics.

        Columns: [degree/50, degree/(n-1), mean incident weight, max incident weight].
        The two degree scalings mirror the existing metadata columns' convention (a
        fixed divisor) while also giving a graph-size-invariant version, since snapshots
        grow over time.
        """
        out = np.zeros((num_nodes, _STRUCTURAL_FEATURES), dtype=np.float64)
        if edge_index.size == 0 or num_nodes == 0:
            return out

        src = np.asarray(edge_index[0], dtype=np.int64)
        w = np.asarray(edge_weights, dtype=np.float64).reshape(-1)
        if w.size != src.size:  # defensive: weights missing or mismatched
            w = np.ones_like(src, dtype=np.float64)

        deg = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(deg, src, 1.0)
        wsum = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(wsum, src, w)
        wmax = np.zeros(num_nodes, dtype=np.float64)
        np.maximum.at(wmax, src, w)

        with np.errstate(invalid="ignore", divide="ignore"):
            wmean = np.where(deg > 0, wsum / np.maximum(deg, 1.0), 0.0)

        out[:, 0] = deg / 50.0
        out[:, 1] = deg / max(num_nodes - 1, 1)
        out[:, 2] = wmean
        out[:, 3] = wmax
        return out

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
        "stale": torch.tensor([item.get("stale", False) for item in batch], dtype=torch.bool),
        "sequence_ids": [item["sequence_id"] for item in batch],
    }
