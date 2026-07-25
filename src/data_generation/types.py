"""
Data types and data structures for memory graph trajectory dataset generation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


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
