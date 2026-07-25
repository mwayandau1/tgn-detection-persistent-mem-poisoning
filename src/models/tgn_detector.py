"""
Temporal Graph Neural Network Detector for Memory Poisoning
============================================================
Architecture:
    Component 1: Node Feature Encoder (frozen sentence transformer — handled at data gen)
    Component 2: Node Memory State (GRU per node, persistent across timesteps)
    Component 3: Graph Attention Network (2-layer GAT with edge attributes)
    Component 4: Temporal Aggregator (graph-level GRU over T snapshots)
    Component 5: Anomaly Scoring Heads (graph-level + node-level)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch
from typing import List, Tuple, Optional, Dict


class NodeMemoryModule(nn.Module):
    """GRU-based memory for each node, updated at each timestep.
    
    Tracks persistent hidden state per node_id across timesteps via
    an external memory bank dictionary.
    """

    def __init__(self, input_dim: int, memory_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.gru_cell = nn.GRUCell(input_dim, memory_dim)

    def forward(self, node_features: torch.Tensor, prev_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: (num_nodes, input_dim)
            prev_memory: (num_nodes, memory_dim) or None for first timestep
        Returns:
            updated_memory: (num_nodes, memory_dim)
        """
        if prev_memory is None:
            prev_memory = torch.zeros(node_features.size(0), self.memory_dim, device=node_features.device)
        return self.gru_cell(node_features, prev_memory)


class GraphEncoder(nn.Module):
    """2-layer GAT that produces node embeddings from the memory graph.
    
    Supports edge attributes (cosine similarity weights) via edge_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4,
                 dropout: float = 0.3, edge_dim: int = 1):
        super().__init__()
        self.gat1 = GATConv(
            input_dim, hidden_dim, heads=num_heads, dropout=dropout,
            concat=True, edge_dim=edge_dim,
        )
        self.gat2 = GATConv(
            hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout,
            concat=False, edge_dim=edge_dim,
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, input_dim)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, 1) — edge weights (cosine similarity)
        Returns:
            node_embeddings: (num_nodes, hidden_dim)
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x


class TemporalAggregator(nn.Module):
    """GRU that reads sequence of graph-level embeddings over T timesteps."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, graph_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph_sequence: (batch_size, T, input_dim) — sequence of graph embeddings
        Returns:
            trajectory_embedding: (batch_size, hidden_dim)
        """
        output, h_n = self.gru(graph_sequence)
        return h_n.squeeze(0)  # (batch_size, hidden_dim)


class GraphAnomalyHead(nn.Module):
    """MLP that scores graph-level anomaly probability."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns anomaly score in [0, 1]."""
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class NodeAnomalyHead(nn.Module):
    """MLP that scores per-node anomaly probability."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Returns per-node anomaly scores in [0, 1]."""
        return torch.sigmoid(self.mlp(node_embeddings)).squeeze(-1)


class TGNDetector(nn.Module):
    """
    Full Temporal GNN Detector for Memory Poisoning.

    Processes a sequence of T memory graph snapshots and outputs:
    - Graph-level anomaly score: P(poisoned | trajectory)
    - Node-level anomaly scores: P(poisoned | node) for each record

    Node memory is persistent across timesteps: a GRU hidden state is
    maintained per node_id and carried forward as new snapshots arrive.
    Nodes appearing for the first time get zero-initialized memory.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]

        # 3 meta features: recency, retrieval_freq, retrieval_recency
        input_dim = model_cfg["node_feature_dim"] + 3
        memory_dim = model_cfg["node_memory_dim"]
        gat_hidden = model_cfg["gat_hidden_dim"]
        gat_heads = model_cfg["gat_heads"]
        gru_hidden = model_cfg["gru_hidden_dim"]
        mlp_hidden = model_cfg["mlp_hidden_dim"]
        dropout = model_cfg["dropout"]

        # Component 2: Node memory
        self.node_memory = NodeMemoryModule(input_dim, memory_dim)

        # Component 3: Graph encoder (GAT with edge attributes)
        # Input to GAT is concatenation of node features + node memory
        self.graph_encoder = GraphEncoder(
            input_dim=input_dim + memory_dim,
            hidden_dim=gat_hidden,
            num_heads=gat_heads,
            dropout=dropout,
            edge_dim=1,  # Cosine similarity edge weights
        )

        # Component 4: Temporal aggregator
        self.temporal_aggregator = TemporalAggregator(
            input_dim=gat_hidden,
            hidden_dim=gru_hidden,
        )

        # Component 5: Scoring heads
        self.graph_head = GraphAnomalyHead(gru_hidden, mlp_hidden)
        self.node_head = NodeAnomalyHead(gat_hidden, mlp_hidden)

        # Store config
        self.gat_hidden = gat_hidden
        self.memory_dim = memory_dim

        # Per-node memory bank: node_id -> hidden state tensor
        # This is populated during forward and reset between sequences
        self._memory_bank: Dict[int, torch.Tensor] = {}

    def reset_memory(self):
        """Clear the node memory bank. Call between independent sequences."""
        self._memory_bank.clear()

    def _get_node_memories(self, node_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Retrieve previous memory states for a batch of node IDs.
        
        Nodes seen before get their stored hidden state; new nodes get zeros.
        """
        num_nodes = node_ids.size(0)
        prev_memory = torch.zeros(num_nodes, self.memory_dim, device=device)
        for i, nid in enumerate(node_ids.tolist()):
            if nid in self._memory_bank:
                prev_memory[i] = self._memory_bank[nid]
        return prev_memory

    def _store_node_memories(self, node_ids: torch.Tensor, memories: torch.Tensor):
        """Store updated memory states back into the bank."""
        for i, nid in enumerate(node_ids.tolist()):
            self._memory_bank[nid] = memories[i].detach()

    def forward(self, graph_sequence: List[Batch]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph_sequence: List of T batched PyG graphs (one per timestep).
                            Each graph should have a `node_ids` attribute
                            mapping nodes to their persistent record IDs.

        Returns:
            graph_scores: (batch_size,) — graph-level anomaly scores
            node_scores: (num_nodes_last,) — node-level scores for last timestep
        """
        T = len(graph_sequence)
        batch_size = graph_sequence[0].num_graphs

        graph_embeddings = []  # Will be (batch_size, T, gat_hidden)

        last_node_embeddings = None

        # Reset memory at the start of each forward pass (each trajectory is independent)
        self.reset_memory()

        for t in range(T):
            g = graph_sequence[t]
            x = g.x  # (total_nodes_in_batch, input_dim)
            edge_index = g.edge_index
            batch_idx = g.batch  # Maps each node to its graph in the batch

            # Get edge attributes if available
            edge_attr = g.edge_attr if hasattr(g, 'edge_attr') and g.edge_attr is not None else None

            # Retrieve previous node memory (persistent across timesteps)
            if hasattr(g, 'node_ids') and g.node_ids is not None:
                prev_memory = self._get_node_memories(g.node_ids, x.device)
            else:
                # Fallback: use positional IDs within the batch
                prev_memory = None

            # Update node memory via GRU
            node_mem = self.node_memory(x, prev_memory)  # (num_nodes, memory_dim)

            # Store updated memories for next timestep
            if hasattr(g, 'node_ids') and g.node_ids is not None:
                self._store_node_memories(g.node_ids, node_mem)

            # Concatenate features + memory for GAT input
            gat_input = torch.cat([x, node_mem], dim=-1)

            # Run GAT with edge attributes
            node_embeddings = self.graph_encoder(gat_input, edge_index, edge_attr=edge_attr)

            # Pool to graph-level
            graph_emb = global_mean_pool(node_embeddings, batch_idx)  # (batch_size, gat_hidden)
            graph_embeddings.append(graph_emb)

            # Keep last timestep's node embeddings for node-level scoring
            if t == T - 1:
                last_node_embeddings = node_embeddings

        # Stack temporal sequence: (batch_size, T, gat_hidden)
        graph_seq_tensor = torch.stack(graph_embeddings, dim=1)

        # Temporal aggregation
        trajectory_emb = self.temporal_aggregator(graph_seq_tensor)  # (batch_size, gru_hidden)

        # Graph-level scoring
        graph_scores = self.graph_head(trajectory_emb)  # (batch_size,)

        # Node-level scoring (on last timestep)
        node_scores = self.node_head(last_node_embeddings)  # (num_nodes_last,)

        return graph_scores, node_scores

    def get_trajectory_embedding(self, graph_sequence: List[Batch]) -> torch.Tensor:
        """Get trajectory embedding for contrastive learning."""
        T = len(graph_sequence)
        graph_embeddings = []

        self.reset_memory()

        for t in range(T):
            g = graph_sequence[t]
            x = g.x
            edge_index = g.edge_index
            batch_idx = g.batch
            edge_attr = g.edge_attr if hasattr(g, 'edge_attr') and g.edge_attr is not None else None

            if hasattr(g, 'node_ids') and g.node_ids is not None:
                prev_memory = self._get_node_memories(g.node_ids, x.device)
            else:
                prev_memory = None

            node_mem = self.node_memory(x, prev_memory)

            if hasattr(g, 'node_ids') and g.node_ids is not None:
                self._store_node_memories(g.node_ids, node_mem)

            gat_input = torch.cat([x, node_mem], dim=-1)
            node_embeddings = self.graph_encoder(gat_input, edge_index, edge_attr=edge_attr)
            graph_emb = global_mean_pool(node_embeddings, batch_idx)
            graph_embeddings.append(graph_emb)

        graph_seq_tensor = torch.stack(graph_embeddings, dim=1)
        trajectory_emb = self.temporal_aggregator(graph_seq_tensor)
        return trajectory_emb
