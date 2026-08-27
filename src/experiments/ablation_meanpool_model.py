"""
Attention-Pooling Ablation Model
===================================
TGNDetectorMeanPool is identical to TGNDetector (src/models/tgn_detector.py) in
every respect - node memory GRU, GAT graph encoder, temporal GRU aggregator,
graph/node heads - except AttentionPool is replaced with plain global_mean_pool.

This isolates the AttentionPool contribution cleanly, unlike the "Static GNN"
baseline in notebooks/05_baselines_comparison.ipynb, which removes attention
pooling, node memory, AND the temporal aggregator simultaneously and so cannot
attribute its performance drop to any one of them.

forward() and get_trajectory_embedding() are duplicated (not just overridden
piecemeal) so the only diff against TGNDetector's versions is the pooling call -
easy to diff/verify against src/models/tgn_detector.py.
"""

from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from src.models.tgn_detector import TGNDetector


class TGNDetectorMeanPool(TGNDetector):
    """Ablation variant: AttentionPool -> global_mean_pool. Everything else unchanged."""

    def __init__(self, config: dict):
        super().__init__(config)
        del self.attn_pool  # unused in this variant; kept out to avoid a dangling unused module

    def forward(self, graph_sequence: List[Batch]) -> Tuple[torch.Tensor, torch.Tensor]:
        T = len(graph_sequence)
        graph_embeddings = []
        last_node_embeddings = None
        last_batch_idx = None

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

            if t == T - 1:
                last_node_embeddings = node_embeddings
                last_batch_idx = batch_idx

        graph_seq_tensor = torch.stack(graph_embeddings, dim=1)
        trajectory_emb = self.temporal_aggregator(graph_seq_tensor)
        graph_scores = self.graph_head(trajectory_emb)
        node_scores = self.node_head(last_node_embeddings, trajectory_emb, last_batch_idx)

        return graph_scores, node_scores

    def get_trajectory_embedding(self, graph_sequence: List[Batch]) -> torch.Tensor:
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
        return self.temporal_aggregator(graph_seq_tensor)
