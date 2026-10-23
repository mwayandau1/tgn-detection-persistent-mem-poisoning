"""
Ablations that split the temporal contribution between its two components, and a static
baseline matched to the detector's capacity.
==========================================================================================

The static GAT of Table 4.5 removes the per-node memory AND the temporal aggregator, so the
0.2517 it costs cannot be assigned between them (Section 6.5 records this), and it carries
306,243 parameters against the detector's 670,915, so "it is simply smaller" stays open
(Section 4.7 concedes this). Three models close those gaps:

  TGNNoAggregator  ("memory only")  node memory runs over all T snapshots exactly as in the
                   detector; the GAT and attention pool then read the FINAL snapshot only, and
                   the heads read that pooled vector. Removes the GRU over snapshot embeddings.
                   The memory GRU's input is the raw node features, not GAT output, so running
                   the GAT on the last snapshot alone computes what the detector would compute
                   for that snapshot.
  TGNNoMemory      ("aggregator only") the GAT reads raw node features at every snapshot (no
                   memory concatenated, so its input is 391 wide, not 519), attention pooling
                   per snapshot, and the temporal GRU over the T pooled vectors. Removes the
                   per-node memory.
  StaticGNNWide    the static attention baseline with its hidden width raised until its
                   parameter count is the closest achievable to the detector's, so a gap
                   between them cannot be blamed on capacity.

All three are interface-compatible with TGNDetector and train through the same Trainer.
"""

from typing import List, Tuple

import copy
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.experiments.static_gnn_model import StaticGNNDetectorAttention
from src.models.tgn_detector import (AttentionPool, GraphAnomalyHead, GraphEncoder,
                                     NodeAnomalyHead, TGNDetector)

DETECTOR_PARAMETERS = 670_915


class TGNNoAggregator(TGNDetector):
    """Per-node memory kept, temporal aggregator removed."""

    def __init__(self, config: dict):
        super().__init__(config)
        del self.temporal_aggregator
        # Heads now read the pooled final snapshot (gat_hidden wide) instead of the GRU state.
        self.graph_head = GraphAnomalyHead(self.gat_hidden, config["model"]["mlp_hidden_dim"])
        self.node_head = NodeAnomalyHead(self.gat_hidden, self.gat_hidden,
                                         config["model"]["mlp_hidden_dim"])

    def forward(self, graph_sequence: List[Batch]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset_memory()
        stride = self._memory_stride(graph_sequence)
        node_mem = None
        for g in graph_sequence:
            slots = self._slot_ids(g, stride)
            prev = self._get_node_memories(slots, g.x.device) if slots is not None else None
            node_mem = self.node_memory(g.x, prev)
            if slots is not None:
                self._store_node_memories(slots, node_mem)
        g = graph_sequence[-1]
        edge_attr = g.edge_attr if getattr(g, "edge_attr", None) is not None else None
        node_emb = self.graph_encoder(torch.cat([g.x, node_mem], dim=-1), g.edge_index,
                                      edge_attr=edge_attr)
        graph_emb = self.attn_pool(node_emb, g.batch, num_graphs=getattr(g, "num_graphs", None))
        return self.graph_head(graph_emb), self.node_head(node_emb, graph_emb, g.batch)

    def get_trajectory_embedding(self, graph_sequence: List[Batch]) -> torch.Tensor:
        raise NotImplementedError("phase one is not used for the ablations")


class TGNNoMemory(TGNDetector):
    """Temporal aggregator kept, per-node memory removed."""

    def __init__(self, config: dict):
        super().__init__(config)
        m = config["model"]
        input_dim = m["node_feature_dim"] + m.get("node_meta_dim", 3)
        del self.node_memory
        self.graph_encoder = GraphEncoder(input_dim=input_dim, hidden_dim=m["gat_hidden_dim"],
                                          num_heads=m["gat_heads"], dropout=m["dropout"],
                                          edge_dim=1)

    def forward(self, graph_sequence: List[Batch]) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled, node_emb = [], None
        for g in graph_sequence:
            edge_attr = g.edge_attr if getattr(g, "edge_attr", None) is not None else None
            node_emb = self.graph_encoder(g.x, g.edge_index, edge_attr=edge_attr)
            pooled.append(self.attn_pool(node_emb, g.batch,
                                         num_graphs=getattr(g, "num_graphs", None)))
        traj = self.temporal_aggregator(torch.stack(pooled, dim=1))
        return self.graph_head(traj), self.node_head(node_emb, traj, graph_sequence[-1].batch)

    def get_trajectory_embedding(self, graph_sequence: List[Batch]) -> torch.Tensor:
        raise NotImplementedError("phase one is not used for the ablations")


def matched_static_hidden(config: dict, target: int = DETECTOR_PARAMETERS) -> int:
    """GAT hidden width at which the static attention baseline's size is closest to target."""
    best = None
    for h in range(config["model"]["gat_hidden_dim"], 512):
        c = copy.deepcopy(config)
        c["model"]["gat_hidden_dim"] = h
        n = sum(p.numel() for p in StaticGNNDetectorAttention(c).parameters())
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (h, n)
        if n > target:
            break
    return best[0]


class StaticGNNWide(StaticGNNDetectorAttention):
    """Static attention baseline widened to the detector's parameter count."""

    def __init__(self, config: dict):
        config["model"]["gat_hidden_dim"] = config["model"].get(
            "static_wide_hidden") or matched_static_hidden(config)
        config["model"]["static_wide_hidden"] = config["model"]["gat_hidden_dim"]
        super().__init__(config)


MODEL_KINDS = {"no_aggregator": TGNNoAggregator, "no_memory": TGNNoMemory,
               "static_wide": StaticGNNWide}
