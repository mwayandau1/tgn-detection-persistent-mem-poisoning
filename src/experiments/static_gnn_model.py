"""
The static-GNN baseline: the detector with the temporal machinery removed.

This is the ablation that ap_burst2k was built to make meaningful. That benchmark is
constructed so a single snapshot carries no signal -- final-state features score 0.4650 with
a 95% interval of [0.4146, 0.5190], and node count is exactly 0.5000 -- so a model that sees
only the last snapshot should sit at chance. If it does while the full detector reaches 0.81,
the temporal components are isolated as the source of the signal, using the thesis's own
published baseline rather than a hand-built feature set.

The same comparison on data/raw/llm_generated was uninformative: every model there reaches
1.000 from record content alone (see archive/llm_benchmark/leak_mechanism.py), so nothing was
being ablated.

WHAT IS REMOVED, AND WHAT IS DELIBERATELY NOT
---------------------------------------------
Removed: NodeMemoryModule (the per-node GRU carrying state between snapshots) and
TemporalAggregator (the GRU over the snapshot sequence). The forward pass reads
graph_sequence[-1] and nothing else.

Kept identical: the GAT encoder, the pooling, the head architectures, the input width, the
loss, the optimiser, the augmentation and the checkpoint-selection rule -- the last six
because this trains through the same src.models.train.Trainer, not a second training loop.

The node head normally receives the trajectory embedding as its context vector. Here it
receives the pooled graph embedding of the single snapshot, so the head keeps its shape and
only the provenance of the context changes.

READOUT
-------
Two are provided, and BOTH should be reported.

  "mean"      global_mean_pool, as in notebooks/05_baselines_comparison.ipynb. This is the
              baseline the thesis already publishes, kept for continuity.

  "attention" the detector's own AttentionPool. This is the honest ablation: with mean
              pooling, a loss could be blamed on the readout rather than on the missing
              temporal component, and Sections 2.6 and 3.8 argue at length that mean pooling
              attenuates precisely this kind of small anomalous subgraph. Holding the readout
              fixed leaves exactly one difference between this model and the detector.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from src.models.tgn_detector import (
    AttentionPool,
    GraphAnomalyHead,
    GraphEncoder,
    NodeAnomalyHead,
)


class StaticGNNDetector(nn.Module):
    """Single-snapshot GAT detector. Interface-compatible with TGNDetector."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]

        # Must match what the dataset emits, exactly as TGNDetector does. When the dataset is
        # built with structural_features=True this is 3 + 4; supplying 3 here would hand the
        # baseline a narrower view than the detector gets and quietly rig the comparison.
        input_dim = model_cfg["node_feature_dim"] + model_cfg.get("node_meta_dim", 3)
        gat_hidden = model_cfg["gat_hidden_dim"]
        gat_heads = model_cfg["gat_heads"]
        mlp_hidden = model_cfg["mlp_hidden_dim"]
        dropout = model_cfg["dropout"]

        # No NodeMemoryModule: the GAT sees the raw node features, not features concatenated
        # with a memory vector carried in from earlier snapshots.
        self.graph_encoder = GraphEncoder(
            input_dim=input_dim,
            hidden_dim=gat_hidden,
            num_heads=gat_heads,
            dropout=dropout,
            edge_dim=1,
        )

        readout = model_cfg.get("static_readout", "attention")
        if readout not in ("mean", "attention"):
            raise ValueError(f"unknown static_readout {readout!r}; expected 'mean' or 'attention'")
        self.readout = readout
        self.attn_pool = AttentionPool(input_dim=gat_hidden, hidden_dim=mlp_hidden) \
            if readout == "attention" else None

        # No TemporalAggregator, so the graph head reads the pooled snapshot embedding
        # directly and the node head takes that same vector as its context.
        self.graph_head = GraphAnomalyHead(gat_hidden, mlp_hidden)
        self.node_head = NodeAnomalyHead(gat_hidden, gat_hidden, mlp_hidden)
        self.gat_hidden = gat_hidden

    def reset_memory(self):
        """No-op. There is no memory to reset; present so Trainer runs unmodified."""

    def _pool(self, node_embeddings: torch.Tensor, batch_idx: torch.Tensor,
              num_graphs: Optional[int]) -> torch.Tensor:
        if self.readout == "attention":
            return self.attn_pool(node_embeddings, batch_idx, num_graphs=num_graphs)
        return global_mean_pool(node_embeddings, batch_idx, size=num_graphs)

    def forward(self, graph_sequence: List[Batch]) -> Tuple[torch.Tensor, torch.Tensor]:
        # The whole point of the baseline: the first T-1 snapshots are discarded unread.
        g = graph_sequence[-1]
        edge_attr = g.edge_attr if hasattr(g, "edge_attr") and g.edge_attr is not None else None

        node_embeddings = self.graph_encoder(g.x, g.edge_index, edge_attr=edge_attr)
        num_graphs = g.num_graphs if hasattr(g, "num_graphs") else None
        graph_emb = self._pool(node_embeddings, g.batch, num_graphs)

        graph_scores = self.graph_head(graph_emb)
        node_scores = self.node_head(node_embeddings, graph_emb, g.batch)
        return graph_scores, node_scores


class StaticGNNDetectorMeanPool(StaticGNNDetector):
    """The published baseline's readout, as a class so Trainer's model_cls can select it."""

    def __init__(self, config: dict):
        config["model"]["static_readout"] = "mean"
        super().__init__(config)


class StaticGNNDetectorAttention(StaticGNNDetector):
    """The clean ablation: identical to the detector but for the temporal components."""

    def __init__(self, config: dict):
        config["model"]["static_readout"] = "attention"
        super().__init__(config)
