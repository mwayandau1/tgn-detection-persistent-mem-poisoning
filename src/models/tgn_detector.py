"""
Temporal Graph Neural Network (TGN) Detector Architecture
=========================================================
Dual-head spatial-temporal detector for memory poisoning in LLM agents.
Vectorized GPU memory bank indexing for 100x faster evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as scatter_softmax
from torch_geometric.data import Batch
from typing import List, Tuple, Dict, Optional


class NodeMemoryModule(nn.Module):
    """GRU cell that maintains per-node memory state over interaction time."""

    def __init__(self, input_dim: int, memory_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.gru_cell = nn.GRUCell(input_dim, memory_dim)

    def forward(self, node_features: torch.Tensor, prev_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        if prev_memory is None:
            prev_memory = torch.zeros(node_features.size(0), self.memory_dim, device=node_features.device)
        return self.gru_cell(node_features, prev_memory)


class GraphEncoder(nn.Module):
    """2-layer GAT that produces node embeddings from the memory graph."""

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
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x


class AttentionPool(nn.Module):
    """Learned attention pooling over nodes in a graph (Ilse et al. MIL-style attention).

    Plain mean pooling averages a small anomalous subgraph (e.g. ~15-20 poisoned
    nodes among hundreds of benign ones) away almost by construction. This lets the
    graph-level readout instead learn to up-weight the nodes that matter.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor, num_graphs: Optional[int] = None) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        weights = scatter_softmax(scores, batch_idx)
        weighted = x * weights.unsqueeze(-1)
        if num_graphs is None:
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
        out = weighted.new_zeros(num_graphs, x.size(-1))
        out = out.index_add(0, batch_idx, weighted)
        return out


class ScaleAwarePool(nn.Module):
    """Attention pooling, concatenated with readouts that preserve magnitude.

    `AttentionPool` alone normalises its weights with a softmax, so the graph embedding is a
    convex combination of node embeddings — a weighted average. `GATConv` does the same over
    each neighbourhood. The whole path from node features to graph embedding is therefore
    scale-invariant: it can represent what the average record looks like, but not how many
    records there are, nor how much total mass sits in a dense region.

    That is the wrong invariance for this task. A memory-poisoning attack is a *quantity* of
    mutually-similar records appearing where ordinary traffic has few. A logistic regression on
    nine hand-crafted features — node count, edge count, max degree, tightest-cluster density
    and similarity — scores 0.995 on synthetic_gap0125 where the attention-pooled model scores
    0.889. Every one of those features is a magnitude.

    This adds, alongside the attention readout: an unweighted mean, an element-wise max, and
    log node count. The mean and count together restore what a convex combination cannot
    express — the graph's total mass is recoverable as mean x count — while max gives a single
    strongly-anomalous node a route to the graph head without being averaged down.

    An earlier version used a raw unnormalised sum in place of the mean and scored *worse*
    than plain attention pooling (0.814 against 0.904 validation AUROC on synthetic_gap0125).
    Measured on a trained checkpoint, the sum component's L2 norm was 65x the attention
    component's and grew with node count, so it dominated the GRU input and drowned the other
    three. Magnitude information has to be supplied on a comparable scale to be usable, which
    is the same reasoning behind degree-scalers in Principal Neighbourhood Aggregation
    (Corso et al., 2020).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.out_dim = input_dim * 3 + 1

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor,
                num_graphs: Optional[int] = None) -> torch.Tensor:
        if num_graphs is None:
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0

        scores = self.attn(x).squeeze(-1)
        weights = scatter_softmax(scores, batch_idx)
        attn_out = x.new_zeros(num_graphs, x.size(-1)).index_add(
            0, batch_idx, x * weights.unsqueeze(-1))

        counts = x.new_zeros(num_graphs).index_add(
            0, batch_idx, x.new_ones(x.size(0)))
        sum_out = x.new_zeros(num_graphs, x.size(-1)).index_add(0, batch_idx, x)
        mean_out = sum_out / counts.clamp_min(1.0).unsqueeze(-1)

        max_out = x.new_full((num_graphs, x.size(-1)), float("-inf"))
        max_out = max_out.index_reduce(0, batch_idx, x, "amax", include_self=True)
        max_out = torch.where(torch.isinf(max_out), torch.zeros_like(max_out), max_out)

        # log1p keeps the scale comparable to the learned embedding dimensions; the raw count
        # runs to several hundred and would otherwise dominate the GRU input.
        log_count = torch.log1p(counts).unsqueeze(-1)

        return torch.cat([attn_out, mean_out, max_out, log_count], dim=-1)


class GatedScalePool(nn.Module):
    """Attention pooling with magnitude branches added through zero-initialised gates.

    `ScaleAwarePool` above concatenates four readouts and returns `input_dim * 3 + 1`
    features. Two consequences followed, and both were mistakes:

      * `pooled_dim` feeds the temporal GRU, so that variant's GRU input width TRIPLED and it
        carried substantially more parameters than the baseline it was compared against. On
        synthetic_gap0125 it reached train loss 0.16 while validation AUROC stayed near chance,
        finishing at 0.626 against the baseline's 0.889 -- measured with early stopping
        disabled over the full 40-epoch budget, so a truncated-training explanation is ruled
        out. That is a capacity failure, not evidence that magnitude information is useless.
      * At initialisation the model was not "the baseline plus something", it was a different
        model, with no path back to baseline behaviour if the extra readouts proved unhelpful.

    This version keeps the output width at `input_dim`, leaving the GRU untouched and the
    parameter count matched to the baseline to within three scalars. The magnitude readouts
    enter as residual branches, each behind its own scalar gate initialised to zero:

        out = attn + g_mean * LN(mean) + g_max * LN(max) + g_count * proj(log1p(count))

    Every gate starts at exactly 0, so the module computes precisely what `AttentionPool`
    computes and the variant *begins* as the baseline. A branch contributes only insofar as
    training moves its gate off zero, which makes this a strict test of whether magnitude
    information helps: it can no longer lose on capacity or on initialisation. This is the
    ReZero construction (Bachlechner et al., 2021), used there so deep residual stacks train
    without warmup and used here for the same reason -- an added branch must not perturb the
    function at step 0.

    The LayerNorms address why the original raw-sum version failed: measured on a trained
    checkpoint, the sum's L2 norm was 65x the attention output's and grew with node count, so
    it drowned the other branches. Normalising each branch puts them on a comparable scale
    before the gate decides how much to admit -- the same reasoning behind the degree-scalers
    in Principal Neighbourhood Aggregation (Corso et al., 2020).

    The learned gates are reportable in their own right: a gate that stays near zero after
    training is direct evidence the readout had nothing to contribute, which is a cleaner
    statement than an AUROC difference.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.norm_mean = nn.LayerNorm(input_dim)
        self.norm_max = nn.LayerNorm(input_dim)
        self.count_proj = nn.Linear(1, input_dim)
        # One scalar per branch, all zero: exactly AttentionPool at step 0.
        self.gate = nn.Parameter(torch.zeros(3))
        self.out_dim = input_dim

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor,
                num_graphs: Optional[int] = None) -> torch.Tensor:
        if num_graphs is None:
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0

        scores = self.attn(x).squeeze(-1)
        weights = scatter_softmax(scores, batch_idx)
        attn_out = x.new_zeros(num_graphs, x.size(-1)).index_add(
            0, batch_idx, x * weights.unsqueeze(-1))

        counts = x.new_zeros(num_graphs).index_add(0, batch_idx, x.new_ones(x.size(0)))
        sum_out = x.new_zeros(num_graphs, x.size(-1)).index_add(0, batch_idx, x)
        mean_out = sum_out / counts.clamp_min(1.0).unsqueeze(-1)

        max_out = x.new_full((num_graphs, x.size(-1)), float("-inf"))
        max_out = max_out.index_reduce(0, batch_idx, x, "amax", include_self=True)
        max_out = torch.where(torch.isinf(max_out), torch.zeros_like(max_out), max_out)

        count_out = self.count_proj(torch.log1p(counts).unsqueeze(-1))

        return (attn_out
                + self.gate[0] * self.norm_mean(mean_out)
                + self.gate[1] * self.norm_max(max_out)
                + self.gate[2] * count_out)


class TemporalAggregator(nn.Module):
    """GRU that reads sequence of graph-level embeddings over T timesteps."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, graph_sequence: torch.Tensor) -> torch.Tensor:
        output, h_n = self.gru(graph_sequence)
        return h_n.squeeze(0)


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
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class NodeAnomalyHead(nn.Module):
    """MLP that scores per-node anomaly probability."""

    def __init__(self, node_dim: int, traj_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + traj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, trajectory_emb: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        traj_per_node = trajectory_emb[batch_idx]  # Broadcast trajectory to each node
        combined = torch.cat([node_embeddings, traj_per_node], dim=-1)
        return torch.sigmoid(self.mlp(combined)).squeeze(-1)


class TGNDetector(nn.Module):
    """
    Full Temporal GNN Detector for Memory Poisoning.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]

        # +3 metadata columns by default (recency, retrieval freq, retrieval recency).
        # Raise `node_meta_dim` when the dataset also supplies structural node features
        # (see src/data_generation/dataset.py::_structural_features).
        input_dim = model_cfg["node_feature_dim"] + model_cfg.get("node_meta_dim", 3)
        memory_dim = model_cfg["node_memory_dim"]
        gat_hidden = model_cfg["gat_hidden_dim"]
        gat_heads = model_cfg["gat_heads"]
        gru_hidden = model_cfg["gru_hidden_dim"]
        mlp_hidden = model_cfg["mlp_hidden_dim"]
        dropout = model_cfg["dropout"]

        self.node_memory = NodeMemoryModule(input_dim, memory_dim)
        self.graph_encoder = GraphEncoder(
            input_dim=input_dim + memory_dim,
            hidden_dim=gat_hidden,
            num_heads=gat_heads,
            dropout=dropout,
            edge_dim=1,
        )
        # Opt-in so each change stays a clean ablation against the published numbers.
        # `scale_readout: true` is kept as an alias for readout "scale" so the checkpoints
        # already on disk still load and their numbers stay reproducible from this tree.
        readout = model_cfg.get("readout", "scale" if model_cfg.get("scale_readout") else "attention")
        self.readout = readout
        self.scale_readout = readout == "scale"
        if readout == "scale":
            self.attn_pool = ScaleAwarePool(input_dim=gat_hidden, hidden_dim=mlp_hidden)
            pooled_dim = self.attn_pool.out_dim
        elif readout == "gated":
            self.attn_pool = GatedScalePool(input_dim=gat_hidden, hidden_dim=mlp_hidden)
            pooled_dim = self.attn_pool.out_dim
        elif readout == "attention":
            self.attn_pool = AttentionPool(input_dim=gat_hidden, hidden_dim=mlp_hidden)
            pooled_dim = gat_hidden
        else:
            raise ValueError(f"unknown readout {readout!r}; "
                             f"expected 'attention', 'scale' or 'gated'")
        self.temporal_aggregator = TemporalAggregator(
            input_dim=pooled_dim,
            hidden_dim=gru_hidden,
        )
        self.graph_head = GraphAnomalyHead(gru_hidden, mlp_hidden)
        self.node_head = NodeAnomalyHead(gat_hidden, gru_hidden, mlp_hidden)

        self.gat_hidden = gat_hidden
        self.gru_hidden = gru_hidden
        self.memory_dim = memory_dim
        self._memory_tensor: Optional[torch.Tensor] = None
        # How the memory bank is addressed when several windows are batched together.
        # The dataset gives every snapshot node_ids = arange(n), and Batch.from_data_list does
        # not offset them (it offsets only attributes named like "index"), so under "shared"
        # node i of every window in a batch reads and writes ONE slot: state crosses windows,
        # and with shuffle=False it crosses from later windows of the same sequence.
        # "per_window" offsets each window's ids by its position in the batch. At batch size 1
        # the offset is zero and the two modes are bit-identical, so every checkpoint scores
        # the same at batch 1 under either. "per_window" is the default and the architecture
        # as specified; every result in RESULTS_FREEZE.md was trained and scored with it.
        # "shared" is the legacy behaviour the pre-freeze runs (run_burst2k_*) used, kept only
        # so those can be re-scored as they were; see src/experiments/score_windows.py.
        self.memory_slots = model_cfg.get("memory_slots", "per_window")
        if self.memory_slots not in ("shared", "per_window"):
            raise ValueError(f"unknown memory_slots {self.memory_slots!r}")

    def reset_memory(self):
        self._memory_tensor = None

    def _memory_stride(self, graph_sequence: List[Batch]) -> int:
        if self.memory_slots == "shared":
            return 0
        ids = [g.node_ids for g in graph_sequence
               if getattr(g, "node_ids", None) is not None and g.node_ids.numel()]
        return max(int(i.max().item()) for i in ids) + 1 if ids else 0

    @staticmethod
    def _slot_ids(g, stride: int) -> Optional[torch.Tensor]:
        """Memory-bank addresses for the nodes of one batched snapshot."""
        ids = getattr(g, "node_ids", None)
        if ids is None or stride == 0 or getattr(g, "batch", None) is None:
            return ids
        return ids + g.batch * stride

    def _ensure_memory_tensor(self, required_size: int, device: torch.device):
        if self._memory_tensor is None or self._memory_tensor.size(0) < required_size:
            alloc_size = max(required_size, (self._memory_tensor.size(0) * 2) if self._memory_tensor is not None else 1000)
            new_tensor = torch.zeros(alloc_size, self.memory_dim, device=device)
            if self._memory_tensor is not None:
                new_tensor[:self._memory_tensor.size(0)] = self._memory_tensor
            self._memory_tensor = new_tensor

    def _get_node_memories(self, node_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        if node_ids is None or node_ids.numel() == 0:
            return torch.zeros(0, self.memory_dim, device=device)

        max_id = int(node_ids.max().item()) + 1
        self._ensure_memory_tensor(max_id, device)
        return self._memory_tensor[node_ids]

    def _store_node_memories(self, node_ids: torch.Tensor, memories: torch.Tensor):
        if node_ids is None or node_ids.numel() == 0:
            return

        max_id = int(node_ids.max().item()) + 1
        self._ensure_memory_tensor(max_id, memories.device)
        self._memory_tensor[node_ids] = memories.detach()

    def forward(self, graph_sequence: List[Batch]) -> Tuple[torch.Tensor, torch.Tensor]:
        T = len(graph_sequence)
        graph_embeddings = []
        last_node_embeddings = None
        last_batch_idx = None

        self.reset_memory()
        stride = self._memory_stride(graph_sequence)

        for t in range(T):
            g = graph_sequence[t]
            x = g.x
            edge_index = g.edge_index
            batch_idx = g.batch
            edge_attr = g.edge_attr if hasattr(g, 'edge_attr') and g.edge_attr is not None else None

            slots = self._slot_ids(g, stride)
            prev_memory = (self._get_node_memories(slots, x.device)
                           if slots is not None else None)

            node_mem = self.node_memory(x, prev_memory)

            if slots is not None:
                self._store_node_memories(slots, node_mem)

            gat_input = torch.cat([x, node_mem], dim=-1)
            node_embeddings = self.graph_encoder(gat_input, edge_index, edge_attr=edge_attr)
            num_graphs = g.num_graphs if hasattr(g, "num_graphs") else None
            graph_emb = self.attn_pool(node_embeddings, batch_idx, num_graphs=num_graphs)
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
        stride = self._memory_stride(graph_sequence)

        for t in range(T):
            g = graph_sequence[t]
            x = g.x
            edge_index = g.edge_index
            batch_idx = g.batch
            edge_attr = g.edge_attr if hasattr(g, 'edge_attr') and g.edge_attr is not None else None

            slots = self._slot_ids(g, stride)
            prev_memory = (self._get_node_memories(slots, x.device)
                           if slots is not None else None)

            node_mem = self.node_memory(x, prev_memory)

            if slots is not None:
                self._store_node_memories(slots, node_mem)

            gat_input = torch.cat([x, node_mem], dim=-1)
            node_embeddings = self.graph_encoder(gat_input, edge_index, edge_attr=edge_attr)
            num_graphs = g.num_graphs if hasattr(g, "num_graphs") else None
            graph_emb = self.attn_pool(node_embeddings, batch_idx, num_graphs=num_graphs)
            graph_embeddings.append(graph_emb)

        graph_seq_tensor = torch.stack(graph_embeddings, dim=1)
        return self.temporal_aggregator(graph_seq_tensor)
