"""
Graph Noise and OOD Perturbation Transforms Module
==================================================
Provides algorithmic noise transforms to evaluate model robustness under Out-of-Distribution (OOD)
conditions: Gaussian embedding noise, cosine similarity threshold shifts, metadata corruption,
and edge dropouts.

GPU-accelerated matrix operations.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Optional


class GraphNoiseTransform:
    """
    Transforms graph snapshots and trajectory windows with controlled noise perturbations.
    """

    @staticmethod
    def add_embedding_noise(data: Data, sigma: float = 0.10, embedding_dim: int = 384, device: torch.device = None) -> Data:
        """
        Adds zero-mean Gaussian noise N(0, sigma^2) to textual node embedding features.
        Preserves metadata features (last 3 columns).
        """
        if sigma <= 0.0:
            return data

        # .to(device) is a no-op (same tensor, not a copy) when data.x is already
        # on that device - always clone explicitly so this never mutates the
        # caller's tensor in place (it's frequently a shared, preloaded tensor
        # reused across many sweep calls).
        x = data.x.to(device).clone() if device is not None else data.x.clone()

        noise = torch.randn_like(x[:, :embedding_dim]) * sigma
        x[:, :embedding_dim] += noise

        return Data(
            x=x,
            edge_index=data.edge_index.to(x.device),
            edge_attr=data.edge_attr.to(x.device) if data.edge_attr is not None else None,
            y=data.y,
            node_labels=data.node_labels,
            node_ids=data.node_ids,
        )

    @staticmethod
    def apply_edge_threshold_shift(data: Data, new_theta: float, embedding_dim: int = 384, device: torch.device = None) -> Data:
        """
        Re-computes graph edge_index and edge_attr using a new cosine similarity threshold.
        Accelerated on GPU.
        """
        target_device = device if device is not None else data.x.device
        x_emb = data.x[:, :embedding_dim].to(target_device)
        num_nodes = x_emb.size(0)

        if num_nodes <= 1:
            return data

        # Normalize embeddings for cosine similarity on GPU
        norm_emb = x_emb / (torch.norm(x_emb, dim=-1, keepdim=True) + 1e-8)
        sim_matrix = torch.mm(norm_emb, norm_emb.t())

        # Mask self-loops and filter by threshold
        mask = (sim_matrix >= new_theta) & (~torch.eye(num_nodes, dtype=torch.bool, device=target_device))
        edge_indices = torch.nonzero(mask, as_tuple=False).t()

        if edge_indices.numel() > 0:
            edge_weights = sim_matrix[edge_indices[0], edge_indices[1]].unsqueeze(-1)
        else:
            edge_indices = torch.zeros((2, 0), dtype=torch.long, device=target_device)
            edge_weights = torch.zeros((0, 1), dtype=torch.float32, device=target_device)

        full_x = data.x.to(target_device)
        return Data(
            x=full_x,
            edge_index=edge_indices,
            edge_attr=edge_weights,
            y=data.y,
            node_labels=data.node_labels,
            node_ids=data.node_ids,
        )

    @staticmethod
    def corrupt_metadata_jitter(data: Data, corrupt_prob: float = 0.10, embedding_dim: int = 384, device: torch.device = None) -> Data:
        """
        Adds random noise/corruption to operational metadata features (last 3 columns).
        """
        if corrupt_prob <= 0.0:
            return data

        target_device = device if device is not None else data.x.device
        x = data.x.to(target_device).clone()
        num_nodes = x.size(0)

        if num_nodes == 0:
            return data

        meta_mask = torch.rand(num_nodes, 3, device=target_device) < corrupt_prob
        jitter = torch.rand(num_nodes, 3, device=target_device)  # Random uniform noise [0, 1]
        x[:, embedding_dim:][meta_mask] = jitter[meta_mask]

        return Data(
            x=x,
            edge_index=data.edge_index.to(target_device),
            edge_attr=data.edge_attr.to(target_device) if data.edge_attr is not None else None,
            y=data.y,
            node_labels=data.node_labels,
            node_ids=data.node_ids,
        )

    @staticmethod
    def drop_random_edges(data: Data, drop_rate: float = 0.15, device: torch.device = None) -> Data:
        """
        Randomly drops a fraction of graph edges to simulate indexing loss.
        """
        if drop_rate <= 0.0 or data.edge_index.numel() == 0:
            return data

        target_device = device if device is not None else data.edge_index.device
        edge_idx = data.edge_index.to(target_device)
        edge_attr = data.edge_attr.to(target_device) if data.edge_attr is not None else None

        num_edges = edge_idx.size(1)
        keep_mask = torch.rand(num_edges, device=target_device) >= drop_rate

        new_edge_index = edge_idx[:, keep_mask]
        new_edge_attr = edge_attr[keep_mask] if edge_attr is not None and edge_attr.numel() > 0 else edge_attr

        return Data(
            x=data.x.to(target_device),
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=data.y,
            node_labels=data.node_labels,
            node_ids=data.node_ids,
        )

    @classmethod
    def perturb_trajectory(cls, graphs: List[Data],
                           embedding_sigma: float = 0.0,
                           threshold: Optional[float] = None,
                           meta_corrupt_prob: float = 0.0,
                           edge_drop_rate: float = 0.0,
                           device: torch.device = None) -> List[Data]:
        """
        Applies a pipeline of GPU-accelerated noise perturbations across all T snapshots.
        """
        perturbed_graphs = []
        for g in graphs:
            p_g = g
            if embedding_sigma > 0.0:
                p_g = cls.add_embedding_noise(p_g, sigma=embedding_sigma, device=device)
            if threshold is not None:
                p_g = cls.apply_edge_threshold_shift(p_g, new_theta=threshold, device=device)
            if meta_corrupt_prob > 0.0:
                p_g = cls.corrupt_metadata_jitter(p_g, corrupt_prob=meta_corrupt_prob, device=device)
            if edge_drop_rate > 0.0:
                p_g = cls.drop_random_edges(p_g, drop_rate=edge_drop_rate, device=device)
            perturbed_graphs.append(p_g)
        return perturbed_graphs
