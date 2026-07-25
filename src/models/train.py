"""
Training script for the TGN Memory Poisoning Detector.
Implements Phase 1 (contrastive pre-training) and Phase 2 (supervised fine-tuning).

Usage:
    python -m src.models.train --config configs/default.yaml --data data/raw/llm_generated
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import yaml

from src.models.tgn_detector import TGNDetector
from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in node-level detection."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target.float(), reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss for trajectory embeddings.
    
    Uses labels to construct positive pairs (same class) and negative pairs
    (different class). This is more principled than NT-Xent with random
    augmentations when labels are available.
    
    Reference: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch_size, dim) — trajectory embeddings
            labels: (batch_size,) — 0 (clean) or 1 (poisoned)
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, dim=1)

        # Pairwise cosine similarity
        sim = torch.mm(features, features.t()) / self.temperature  # (B, B)

        # Mask out self-similarity
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, -1e9)

        # Positive mask: same label (excluding self)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & ~self_mask  # (B, B)

        # If no positive pairs exist, return 0
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Log-softmax over non-self similarities
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # Mean of log-prob over positive pairs
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)

        loss = -mean_log_prob_pos.mean()
        return loss


class Trainer:
    """Handles training loop for both phases."""

    def __init__(self, config: dict, data_dir: str, device: str = "auto"):
        self.config = config
        self.train_cfg = config["training"]
        self.data_dir = data_dir

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = TGNDetector(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Initialize datasets
        T = self.train_cfg["temporal_window"]
        embedding_dim = config["simulation"]["embedding_dim"]
        self.train_dataset = MemoryTrajectoryDataset(data_dir, split="train", temporal_window=T, embedding_dim=embedding_dim)
        self.val_dataset = MemoryTrajectoryDataset(data_dir, split="val", temporal_window=T, embedding_dim=embedding_dim)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
            collate_fn=collate_trajectories,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=False,
            collate_fn=collate_trajectories,
        )

        # Loss functions
        self.graph_loss_fn = nn.BCELoss()
        self.node_loss_fn = FocalLoss(gamma=self.train_cfg["focal_loss_gamma"])
        self.contrastive_loss_fn = SupConLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_cfg["epochs_supervised"]
        )

    def train_phase1_contrastive(self, save_dir: str):
        """Phase 1: Supervised contrastive pre-training on trajectory embeddings.
        
        Uses labels to group same-class trajectories as positive pairs,
        pulling clean trajectories together and poisoned trajectories together
        while pushing the two clusters apart in embedding space.
        """
        print("\n=== Phase 1: Supervised Contrastive Pre-training ===")
        epochs = self.train_cfg["epochs_contrastive"]
        patience = self.train_cfg["early_stopping_patience"]
        best_loss = float("inf")
        no_improve = 0

        # Separate LR scheduler for Phase 1
        phase1_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        phase1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            phase1_optimizer, T_max=epochs
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                graphs = [g.to(self.device) for g in batch["graphs"]]
                labels = batch["graph_labels"].to(self.device)

                # Need at least 2 samples with different labels for contrastive learning
                if labels.size(0) < 2:
                    continue

                # Get trajectory embeddings
                embeddings = self.model.get_trajectory_embedding(graphs)

                # Supervised contrastive loss using actual labels
                loss = self.contrastive_loss_fn(embeddings, labels)

                phase1_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg["gradient_clip"])
                phase1_optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            phase1_scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Epoch {epoch+1}: Contrastive Loss = {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                torch.save(self.model.state_dict(), Path(save_dir) / "phase1_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    def train_phase2_supervised(self, save_dir: str):
        """Phase 2: Supervised fine-tuning with multi-task loss."""
        print("\n=== Phase 2: Supervised Fine-tuning ===")
        epochs = self.train_cfg["epochs_supervised"]
        patience = self.train_cfg["early_stopping_patience"]
        alpha = self.train_cfg["loss_alpha"]
        beta = self.train_cfg["loss_beta"]

        best_val_loss = float("inf")
        no_improve = 0
        history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_node_auroc": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = self._train_epoch(alpha, beta)
            history["train_loss"].append(train_loss)

            # Validation (now includes node-level loss to match training objective)
            val_loss, val_auroc, val_node_auroc = self._validate(alpha, beta)
            history["val_loss"].append(val_loss)
            history["val_auroc"].append(val_auroc)
            history["val_node_auroc"].append(val_node_auroc)

            self.scheduler.step()

            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val AUROC={val_auroc:.4f}, Val Node AUROC={val_node_auroc:.4f}")

            # Early stopping on val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), Path(save_dir) / "phase2_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Save final model (regardless of val loss)
        torch.save(self.model.state_dict(), Path(save_dir) / "phase2_final.pt")

        # Save training history
        with open(Path(save_dir) / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"  Models saved: phase2_best.pt (best val) + phase2_final.pt (last epoch)")

    def _train_epoch(self, alpha: float, beta: float) -> float:
        """Run one training epoch."""
        self.model.reset_memory()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            graphs = [g.to(self.device) for g in batch["graphs"]]
            graph_labels = batch["graph_labels"].to(self.device)

            # Forward pass
            graph_scores, node_scores = self.model(graphs)

            # Graph-level loss
            l_graph = self.graph_loss_fn(graph_scores, graph_labels)

            # Node-level loss (from last timestep)
            last_graph = graphs[-1]
            if hasattr(last_graph, "node_labels"):
                node_labels = last_graph.node_labels.to(self.device).float()
                # Ensure sizes match
                min_len = min(node_scores.size(0), node_labels.size(0))
                l_node = self.node_loss_fn(node_scores[:min_len], node_labels[:min_len])
            else:
                l_node = torch.tensor(0.0, device=self.device)

            # Combined loss
            loss = alpha * l_graph + beta * l_node

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg["gradient_clip"])
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, alpha: float, beta: float) -> tuple:
        """Run validation and compute metrics.
        
        Now includes both graph-level and node-level loss to match the
        training objective, plus AUROC for both levels.
        """
        from sklearn.metrics import roc_auc_score

        self.model.eval()
        self.model.reset_memory()
        total_loss = 0
        num_batches = 0
        all_graph_scores = []
        all_graph_labels = []
        all_node_scores = []
        all_node_labels = []

        for batch in self.val_loader:
            graphs = [g.to(self.device) for g in batch["graphs"]]
            graph_labels = batch["graph_labels"].to(self.device)

            graph_scores, node_scores = self.model(graphs)

            # Graph-level loss
            l_graph = self.graph_loss_fn(graph_scores, graph_labels)

            # Node-level loss (matching training objective)
            last_graph = graphs[-1]
            if hasattr(last_graph, "node_labels"):
                node_labels_t = last_graph.node_labels.to(self.device).float()
                min_len = min(node_scores.size(0), node_labels_t.size(0))
                l_node = self.node_loss_fn(node_scores[:min_len], node_labels_t[:min_len])

                all_node_scores.extend(node_scores[:min_len].cpu().numpy())
                all_node_labels.extend(node_labels_t[:min_len].cpu().numpy())
            else:
                l_node = torch.tensor(0.0, device=self.device)

            total_loss += (alpha * l_graph + beta * l_node).item()
            num_batches += 1

            all_graph_scores.extend(graph_scores.cpu().numpy())
            all_graph_labels.extend(graph_labels.cpu().numpy())

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(num_batches, 1)

        # Compute graph-level AUROC
        all_graph_scores = np.array(all_graph_scores)
        all_graph_labels = np.array(all_graph_labels)
        if len(np.unique(all_graph_labels)) > 1:
            graph_auroc = roc_auc_score(all_graph_labels, all_graph_scores)
        else:
            graph_auroc = 0.0

        # Compute node-level AUROC
        all_node_scores = np.array(all_node_scores)
        all_node_labels = np.array(all_node_labels)
        if len(all_node_labels) > 0 and len(np.unique(all_node_labels)) > 1:
            node_auroc = roc_auc_score(all_node_labels, all_node_scores)
        else:
            node_auroc = 0.0

        return avg_loss, graph_auroc, node_auroc


def main():
    parser = argparse.ArgumentParser(description="Train TGN Memory Poisoning Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default="data/raw/llm_generated")
    parser.add_argument("--output", type=str, default="experiments/run_001")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip contrastive pre-training")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set reproducibility seeds
    set_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    trainer = Trainer(config, args.data, device=args.device)

    # Phase 1: Contrastive pre-training
    if not args.skip_phase1:
        trainer.train_phase1_contrastive(str(save_dir))

    # Phase 2: Supervised fine-tuning
    trainer.train_phase2_supervised(str(save_dir))

    print(f"\nTraining complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
