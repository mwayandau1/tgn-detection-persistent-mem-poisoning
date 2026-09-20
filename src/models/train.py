"""
Training script for the TGN memory-poisoning detector.
Phase 1: supervised contrastive pre-training on trajectory embeddings.
Phase 2: supervised fine-tuning with joint graph-level + node-level loss.

Adds two robustness measures on top of the original notebook training loop:
  - embedding_noise augmentation during Phase 2 (randomly perturbs node text
    embeddings each batch, so the model can't rely on a brittle, noise-free
    decision boundary in raw embedding space).
  - an optional --holdout-attack-type flag: excludes every sequence of that
    attack type from train/val entirely, then reports (does not use for
    checkpoint selection) the trained model's AUROC on that unseen attack
    type, as a generalization check distinct from the random train/test split.
"""

import sys
import json
import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.tgn_detector import TGNDetector
from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories

_EMBEDDING_DIM = 384


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
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
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        if batch_size < 2:
            # Disconnected zero would crash .backward() (no grad_fn); keep it
            # attached to the graph so this batch just contributes zero grad.
            return features.sum() * 0.0

        features = F.normalize(features, dim=1)
        sim = torch.mm(features, features.t()) / self.temperature
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        sim = sim.masked_fill(self_mask, -1e9)

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & ~self_mask

        if pos_mask.sum() == 0:
            return features.sum() * 0.0

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)
        return -mean_log_prob_pos.mean()


def apply_embedding_noise_augment(graphs: list, prob: float, sigma_max: float,
                                   embedding_dim: int = _EMBEDDING_DIM) -> list:
    """Randomly perturbs node text embeddings (in place on cloned tensors) for a
    fraction of training batches. Sigma is sampled per-batch from U(0, sigma_max).
    Edge structure, node_ids, and labels are untouched - only meant to stop the
    model from relying on exact, noise-free embedding coordinates."""
    if random.random() >= prob:
        return graphs
    sigma = random.uniform(0.0, sigma_max)
    if sigma <= 0.0:
        return graphs
    for g in graphs:
        g.x = g.x.clone()
        noise = torch.randn_like(g.x[:, :embedding_dim]) * sigma
        g.x[:, :embedding_dim] = g.x[:, :embedding_dim] + noise
    return graphs


class Trainer:
    def __init__(self, config: dict, data_dir: str, device: torch.device,
                 holdout_attack_type: Optional[str] = None,
                 noise_augment_prob: float = 0.0, noise_sigma_max: float = 0.1,
                 model_cls: type = TGNDetector,
                 structural_features: bool = False,
                 select_by: str = "auroc",
                 scale_readout: bool = False,
                 readout: Optional[str] = None):
        self.config = config
        self.train_cfg = config["training"]
        self.data_dir = Path(data_dir)
        self.device = device
        self.noise_augment_prob = noise_augment_prob
        self.noise_sigma_max = noise_sigma_max
        self.structural_features = structural_features
        self.select_by = select_by
        # Must be set before the model is constructed: it changes the pooled dimension
        # the temporal GRU is built to accept.
        config["model"]["scale_readout"] = bool(scale_readout)
        if readout:
            config["model"]["readout"] = readout
        if scale_readout:
            print("  Scale-aware readout: attention + sum + max + log node count")
        if readout == "gated":
            print("  Gated scale readout: attention + zero-initialised mean/max/count gates "
                  "(identical to attention pooling at initialisation)")

        # The model's input width must match what the dataset emits.
        if structural_features:
            config["model"]["node_meta_dim"] = 3 + 4
            print("Structural node features enabled (degree + edge-weight stats): "
                  "node_meta_dim=7")

        self.model = model_cls(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        T = self.train_cfg["temporal_window"]
        embedding_dim = config["simulation"]["embedding_dim"]

        train_ids, val_ids, holdout_ids = self._build_splits(holdout_attack_type)
        self.holdout_ids = holdout_ids

        self.train_dataset = MemoryTrajectoryDataset(
            data_dir, temporal_window=T, embedding_dim=embedding_dim, sequence_ids=train_ids,
            structural_features=structural_features)
        self.val_dataset = MemoryTrajectoryDataset(
            data_dir, temporal_window=T, embedding_dim=embedding_dim, sequence_ids=val_ids,
            structural_features=structural_features)
        self.holdout_dataset = None
        if holdout_ids:
            self.holdout_dataset = MemoryTrajectoryDataset(
                data_dir, temporal_window=T, embedding_dim=embedding_dim, sequence_ids=holdout_ids,
                structural_features=structural_features)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.train_cfg["batch_size"],
            shuffle=True, collate_fn=collate_trajectories,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.train_cfg["batch_size"],
            shuffle=False, collate_fn=collate_trajectories,
        )
        self.holdout_loader = None
        if self.holdout_dataset is not None:
            self.holdout_loader = DataLoader(
                self.holdout_dataset, batch_size=self.train_cfg["batch_size"],
                shuffle=False, collate_fn=collate_trajectories,
            )

        self.graph_loss_fn = nn.BCELoss()
        self.node_loss_fn = FocalLoss(gamma=self.train_cfg["focal_loss_gamma"],
                                       alpha=self.train_cfg.get("focal_loss_alpha", 0.75))
        self.contrastive_loss_fn = SupConLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_cfg["epochs_supervised"]
        )

    def _build_splits(self, holdout_attack_type: Optional[str]):
        with open(self.data_dir / "splits.json") as f:
            splits = json.load(f)
        with open(self.data_dir / "metadata.json") as f:
            metadata = {m["sequence_id"]: m for m in json.load(f)}

        if not holdout_attack_type:
            return splits["train"], splits["val"], []

        def is_holdout(seq_id):
            return metadata[seq_id].get("attack_type") == holdout_attack_type

        train_ids = [s for s in splits["train"] if not is_holdout(s)]
        val_ids = [s for s in splits["val"] if not is_holdout(s)]
        holdout_ids = [s for s in (splits["train"] + splits["val"] + splits["test"]) if is_holdout(s)]

        n_removed = (len(splits["train"]) - len(train_ids)) + (len(splits["val"]) - len(val_ids))
        print(f"Holdout attack type '{holdout_attack_type}': removed {n_removed} sequences from "
              f"train/val, {len(holdout_ids)} sequences reserved for generalization eval.")
        return train_ids, val_ids, holdout_ids

    def train_phase1_contrastive(self, save_dir: str):
        print("\n=== Phase 1: Supervised Contrastive Pre-training ===")
        epochs = self.train_cfg["epochs_contrastive"]
        patience = self.train_cfg["early_stopping_patience"]
        best_loss = float("inf")
        no_improve = 0

        phase1_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        phase1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(phase1_optimizer, T_max=epochs)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch in tqdm(self.train_loader, desc=f"Phase 1 Epoch {epoch+1}/{epochs}"):
                graphs = [g.to(self.device) for g in batch["graphs"]]
                labels = batch["graph_labels"].to(self.device)

                if labels.size(0) < 2:
                    continue

                embeddings = self.model.get_trajectory_embedding(graphs)
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
        print("\n=== Phase 2: Supervised Fine-tuning ===")
        print(f"  Embedding-noise augmentation: prob={self.noise_augment_prob}, sigma_max={self.noise_sigma_max}")
        epochs = self.train_cfg["epochs_supervised"]
        patience = self.train_cfg["early_stopping_patience"]
        alpha = self.train_cfg["loss_alpha"]
        beta = self.train_cfg["loss_beta"]

        # Checkpoint selection criterion.
        #
        # Selecting on validation LOSS silently picks the wrong model whenever loss and
        # AUROC diverge, which they do on any benchmark hard enough for the model to
        # start overfitting: on data/raw/synthetic_hard val loss rose monotonically
        # (0.648 -> 0.966) while val AUROC climbed (0.335 -> 0.476), so loss-based
        # selection saved epoch 1 — the worst checkpoint by the metric actually
        # reported. Default to selecting on the reported metric; pass
        # `--select-by loss` for the old behaviour.
        select_by = getattr(self, "select_by", "auroc")
        print(f"  Checkpoint selection: best validation {select_by}")
        best_score = float("inf") if select_by == "loss" else -float("inf")
        best_epoch = 0
        no_improve = 0
        history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_node_auroc": []}

        for epoch in range(epochs):
            self.model.train()
            train_loss = self._train_epoch(alpha, beta, epoch, epochs)
            history["train_loss"].append(train_loss)

            val_loss, val_auroc, val_node_auroc = self._evaluate(self.val_loader, alpha, beta)
            history["val_loss"].append(val_loss)
            history["val_auroc"].append(val_auroc)
            history["val_node_auroc"].append(val_node_auroc)

            self.scheduler.step()

            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val AUROC={val_auroc:.4f}, Val Node AUROC={val_node_auroc:.4f}")

            improved = (val_loss < best_score) if select_by == "loss" else (val_auroc > best_score)
            if improved:
                best_score = val_loss if select_by == "loss" else val_auroc
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(self.model.state_dict(), Path(save_dir) / "phase2_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        torch.save(self.model.state_dict(), Path(save_dir) / "phase2_final.pt")
        history["selection"] = {"criterion": select_by, "best_epoch": best_epoch,
                                "best_score": best_score}
        with open(Path(save_dir) / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"  Best checkpoint: epoch {best_epoch} (val {select_by}={best_score:.4f})")
        print(f"  Models saved to {save_dir}")

    def _train_epoch(self, alpha: float, beta: float, epoch: int, epochs: int) -> float:
        self.model.reset_memory()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Phase 2 Epoch {epoch+1}/{epochs}", leave=False):
            graphs = [g.to(self.device) for g in batch["graphs"]]
            graphs = apply_embedding_noise_augment(graphs, self.noise_augment_prob, self.noise_sigma_max)
            graph_labels = batch["graph_labels"].to(self.device)

            graph_scores, node_scores = self.model(graphs)
            l_graph = self.graph_loss_fn(graph_scores, graph_labels)

            last_graph = graphs[-1]
            if hasattr(last_graph, "node_labels"):
                node_labels = last_graph.node_labels.to(self.device).float()
                min_len = min(node_scores.size(0), node_labels.size(0))
                l_node = self.node_loss_fn(node_scores[:min_len], node_labels[:min_len])
            else:
                l_node = torch.tensor(0.0, device=self.device)

            loss = alpha * l_graph + beta * l_node

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg["gradient_clip"])
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, alpha: float, beta: float) -> tuple:
        self.model.eval()
        self.model.reset_memory()
        total_loss = 0
        num_batches = 0
        all_graph_scores, all_graph_labels = [], []
        all_node_scores, all_node_labels = [], []

        for batch in loader:
            graphs = [g.to(self.device) for g in batch["graphs"]]
            graph_labels = batch["graph_labels"].to(self.device)

            graph_scores, node_scores = self.model(graphs)
            l_graph = self.graph_loss_fn(graph_scores, graph_labels)

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

        all_graph_scores = np.array(all_graph_scores)
        all_graph_labels = np.array(all_graph_labels)
        graph_auroc = (roc_auc_score(all_graph_labels, all_graph_scores)
                       if len(np.unique(all_graph_labels)) > 1 else 0.0)

        all_node_scores = np.array(all_node_scores)
        all_node_labels = np.array(all_node_labels)
        node_auroc = (roc_auc_score(all_node_labels, all_node_scores)
                      if len(all_node_labels) > 0 and len(np.unique(all_node_labels)) > 1 else 0.0)

        return avg_loss, graph_auroc, node_auroc

    def evaluate_holdout(self) -> Optional[dict]:
        if self.holdout_loader is None:
            return None
        alpha = self.train_cfg["loss_alpha"]
        beta = self.train_cfg["loss_beta"]
        loss, graph_auroc, node_auroc = self._evaluate(self.holdout_loader, alpha, beta)
        return {"loss": loss, "graph_auroc": graph_auroc, "node_auroc": node_auroc,
                "n_sequences": len(self.holdout_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/raw/llm_generated")
    parser.add_argument("--output", default="experiments/run_001")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--holdout-attack-type", default=None,
                         help="Attack type name to exclude entirely from train/val, "
                              "evaluated afterward as a generalization check.")
    parser.add_argument("--noise-augment-prob", type=float, default=0.5,
                         help="Probability that a given Phase 2 training batch gets embedding noise.")
    parser.add_argument("--noise-sigma-max", type=float, default=0.15,
                         help="Max sigma for the training-time embedding noise (sampled uniformly per batch).")
    parser.add_argument("--epochs-supervised", type=int, default=None,
                         help="Override config['training']['epochs_supervised']. The default of 10 "
                              "is too few on harder benchmarks, where validation AUROC is still "
                              "climbing when training stops.")
    parser.add_argument("--structural-features", action="store_true",
                         help="Append per-node degree and edge-weight statistics to the node "
                              "features. GATConv's softmax-normalised attention is a convex "
                              "combination and so cannot count neighbours; this supplies the "
                              "local-density information its aggregator cannot derive.")
    parser.add_argument("--select-by", choices=["auroc", "loss"], default="auroc",
                         help="Criterion for saving phase2_best.pt. Validation loss and AUROC "
                              "diverge once the model starts overfitting, so loss-based "
                              "selection can save the worst checkpoint by the reported metric.")
    parser.add_argument("--scale-readout", action="store_true",
                         help="Concatenate sum, max and log-node-count readouts to the "
                              "attention pool. Attention pooling and GATConv are both "
                              "softmax-normalised, so the whole node-to-graph path is a "
                              "weighted average and cannot represent how MANY records sit in "
                              "a dense region -- which is what the attack is. A logistic "
                              "regression on nine magnitude features scores 0.995 on "
                              "synthetic_gap0125 where this model scores 0.889.")
    parser.add_argument("--readout", choices=["attention", "scale", "gated"], default=None,
                        help="Graph readout. 'attention' is the published baseline. 'scale' is "
                             "the concatenating variant --scale-readout selects, which triples "
                             "the GRU input width and so carries 98,688 more parameters than "
                             "the baseline; it scored 0.626 against the baseline's 0.889 on "
                             "synthetic_gap0125 over a full 40-epoch budget with early stopping "
                             "disabled, which is a capacity failure rather than a verdict on "
                             "magnitude information. 'gated' adds the same mean, max and count "
                             "readouts as residual branches behind zero-initialised scalar "
                             "gates: +771 parameters, and bit-identical to attention pooling at "
                             "initialisation, so the comparison isolates the information.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.epochs_supervised is not None:
        config["training"]["epochs_supervised"] = args.epochs_supervised
    set_seed(config.get("dataset", {}).get("seed", args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        config, args.data, device,
        holdout_attack_type=args.holdout_attack_type,
        noise_augment_prob=args.noise_augment_prob,
        noise_sigma_max=args.noise_sigma_max,
        structural_features=args.structural_features,
        select_by=args.select_by,
        scale_readout=args.scale_readout,
        readout=args.readout,
    )

    if not args.skip_phase1:
        trainer.train_phase1_contrastive(str(output_dir))
    trainer.train_phase2_supervised(str(output_dir))

    holdout_result = trainer.evaluate_holdout()
    if holdout_result is not None:
        print(f"\n=== Holdout attack-type generalization ({args.holdout_attack_type}) ===")
        print(f"  n_sequences={holdout_result['n_sequences']}  "
              f"graph_auroc={holdout_result['graph_auroc']:.4f}  "
              f"node_auroc={holdout_result['node_auroc']:.4f}")
        with open(output_dir / "holdout_attack_type_results.json", "w") as f:
            json.dump({"attack_type": args.holdout_attack_type, **holdout_result}, f, indent=2)

    print(f"\nTraining finished. Checkpoints and history saved in: {output_dir}")


if __name__ == "__main__":
    main()
