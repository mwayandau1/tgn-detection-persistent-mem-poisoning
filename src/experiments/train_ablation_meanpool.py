"""
Trains the attention-pooling ablation (TGNDetectorMeanPool) with the exact same
recipe used for experiments/run_current_arch (same config, seed, noise-augment
settings, Phase 1 + Phase 2), so the only difference between the two checkpoints
is AttentionPool vs global_mean_pool.
"""

import sys
import yaml
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.train import Trainer, set_seed
from src.experiments.ablation_meanpool_model import TGNDetectorMeanPool


def main():
    config_path = "configs/default.yaml"
    data_dir = "data/raw/llm_generated"
    output_dir = Path("experiments/run_ablation_meanpool")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    set_seed(config.get("dataset", {}).get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer = Trainer(
        config, data_dir, device,
        noise_augment_prob=0.5, noise_sigma_max=0.15,
        model_cls=TGNDetectorMeanPool,
    )

    trainer.train_phase1_contrastive(str(output_dir))
    trainer.train_phase2_supervised(str(output_dir))

    print(f"\nAblation training finished. Checkpoints saved in: {output_dir}")


if __name__ == "__main__":
    main()
