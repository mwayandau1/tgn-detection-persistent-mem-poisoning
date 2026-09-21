"""
Trains the attention-pooling ablation (TGNDetectorMeanPool) under the detector's exact recipe.

AttentionPool is replaced with global_mean_pool and NOTHING else changes -- node memory GRU,
GAT encoder, temporal GRU, both heads, loss, optimiser, budget and checkpoint selection are
all the detector's, because this routes through the same src.models.train.Trainer and only
swaps model_cls. That makes it a clean isolation of the readout, unlike the static-GNN
baseline, which removes pooling, node memory AND the temporal aggregator at once and so
cannot attribute its drop to any one of them.

This measures a claim the thesis argues in Sections 2.6 and 3.8 but has never tested on the
replacement benchmark: that mean pooling averages a small anomalous subgraph away almost by
construction, and that learned attention pooling is what recovers it. The original measurement
was made on data/raw/llm_generated, where a per-record content probe scores AUROC 1.000, so it
tells us nothing about pooling.

Window scores on the test split are written to window_scores_test.npz so the ablation is
aggregated and bootstrapped by sequence_eval.py through the identical code path as the
detector.

Usage:
    python -m src.experiments.train_ablation_meanpool \
        --data data/raw/ap_burst2k --output experiments/run_burst2k_meanpool \
        --structural-features --skip-phase1 --epochs-supervised 12 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories  # noqa: E402
from src.experiments.ablation_meanpool_model import TGNDetectorMeanPool  # noqa: E402
from src.models.train import Trainer, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/llm_generated")
    ap.add_argument("--output", default="experiments/run_ablation_meanpool")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs-supervised", type=int, default=None,
                    help="Must match the detector run being compared against.")
    ap.add_argument("--skip-phase1", action="store_true",
                    help="Matches the ap_burst2k runs, which skip contrastive pre-training.")
    ap.add_argument("--structural-features", action="store_true")
    ap.add_argument("--noise-augment-prob", type=float, default=0.5)
    ap.add_argument("--noise-sigma-max", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs_supervised:
        cfg["training"]["epochs_supervised"] = args.epochs_supervised
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mean-pool ablation | device={device}")
    print("  AttentionPool -> global_mean_pool; everything else identical to the detector")

    trainer = Trainer(
        cfg, args.data, device,
        noise_augment_prob=args.noise_augment_prob,
        noise_sigma_max=args.noise_sigma_max,
        model_cls=TGNDetectorMeanPool,
        structural_features=args.structural_features,
        select_by="auroc",
    )
    if not args.skip_phase1:
        trainer.train_phase1_contrastive(str(out))
    trainer.train_phase2_supervised(str(out))

    # --- window scores on test, for sequence_eval ------------------------------------
    # Reload the SELECTED checkpoint rather than scoring with final-epoch weights: the
    # checkpoint is what the selection rule chose, and scoring the other one reports a
    # different model from the one being claimed.
    if args.structural_features:
        cfg["model"]["node_meta_dim"] = 3 + 4
    model = TGNDetectorMeanPool(cfg).to(device)
    model.load_state_dict(torch.load(out / "phase2_best.pt", map_location=device,
                                     weights_only=True))
    model.eval()

    ds = MemoryTrajectoryDataset(
        args.data, split="test",
        temporal_window=cfg["training"]["temporal_window"],
        embedding_dim=cfg["simulation"]["embedding_dim"],
        structural_features=args.structural_features,
    )
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        collate_fn=collate_trajectories)
    model.reset_memory()
    scores = []
    with torch.no_grad():
        for batch in loader:
            graphs = [g.to(device) for g in batch["graphs"]]
            s, _ = model(graphs)
            scores.extend(s.cpu().numpy().tolist())

    scores = np.asarray(scores, dtype=float)
    wseq = np.array([w[0] for w in ds.windows])
    assert len(scores) == len(wseq), f"{len(scores)} scores for {len(wseq)} windows"
    np.savez(out / "window_scores_test.npz", score=scores, window_sequence_id=wseq)

    json.dump({"ablation": "mean_pool", "seed": args.seed,
               "structural_features": bool(args.structural_features),
               "epochs_supervised": cfg["training"]["epochs_supervised"],
               "n_windows": int(len(scores)),
               "parameters": int(sum(p.numel() for p in model.parameters()))},
              open(out / "ablation_meanpool.json", "w"), indent=2)

    print(f"\nWindow scores for {len(scores)} test windows -> {out / 'window_scores_test.npz'}")
    print(f"Evaluate with:\n"
          f"  python -m src.experiments.sequence_eval --data {args.data} \\\n"
          f"    --run-dir {out} --window-scores {out / 'window_scores_test.npz'} \\\n"
          f"    --label \"T-GNN mean-pool readout\"")


if __name__ == "__main__":
    main()
