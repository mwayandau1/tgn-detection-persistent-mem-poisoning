"""
Train the static-GNN baseline under the detector's exact recipe, and dump its window scores.

Everything that could differ between the baseline and the detector, other than the temporal
components themselves, is held fixed by routing through src.models.train.Trainer: the same
loss (alpha*BCE + beta*focal), the same optimiser and schedule, the same batch size, the same
embedding-noise augmentation, the same validation-AUROC checkpoint selection and the same
epoch budget. Only `model_cls` changes.

The earlier version of this baseline, in notebooks/05_baselines_comparison.ipynb, differed
from the detector in four ways at once -- it trained for 5 epochs instead of 12, took the
final weights rather than the best validation checkpoint, was built with node_meta_dim=3 while
the detector runs with 7, and pooled with global_mean_pool against the detector's attention
pooling. Three of those handicap the baseline, so a win for the detector would have proved
nothing. This script removes all four; `--readout mean` reinstates the fourth deliberately,
to keep continuity with the published table.

Window scores on the test split are written to window_scores_test.npz so the baseline can be
aggregated and bootstrapped by sequence_eval.py through the identical code path as the
detector, rather than a second implementation of the same statistics.

Usage:
    python -m src.experiments.train_static_baseline \
        --data data/raw/ap_burst2k --output experiments/run_burst2k_static_attn \
        --readout attention --epochs-supervised 12 --seed 42
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
from src.experiments.static_gnn_model import (  # noqa: E402
    StaticGNNDetectorAttention,
    StaticGNNDetectorMeanPool,
)
from src.models.train import Trainer, set_seed  # noqa: E402

MODELS = {"attention": StaticGNNDetectorAttention, "mean": StaticGNNDetectorMeanPool}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--readout", choices=sorted(MODELS), default="attention")
    ap.add_argument("--epochs-supervised", type=int, default=12,
                    help="Must match the detector run being compared against.")
    ap.add_argument("--noise-augment-prob", type=float, default=0.5)
    ap.add_argument("--noise-sigma-max", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--structural-features", action="store_true", default=True)
    ap.add_argument("--no-structural-features", dest="structural_features",
                    action="store_false")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["training"]["epochs_supervised"] = args.epochs_supervised
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Static-GNN baseline | readout={args.readout} | device={device}")
    print("  temporal components removed: NodeMemoryModule and TemporalAggregator")

    trainer = Trainer(
        cfg, args.data, device,
        noise_augment_prob=args.noise_augment_prob,
        noise_sigma_max=args.noise_sigma_max,
        model_cls=MODELS[args.readout],
        structural_features=args.structural_features,
        select_by="auroc",
    )
    trainer.train_phase2_supervised(str(out))

    # --- window scores on test, for sequence_eval -------------------------------------
    # Reload the selected checkpoint rather than scoring with the final-epoch weights: the
    # checkpoint is what the selection rule chose, and scoring the wrong one is how an
    # ablation quietly reports a different model from the one it claims.
    model = MODELS[args.readout](cfg).to(device)
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

    meta = {"readout": args.readout, "seed": args.seed,
            "epochs_supervised": args.epochs_supervised,
            "structural_features": bool(args.structural_features),
            "n_windows": int(len(scores)),
            "parameters": int(sum(p.numel() for p in model.parameters()))}
    json.dump(meta, open(out / "static_baseline.json", "w"), indent=2)
    print(f"\nWindow scores for {len(scores)} test windows -> {out / 'window_scores_test.npz'}")
    print(f"Evaluate at sequence level with:\n"
          f"  python -m src.experiments.sequence_eval --data {args.data} \\\n"
          f"    --run-dir {out} --window-scores {out / 'window_scores_test.npz'} \\\n"
          f"    --label \"static GNN ({args.readout})\"")


if __name__ == "__main__":
    main()
