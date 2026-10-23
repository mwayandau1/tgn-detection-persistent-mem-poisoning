"""
Train any detector variant under the detector's recipe, then score it at batch size 1.
=====================================================================================

One entry point for the multi-seed study (E2) and the component ablations (E3), so every
configuration is trained by the same Trainer -- same loss, optimiser, schedule, augmentation,
epoch budget and validation-AUROC checkpoint rule -- and differs only in `--model` and
`--seed`. train_static_baseline.py and train_ablation_meanpool.py remain as the record of how
the run_burst2k_* checkpoints were produced; this does not change what they do.

After training, the selected checkpoint is scored on the validation and test splits at batch
size 1 (see score_windows.py for why the batch size matters), and the per-window scores are
written beside the checkpoint as window_scores_{val,test}_b1.npz.

Models:
  tgn            the detector (attention readout)
  meanpool       attention pooling replaced by mean pooling
  nostruct       the detector without the four structural node columns
  static_attn    final snapshot only, attention readout          (306,243 parameters)
  static_mean    final snapshot only, mean readout
  static_wide    static_attn widened to the detector's parameter count
  no_memory      temporal aggregator kept, per-node memory removed
  no_aggregator  per-node memory kept, temporal aggregator removed

Usage:
    python -m src.experiments.train_variant --data data/raw/ap_burst2k --model tgn \
        --seed 1 --output experiments/seeds/tgn_s1 --memory-slots per_window --cached
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.train import Trainer, set_seed, write_run_config  # noqa: E402
from src.experiments.score_windows import build_model, score_split  # noqa: E402

KINDS = ["tgn", "meanpool", "nostruct", "static_attn", "static_mean", "static_wide",
         "no_memory", "no_aggregator"]


def model_class(kind: str):
    if kind in ("tgn", "nostruct"):
        from src.models.tgn_detector import TGNDetector
        return TGNDetector
    if kind == "meanpool":
        from src.experiments.ablation_meanpool_model import TGNDetectorMeanPool
        return TGNDetectorMeanPool
    if kind in ("static_attn", "static_mean"):
        from src.experiments.static_gnn_model import (StaticGNNDetectorAttention,
                                                      StaticGNNDetectorMeanPool)
        return StaticGNNDetectorAttention if kind == "static_attn" else StaticGNNDetectorMeanPool
    from src.experiments.component_ablation_models import MODEL_KINDS
    return MODEL_KINDS[kind]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", choices=KINDS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs-supervised", type=int, default=12)
    ap.add_argument("--noise-augment-prob", type=float, default=0.5)
    ap.add_argument("--noise-sigma-max", type=float, default=0.15)
    ap.add_argument("--memory-slots", choices=["shared", "per_window"], default="per_window")
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--skip-scoring", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "window_scores_test_b1.npz").exists():
        print(f"{out} already complete; skipping")
        return

    cfg = yaml.safe_load(open(args.config))
    cfg["training"]["epochs_supervised"] = args.epochs_supervised
    cfg["model"]["memory_slots"] = args.memory_slots
    cfg["model"]["readout"] = "attention"
    structural = args.model != "nostruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    write_run_config(out, args, cfg, device)
    dataset_cls = None
    if args.cached:
        from src.data_generation.cached_dataset import CachedTrajectoryDataset
        dataset_cls = CachedTrajectoryDataset
    else:
        from src.data_generation.dataset import MemoryTrajectoryDataset
        dataset_cls = MemoryTrajectoryDataset

    t0 = time.time()
    trainer = Trainer(cfg, args.data, device,
                      noise_augment_prob=args.noise_augment_prob,
                      noise_sigma_max=args.noise_sigma_max,
                      model_cls=model_class(args.model),
                      structural_features=structural, select_by="auroc",
                      dataset_cls=dataset_cls)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    trainer.train_phase2_supervised(str(out))
    train_secs = time.time() - t0

    record = {"model": args.model, "seed": args.seed, "parameters": int(n_params),
              "memory_slots": args.memory_slots, "train_seconds": round(train_secs, 1),
              "static_wide_hidden": cfg["model"].get("static_wide_hidden")}
    if not args.skip_scoring:
        scfg = json.loads(json.dumps(cfg))
        model = build_model(args.model if args.model not in ("tgn", "nostruct") else "tgn",
                            scfg).to(device)
        model.load_state_dict(torch.load(out / "phase2_best.pt", map_location=device,
                                         weights_only=True))
        T = cfg["training"]["temporal_window"]
        for split in ("val", "test"):
            ds = dataset_cls(args.data, split=split, temporal_window=T,
                             embedding_dim=cfg["simulation"]["embedding_dim"],
                             structural_features=structural)
            res, _, secs = score_split(model, ds, device, 1, T, node_scores=True, log_every=0)
            np.savez(out / f"window_scores_{split}_b1.npz", **res, batch_size=1,
                     checkpoint="phase2_best.pt", model=args.model)
            from sklearn.metrics import roc_auc_score
            record[f"{split}_window_auroc_b1"] = float(roc_auc_score(res["y"], res["score"]))
            record[f"{split}_score_seconds"] = round(secs, 1)
            print(f"  {split}: window AUROC {record[f'{split}_window_auroc_b1']:.4f} "
                  f"({secs:.0f}s)")
    json.dump(record, open(out / "variant.json", "w"), indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
