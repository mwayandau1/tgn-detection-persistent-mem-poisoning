"""
Inference cost of a trained checkpoint on real benchmark windows (Table 4.11).
=============================================================================

Replaces the earlier compute_profile.py, whose source was deleted and whose only output
(experiments/computational_profile.json, 25 August) profiled a 667,331-parameter model on the
superseded LLM benchmark. This one profiles the checkpoint it is given, on windows sampled from
the benchmark it is given, and records both in the output so the provenance travels with the
numbers.

Latency is wall-clock time for one forward pass over one batch of windows, synchronised on
the GPU, after a warm-up, over a seeded random sample of test windows. Batches use
per_window memory slots, so batch size 4 computes what batch size 1 computes.

Usage:
    python -m src.experiments.compute_profile --data data/raw/ap_burst2k \
        --run-dir experiments/run_burst2k_base --output experiments/audit/compute_profile.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.cached_dataset import CachedTrajectoryDataset  # noqa: E402
from src.experiments.score_windows import build_model  # noqa: E402


def stats(ms):
    ms = np.asarray(ms)
    return {"mean_ms": float(ms.mean()), "std_ms": float(ms.std()),
            "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "n": int(ms.size)}


@torch.no_grad()
def profile(model, ds, idx, device, batch_size, T, warmup=10):
    model.eval()
    windows = [ds[int(i)]["graphs"] for i in idx]
    times = []
    for b0 in range(0, len(windows) - batch_size + 1, batch_size):
        chunk = windows[b0:b0 + batch_size]
        graphs = [Batch.from_data_list([w[t] for w in chunk]).to(device) for t in range(T)]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(graphs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = times[warmup // batch_size:]
    out = stats(times)
    out["throughput_windows_per_sec"] = float(1000.0 * batch_size / out["mean_ms"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="phase2_best.pt")
    ap.add_argument("--model", default="tgn")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-windows", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["model"]["node_meta_dim"] = 7
    cfg["model"]["readout"] = "attention"
    cfg["model"]["memory_slots"] = "per_window"
    T = cfg["training"]["temporal_window"]
    ckpt = Path(args.run_dir) / args.checkpoint
    ds = CachedTrajectoryDataset(args.data, split="test", temporal_window=T,
                                 structural_features=True)
    idx = np.random.default_rng(args.seed).choice(len(ds), size=args.n_windows, replace=False)

    result = {"checkpoint": str(ckpt), "checkpoint_bytes": ckpt.stat().st_size,
              "checkpoint_mib": ckpt.stat().st_size / 2 ** 20, "data": args.data,
              "n_windows": args.n_windows, "torch": torch.__version__}
    for dev_name in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        device = torch.device(dev_name)
        model = build_model(args.model, json.loads(json.dumps(cfg))).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        result["parameters"] = int(sum(p.numel() for p in model.parameters()))
        result["parameters_by_submodule"] = {n: int(sum(p.numel() for p in m.parameters()))
                                             for n, m in model.named_children()}
        block = {}
        for bs in (1, 4):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            block[f"batch_size_{bs}"] = profile(model, ds, idx, device, bs, T)
            if device.type == "cuda":
                block[f"batch_size_{bs}"]["peak_gpu_memory_mib"] = \
                    torch.cuda.max_memory_allocated() / 2 ** 20
            print(dev_name, bs, json.dumps(block[f"batch_size_{bs}"]), flush=True)
        result[dev_name] = block
        if device.type == "cuda":
            result["gpu"] = torch.cuda.get_device_name(0)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
