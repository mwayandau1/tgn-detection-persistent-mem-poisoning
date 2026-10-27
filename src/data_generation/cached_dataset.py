"""
A per-sequence cache of the snapshot corpus, and a drop-in replacement for
MemoryTrajectoryDataset that reads from it.

Why
---
MemoryTrajectoryDataset loads ten compressed .npz files and recomputes the structural node
features for every window it returns. Windows overlap by nine snapshots, so each snapshot is
decompressed and re-featurised up to ten times per epoch, and on the project's laptop CPU that
-- not the GPU -- sets the pace of training and evaluation (scoring the test split at batch
size 1 ran the GPU at under 30 per cent).

Snapshots of one sequence are nested: snapshot t holds records 0 .. 10(t+1)-1, and a record's
384 embedding columns never change. So a sequence is stored once as
    emb      (n_final, 384)          float32
    side_t   (n_t, 3 or 7)           float32   metadata (+ structural) columns of snapshot t
    edges_t, weights_t                          exactly as persisted for snapshot t
    labels   (n_final,)              node labels of the final snapshot
which is about 0.4 MB per sequence instead of twenty compressed files.

Exactness
---------
Every float is cast to float32 from the same float64 value the original loader casts, edge
arrays are copied verbatim (order included, since scatter order changes floating-point sums),
so `CachedTrajectoryDataset` returns tensors bit-identical to MemoryTrajectoryDataset's.
`verify()` checks that on a sample of windows and is run by `build` before the cache is used.

Usage:
    python -m src.data_generation.cached_dataset build --data data/raw/ap_burst2k
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data_generation.dataset import MemoryTrajectoryDataset  # noqa: E402

CACHE_DIRNAME = "_seqcache"


def cache_dir(data_dir: Path, structural: bool) -> Path:
    return Path(data_dir) / CACHE_DIRNAME / ("struct" if structural else "plain")


def build_sequence(seq_dir: Path, emb_dim: int, structural: bool) -> dict:
    paths = sorted(seq_dir.glob("snapshot_*.npz"))
    side, eidx, ew, eoff, n_nodes, glabel, embs = [], [], [], [0], [], [], []
    emb = labels = None
    for p in paths:
        with np.load(p) as z:
            f = z["node_features"]
            ei, w = z["edge_index"], z["edge_weights"]
            nl, gl = z["node_labels"], z["graph_label"]
        if structural:
            f = np.concatenate(
                [f, MemoryTrajectoryDataset._structural_features(f.shape[0], ei, w)], axis=1)
        n = f.shape[0]
        embs.append(f[:, :emb_dim].astype(np.float32))
        if emb is None or n > emb.shape[0]:
            emb = embs[-1]
            labels = np.asarray(nl).astype(np.int64)
        side.append(f[:, emb_dim:].astype(np.float32))
        eidx.append(np.asarray(ei, dtype=np.int64).reshape(2, -1))
        ew.append(np.asarray(w, dtype=np.float32).reshape(-1))
        eoff.append(eoff[-1] + eidx[-1].shape[1])
        n_nodes.append(n)
        glabel.append(int(np.asarray(gl).reshape(-1)[0]))
    # A record's embedding columns are written once by the generator and never change, so
    # the final snapshot's copy serves every earlier one. Checked rather than assumed.
    for e in embs:
        assert np.array_equal(e, emb[:e.shape[0]]), f"{seq_dir.name}: embeddings not nested"
    return {"emb": emb, "labels": labels,
            "side": np.concatenate(side, axis=0), "side_off": np.cumsum([0] + n_nodes),
            "edge_index": np.concatenate(eidx, axis=1), "edge_weight": np.concatenate(ew),
            "edge_off": np.array(eoff), "n_nodes": np.array(n_nodes),
            "graph_label": np.array(glabel)}


class SequenceStore:
    """Loads cached sequences on demand and keeps the most recent ones in RAM."""

    def __init__(self, data_dir: Path, structural: bool, keep: Optional[int] = 64):
        self.dir = cache_dir(data_dir, structural)
        # The structural columns are appended after the three metadata columns, so the
        # plain view is an exact column slice of the structural cache.
        self.side_cols = None
        if not structural and not self.dir.exists() and cache_dir(data_dir, True).exists():
            self.dir, self.side_cols = cache_dir(data_dir, True), 3
        if not self.dir.exists():
            raise FileNotFoundError(f"{self.dir} missing; run "
                                    f"`python -m src.data_generation.cached_dataset build`")
        self.keep, self.mem = keep, {}

    def get(self, seq_id: str) -> dict:
        if seq_id not in self.mem:
            if self.keep is not None and len(self.mem) >= self.keep:
                self.mem.pop(next(iter(self.mem)))
            with np.load(self.dir / f"{seq_id}.npz") as z:
                self.mem[seq_id] = {k: z[k] for k in z.files}
        return self.mem[seq_id]

    def snapshot(self, seq_id: str, t: int) -> Data:
        s = self.get(seq_id)
        n = int(s["n_nodes"][t])
        side = s["side"][s["side_off"][t]:s["side_off"][t + 1], :self.side_cols]
        x = np.concatenate([s["emb"][:n], side], axis=1)
        e0, e1 = s["edge_off"][t], s["edge_off"][t + 1]
        w = torch.from_numpy(s["edge_weight"][e0:e1].copy())
        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(s["edge_index"][:, e0:e1].copy()),
            edge_attr=w.unsqueeze(-1) if w.numel() > 0 else torch.zeros((0, 1)),
            y=torch.tensor(int(s["graph_label"][t])),
            node_labels=torch.from_numpy(s["labels"][:n].copy()),
            node_ids=torch.arange(n, dtype=torch.long),
        )


class CachedTrajectoryDataset(Dataset):
    """Same windows, same order, same tensors as MemoryTrajectoryDataset."""

    def __init__(self, data_dir: str, split: str = "train", temporal_window: int = 10,
                 embedding_dim: int = 384, sequence_ids: Optional[List[str]] = None,
                 structural_features: bool = False, keep: Optional[int] = 64):
        self.data_dir = Path(data_dir)
        self.temporal_window = temporal_window
        self.structural_features = structural_features
        self.metadata = {m["sequence_id"]: m
                         for m in json.load(open(self.data_dir / "metadata.json"))}
        if sequence_ids is None:
            sequence_ids = json.load(open(self.data_dir / "splits.json"))[split]
        self.sequence_ids = list(sequence_ids)
        self.store = SequenceStore(self.data_dir, structural_features, keep=keep)
        self.windows = []
        for sid in self.sequence_ids:
            with np.load(self.store.dir / f"{sid}.npz") as z:
                n_snap = len(z["n_nodes"])
            for start in range(max(1, n_snap - temporal_window + 1)):
                self.windows.append((sid, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        sid, start = self.windows[idx]
        graphs = [self.store.snapshot(sid, t) for t in range(start, start + self.temporal_window)]
        return {"graphs": graphs, "graph_label": graphs[-1].y.item(),
                "node_labels": graphs[-1].node_labels, "sequence_id": sid,
                "window_start": start}


def verify(data_dir: Path, structural: bool, n_windows: int = 40, seed: int = 0) -> int:
    """Compare against MemoryTrajectoryDataset tensor by tensor; returns windows checked."""
    splits = json.load(open(data_dir / "splits.json"))
    ids = splits["train"][:5] + splits["val"][:5] + splits["test"][:5]
    ref = MemoryTrajectoryDataset(str(data_dir), sequence_ids=ids,
                                  structural_features=structural)
    new = CachedTrajectoryDataset(str(data_dir), sequence_ids=ids,
                                  structural_features=structural)
    assert ref.windows == new.windows
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ref), size=min(n_windows, len(ref)), replace=False)
    for i in pick:
        a, b = ref[int(i)], new[int(i)]
        assert a["graph_label"] == b["graph_label"]
        for ga, gb in zip(a["graphs"], b["graphs"]):
            for key in ("x", "edge_index", "edge_attr", "node_labels", "node_ids"):
                ta, tb = getattr(ga, key), getattr(gb, key)
                assert ta.dtype == tb.dtype and torch.equal(ta, tb), (i, key)
            assert int(ga.y) == int(gb.y)
    return len(pick)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "verify"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--emb-dim", type=int, default=384)
    ap.add_argument("--no-structural-features", dest="structural", action="store_false",
                    default=True)
    args = ap.parse_args()
    data = Path(args.data)
    out = cache_dir(data, args.structural)

    if args.command == "build":
        out.mkdir(parents=True, exist_ok=True)
        meta = json.load(open(data / "metadata.json"))
        for k, m in enumerate(meta):
            dst = out / f"{m['sequence_id']}.npz"
            if not dst.exists():
                np.savez(dst, **build_sequence(data / m["sequence_id"], args.emb_dim,
                                               args.structural))
            if k % 200 == 0:
                print(f"  {k}/{len(meta)}", flush=True)
    n = verify(data, args.structural)
    print(f"verified {n} windows bit-identical to MemoryTrajectoryDataset ({out})")


if __name__ == "__main__":
    main()
