import json
import numpy as np
import torch
from pathlib import Path
from src.data_generation.dataset import MemoryTrajectoryDataset, collate_trajectories

def verify_dataset(data_dir: str):
    data_path = Path(data_dir)
    print("=" * 60)
    print(f"DATASET VERIFICATION REPORT FOR: {data_dir}")
    print("=" * 60)

    # 1. Metadata check
    meta_file = data_path / "metadata.json"
    splits_file = data_path / "splits.json"
    if not meta_file.exists() or not splits_file.exists():
        print("❌ ERROR: Missing metadata.json or splits.json!")
        return

    with open(meta_file) as f:
        meta = json.load(f)
    with open(splits_file) as f:
        splits = json.load(f)

    print(f"[OK] Total sequences generated: {len(meta)}")
    print(f"[OK] Split counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    clean_count = sum(1 for m in meta if m['label'] == 'clean')
    poisoned_count = sum(1 for m in meta if m['label'] == 'poisoned')
    print(f"   - Clean sequences: {clean_count}")
    print(f"   - Poisoned sequences: {poisoned_count}")

    # Breakdown of attack types
    attack_types = {}
    for m in meta:
        at = m.get('attack_type')
        if at:
            attack_types[at] = attack_types.get(at, 0) + 1
    if attack_types:
        print("   - Attack type breakdown:")
        for at, cnt in attack_types.items():
            print(f"       * {at}: {cnt}")

    # 2. PyTorch Dataset & Snapshots Verification
    print("\n" + "-" * 60)
    print("PyTorch Trajectory Loader & Tensor Validation")
    print("-" * 60)

    for split_name in ['train', 'val', 'test']:
        if len(splits[split_name]) == 0:
            print(f"[INFO] Split '{split_name}': empty (no sequences assigned)")
            continue

        dataset = MemoryTrajectoryDataset(data_dir, split=split_name, temporal_window=10)
        print(f"[OK] Split '{split_name}': successfully initialized ({len(dataset)} sliding temporal windows)")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   [Sample Window 0]")
            print(f"   - Sequence ID: {sample['sequence_id']}")
            print(f"   - Window start snapshot index: {sample['window_start']}")
            print(f"   - Window trajectory length T: {len(sample['graphs'])}")
            print(f"   - Graph Label (Trajectory Poisoned?): {sample['graph_label']}")

            # Check snapshot tensors
            g0 = sample['graphs'][-1]  # Check last graph in window
            print(f"   - Node features shape: {tuple(g0.x.shape)} (Expected: [N, 387])")
            print(f"   - Edge index shape: {tuple(g0.edge_index.shape)} (Expected: [2, E])")
            print(f"   - Edge attr shape: {tuple(g0.edge_attr.shape)} (Expected: [E, 1])")
            print(f"   - Node labels shape: {tuple(g0.node_labels.shape)} (Expected: [N])")

            # Check node labels
            num_poisoned_nodes = (g0.node_labels == 1).sum().item()
            print(f"   - Poisoned nodes at final snapshot: {num_poisoned_nodes} / {g0.x.shape[0]}")

            # Verify NaN/Inf sanity
            assert not torch.isnan(g0.x).any(), "NaN detected in node features!"
            assert not torch.isinf(g0.x).any(), "Inf detected in node features!"
            print("   - Feature Tensor Sanity: [OK] NO NaN or Inf values detected")

    print("=" * 60)
    print("VERIFICATION SUMMARY: DATASET IS VALID & READY FOR THESIS EXPERIMENTS!")
    print("=" * 60)

if __name__ == "__main__":
    verify_dataset("data/raw/llm_generated_test")
