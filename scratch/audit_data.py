import json
from pathlib import Path

def audit_data():
    raw_dir = Path("data/raw")
    print("=" * 60)
    print("COMPREHENSIVE DATASET AUDIT REPORT")
    print("=" * 60)

    if not raw_dir.exists():
        print("ERROR: data/raw directory does not exist!")
        return

    for folder in raw_dir.iterdir():
        if folder.is_dir():
            print(f"\n[Folder: {folder.name}]")
            clean_dirs = sorted([d.name for d in folder.glob("clean_*") if d.is_dir()])
            poisoned_dirs = sorted([d.name for d in folder.glob("poisoned_*") if d.is_dir()])
            
            print(f"  * Clean sequences count: {len(clean_dirs)}")
            if clean_dirs:
                print(f"    - First: {clean_dirs[0]}, Last: {clean_dirs[-1]}")
            
            print(f"  * Poisoned sequences count: {len(poisoned_dirs)}")
            if poisoned_dirs:
                attacks = {}
                for pd in poisoned_dirs:
                    atype = pd.replace("poisoned_", "").rsplit("_", 1)[0]
                    attacks[atype] = attacks.get(atype, 0) + 1
                for atype, cnt in attacks.items():
                    print(f"    - {atype}: {cnt} sequences")
                    
            has_meta = (folder / "metadata.json").exists()
            has_splits = (folder / "splits.json").exists()
            print(f"  * metadata.json: {'PRESENT' if has_meta else 'MISSING'}")
            print(f"  * splits.json:   {'PRESENT' if has_splits else 'MISSING'}")

            # Check snapshot counts inside first 3 clean and poisoned dirs
            sample_dirs = (clean_dirs[:2] + poisoned_dirs[:2])
            for sd in sample_dirs:
                snaps = len(list((folder / sd).glob("snapshot_*.npz")))
                print(f"    - Sequence '{sd}': {snaps} snapshots")

    print("=" * 60)

if __name__ == "__main__":
    audit_data()
