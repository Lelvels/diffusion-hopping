#!/usr/bin/env python
"""
Check dataset statistics including raw and processed data counts.

This script provides detailed statistics about datasets:
- Raw data file counts
- Processed data counts per split (train/val/test)
- Dataset configuration information

Example usage:
    python check_data.py pdbbind_filtered
    python check_data.py crossdocked_filtered
    python check_data.py --dataset pdbbind_filtered_full
"""

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from config import DATA_ROOT
from _util import get_data_module_choices, get_datamodule


def count_raw_files(dataset_name: str) -> dict:
    """Count files in the raw dataset directory."""    
    raw_dir = DATA_ROOT / dataset_name / "raw"
    
    if not raw_dir.exists():
        return {
            "exists": False,
            "path": str(raw_dir),
            "files": 0,
            "directories": 0,
        }
    
    files = 0
    directories = 0
    
    for item in raw_dir.rglob("*"):
        if item.is_file():
            files += 1
        elif item.is_dir():
            directories += 1
    
    # Try to count complexes specifically
    complexes = 0
    if "pdbbind" in dataset_name:
        # PDBBind structure: raw/refined-set/XXXX/ directories
        refined_set = raw_dir / "refined-set"
        if refined_set.exists():
            complexes = len([d for d in refined_set.iterdir() if d.is_dir()])
    elif "crossdocked" in dataset_name:
        # CrossDocked structure: raw/crossdocked_pocket10/XXXX_YYYY_XXX/ directories
        crossdocked_dir = raw_dir / "crossdocked_pocket10"
        if crossdocked_dir.exists():
            complexes = len([d for d in crossdocked_dir.iterdir() if d.is_dir()])
    
    return {
        "exists": True,
        "path": str(raw_dir),
        "files": files,
        "directories": directories,
        "complexes": complexes if complexes > 0 else None,
    }


def count_processed_files(dataset_name: str) -> dict:
    """Count files in the processed dataset directory."""
    processed_root = DATA_ROOT / dataset_name / "processed"
    processed_dir = processed_root

    # Prefer the root layout when split files exist directly under processed/
    if processed_root.exists():
        root_has_splits = all(
            (processed_root / name).exists() for name in ("train.pt", "val.pt", "test.pt")
        )
        if not root_has_splits:
            # Otherwise, look for a subdirectory that contains the split files
            for candidate in sorted(p for p in processed_root.iterdir() if p.is_dir()):
                has_splits = all(
                    (candidate / name).exists() for name in ("train.pt", "val.pt", "test.pt")
                )
                if has_splits:
                    processed_dir = candidate
                    break
    
    if not processed_dir.exists():
        return {
            "exists": False,
            "path": str(processed_dir),
            "train": 0,
            "val": 0,
            "test": 0,
            "total": 0,
        }
    
    # Count split files
    train_file = processed_dir / "train.pt"
    val_file = processed_dir / "val.pt"
    test_file = processed_dir / "test.pt"
    
    processed_complexes_dir = processed_dir / "processed_complexes"
    processed_complexes = None
    if processed_complexes_dir.exists():
        processed_complexes = len([d for d in processed_complexes_dir.iterdir() if d.is_dir()])

    result = {
        "exists": True,
        "path": str(processed_dir),
        "train": 0,
        "val": 0,
        "test": 0,
        "total": 0,
        "processed_complexes": processed_complexes,
        "train_exists": train_file.exists(),
        "val_exists": val_file.exists(),
        "test_exists": test_file.exists(),
    }
    
    # Try to load and count actual data samples
    try:
        import torch
        
        if train_file.exists():
            train_data = torch.load(train_file, weights_only=False)
            result["train"] = len(train_data)
        
        if val_file.exists():
            val_data = torch.load(val_file, weights_only=False)
            result["val"] = len(val_data)
        
        if test_file.exists():
            test_data = torch.load(test_file, weights_only=False)
            result["test"] = len(test_data)
        
        result["total"] = result["train"] + result["val"] + result["test"]
    except Exception as e:
        result["error"] = str(e)
    
    return result


def get_dataset_info(dataset_name: str) -> dict:
    """Get dataset configuration information."""
    dataset_parts = dataset_name.split("_")
    
    info = {
        "name": dataset_name,
        "base": dataset_parts[0],
    }
    
    if len(dataset_parts) == 1:
        info["featurization"] = "c_alpha_only=True, cutoff=8.0, mode=residue"
        info["filter"] = "None"
    elif len(dataset_parts) == 2:
        if dataset_parts[1] == "filtered":
            info["featurization"] = "c_alpha_only=True, cutoff=8.0, mode=residue"
            info["filter"] = "QEDThresholdFilter(0.3)"
        elif dataset_parts[1] == "full":
            info["featurization"] = "c_alpha_only=False, cutoff=8.0, mode=residue"
            info["filter"] = "None"
    elif len(dataset_parts) == 3:
        if dataset_parts[1] == "filtered" and dataset_parts[2] == "full":
            info["featurization"] = "c_alpha_only=False, cutoff=8.0, mode=residue"
            info["filter"] = "QEDThresholdFilter(0.3)"
    
    return info


def print_statistics(dataset_name: str):
    """Print comprehensive dataset statistics."""
    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS: {dataset_name}")
    print(f"{'='*80}\n")
    
    # Dataset info
    info = get_dataset_info(dataset_name)
    print(f"Dataset Configuration:")
    print(f"  Base dataset: {info['base']}")
    print(f"  Featurization: {info.get('featurization', 'Unknown')}")
    print(f"  Filter: {info.get('filter', 'Unknown')}")
    print(f"  Data root: {DATA_ROOT}")
    print()
    
    # Raw data statistics
    print(f"{'─'*80}")
    print(f"RAW DATA")
    print(f"{'─'*80}")
    raw_stats = count_raw_files(dataset_name)
    
    if raw_stats["exists"]:
        print(f"  Path: {raw_stats['path']}")
        print(f"  Total files: {raw_stats['files']:,}")
        print(f"  Total directories: {raw_stats['directories']:,}")
        if raw_stats.get("complexes"):
            print(f"  Protein-ligand complexes: {raw_stats['complexes']:,}")
    else:
        print(f"  ⚠ Raw data directory not found: {raw_stats['path']}")
        print(f"  Please download and extract the dataset first.")
    print()
    
    # Processed data statistics
    print(f"{'─'*80}")
    print(f"PROCESSED DATA")
    print(f"{'─'*80}")
    processed_stats = count_processed_files(dataset_name)
    
    if processed_stats["exists"]:
        print(f"  Path: {processed_stats['path']}")
        print(f"\n  Split Statistics:")
        print(f"    Train split: {processed_stats['train']:,} samples {'✓' if processed_stats['train_exists'] else '✗'}")
        print(f"    Val split:   {processed_stats['val']:,} samples {'✓' if processed_stats['val_exists'] else '✗'}")
        print(f"    Test split:  {processed_stats['test']:,} samples {'✓' if processed_stats['test_exists'] else '✗'}")
        print(f"    ────────────────────────────")
        print(f"    Total:       {processed_stats['total']:,} samples")
        if processed_stats.get("processed_complexes") is not None:
            print(f"    Complexes:   {processed_stats['processed_complexes']:,} folders")
        
        if processed_stats.get("error"):
            print(f"\n  ⚠ Warning: {processed_stats['error']}")
    else:
        print(f"  ⚠ Processed data directory not found: {processed_stats['path']}")
        print(f"  Run 'python create_dataset.py {dataset_name}' to process the dataset.")
    
    print(f"\n{'='*80}\n")
    
    # Try to load actual datamodule for verification
    try:
        print(f"Verifying dataset can be loaded...")
        data_module = get_datamodule(dataset_name, batch_size=1)
        data_module.setup("fit")
        data_module.setup("test")
        
        train_size = len(data_module.train_dataset) if data_module.train_dataset else 0
        val_size = len(data_module.val_dataset) if data_module.val_dataset else 0
        test_size = len(data_module.test_dataset) if data_module.test_dataset else 0
        
        print(f"  ✓ Dataset loaded successfully")
        print(f"  Verified counts:")
        print(f"    Train: {train_size:,}")
        print(f"    Val:   {val_size:,}")
        print(f"    Test:  {test_size:,}")
        print(f"    Total: {train_size + val_size + test_size:,}")
        
    except Exception as e:
        print(f"  ⚠ Could not load dataset: {e}")
    
    print(f"\n{'='*80}\n")


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        prog="check_data.py",
        description="Check dataset statistics including raw and processed data",
        epilog="Example: python check_data.py pdbbind_filtered",
    )
    parser.add_argument(
        "dataset",
        type=str,
        nargs="?",
        default="pdbbind_filtered",
        choices=get_data_module_choices(),
        help="Dataset name to check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output statistics as JSON",
    )
    
    args = parser.parse_args()
    
    if args.json:
        # Output as JSON
        stats = {
            "dataset": args.dataset,
            "info": get_dataset_info(args.dataset),
            "raw": count_raw_files(args.dataset),
            "processed": count_processed_files(args.dataset),
        }
        print(json.dumps(stats, indent=2))
    else:
        # Print human-readable statistics
        print_statistics(args.dataset)


if __name__ == "__main__":
    main()
