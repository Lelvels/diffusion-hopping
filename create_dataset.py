import argparse
import os
from pathlib import Path

from _util import get_datamodule
from config import DATA_ROOT
from diffusion_hopping.util import disable_obabel_and_rdkit_logging

if __name__ == "__main__":
    disable_obabel_and_rdkit_logging()
    parser = argparse.ArgumentParser(
        prog="create_dataset.py",
        description="Create and preprocess dataset with given name",
        epilog="Example: python create_dataset.py pdbbind_filtered",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset_name", 
        type=str, 
        help="Name of the dataset (e.g., pdbbind_filtered, crossdocked_filtered)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"Override data root directory (default: {DATA_ROOT})"
    )
    args = parser.parse_args()
    
    # Override DATA_ROOT if specified
    if args.data_root:
        os.environ["DIFFUSION_HOPPING_DATA_ROOT"] = args.data_root
        # Re-import to get updated DATA_ROOT
        import importlib
        import config
        importlib.reload(config)
        from config import DATA_ROOT
    
    print("=" * 70)
    print(f"Dataset Creation: {args.dataset_name}")
    print("=" * 70)
    print(f"Data root directory: {DATA_ROOT}")
    print(f"Dataset path: {DATA_ROOT / args.dataset_name}")
    print()
    
    # Check if data root exists
    if not DATA_ROOT.exists():
        print(f"⚠ Warning: Data root directory does not exist: {DATA_ROOT}")
        print("Creating directory...")
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Get data module
    data_module = get_datamodule(args.dataset_name)
    
    print(f"Configuration:")
    print(f"  Pre-transform: {data_module.pre_transform}")
    print(f"  Pre-filter: {data_module.pre_filter}")
    print(f"  Batch size: {data_module.hparams.batch_size}")
    print()
    
    # Setup training/validation data
    print("=" * 70)
    print("Processing training and validation data...")
    print("=" * 70)
    try:
        data_module.setup("fit")
        train_dataset = data_module.train_dataloader().dataset
        val_dataset = data_module.val_dataloader().dataset
        print(f"✓ Training set: {len(train_dataset)} samples")
        print(f"✓ Validation set: {len(val_dataset)} samples")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print()
        print("Please ensure the raw data files are in the correct location:")
        print(f"  {DATA_ROOT / args.dataset_name / 'raw'}")
        print()
        print("Refer to README.md for instructions on obtaining the dataset.")
        exit(1)
    except Exception as e:
        print(f"✗ Error during training/validation setup: {e}")
        raise
    
    print()
    
    # Setup test data
    print("=" * 70)
    print("Processing test data...")
    print("=" * 70)
    try:
        data_module.setup("test")
        test_dataset = data_module.test_dataloader().dataset
        print(f"✓ Test set: {len(test_dataset)} samples")
    except Exception as e:
        print(f"✗ Error during test setup: {e}")
        raise
    
    print()
    print("=" * 70)
    print("Dataset Creation Complete!")
    print("=" * 70)
    print(f"Dataset: {args.dataset_name}")
    print(f"Location: {DATA_ROOT / args.dataset_name}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"  - Training: {len(train_dataset)}")
    print(f"  - Validation: {len(val_dataset)}")
    print(f"  - Test: {len(test_dataset)}")
    print()
    print("✓ Ready for training and evaluation!")
