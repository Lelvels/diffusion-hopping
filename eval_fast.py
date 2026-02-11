#!/usr/bin/env python
"""
Evaluation script for the diffusion-hopping model.

This script loads a model from a checkpoint file and evaluates it on a test dataset.
You can control generation and evaluation modes, batch size, and other parameters.

Example usage:
    python eval_diffhopp.py checkpoints/gvp_conditional.ckpt --limit_samples 500
    python eval_diffhopp.py path/to/my_model.ckpt --dataset pdbbind_filtered --batch_size 16
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv
import torch
import wandb
import os

from _util import get_datamodule
from diffusion_hopping.analysis.evaluate import Evaluator
from diffusion_hopping.model import DiffusionHoppingModel
from diffusion_hopping.util import disable_obabel_and_rdkit_logging

def generate_molecules(
    evaluator: Evaluator,
    output_path: Path,
    mode: str = "all",
    r: int = 10,
    j: int = 10,
    limit_samples: int = 500,
    molecules_per_pocket: int = 10,
    batch_size: int = 32,
):
    """Generate molecules for evaluation."""
    is_repainting_compatible = evaluator.is_model_repainting_compatible()
    
    print(f"\n{'='*60}")
    print(f"MOLECULE GENERATION - Limited to {limit_samples} samples")
    print(f"{'='*60}\n")
    
    if (
        mode == "ground_truth"
        or mode == "all"
        or (mode == "inpaint_generation" and is_repainting_compatible)
    ):
        print(f"[1/3] Generating ground truth molecules...")
        evaluator.use_ground_truth_molecules(limit_samples=limit_samples)
        evaluator.to_tensor(output_path / "molecules_ground_truth.pt")
        print(f"✓ Ground truth molecules saved")

    if mode == "ligand_generation" or mode == "all":
        print(f"\n[2/3] Generating ligand molecules...")
        print(f"  - Molecules per pocket: {molecules_per_pocket}")
        print(f"  - Batch size: {batch_size}")
        evaluator.generate_molecules(
            limit_samples=limit_samples,
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
        )
        evaluator.to_tensor(output_path / "molecules_ligand_generation.pt")
        print(f"✓ Ligand generation molecules saved")

    if mode == "inpaint_generation" or (mode == "all" and is_repainting_compatible):
        print(f"\n[3/3] Generating inpaint molecules (r={r}, j={j})...")
        evaluator.generate_molecules_inpainting(
            r=r,
            j=j,
            limit_samples=limit_samples,
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
        )
        evaluator.to_tensor(output_path / "molecules_inpaint_generation.pt")
        print(f"✓ Inpaint generation molecules saved")

def evaluate_molecules(evaluator, output_path, mode="all", scorer="autodock_gpu"):
    """Evaluate generated molecules."""
    is_repainting_compatible = evaluator.is_model_repainting_compatible()
    
    print(f"\n{'='*60}")
    print(f"MOLECULE EVALUATION")
    print(f"{'='*60}\n")
    
    output_str = f"Output path: {output_path}\n"
    output_str += f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n"
    
    if (
        mode == "ground_truth"
        or mode == "all"
        or (mode == "inpaint_generation" and is_repainting_compatible)
    ):
        print("[1/3] Evaluating ground truth molecules...")
        evaluator.from_tensor(output_path / "molecules_ground_truth.pt")
        evaluator.evaluate(apply_transform=False, scorer=scorer, output_format='sdf')
        evaluator.to_html(output_path / "results_ground_truth.html")
        evaluator.to_tensor(output_path / "results_ground_truth.pt")
        evaluator.print_summary_statistics()
        output_str += f"Ground truth results:\n{evaluator.get_summary_string()}\n\n"
        print("✓ Ground truth evaluation complete")

    if mode == "ligand_generation" or mode == "all":
        print("\n[2/3] Evaluating ligand generation molecules...")
        evaluator.from_tensor(output_path / "molecules_ligand_generation.pt")
        evaluator.evaluate(apply_transform=True, scorer=scorer, output_format='sdf')
        evaluator.to_html(output_path / "results_ligand_generation.html")
        evaluator.to_tensor(output_path / "results_ligand_generation.pt")
        evaluator.print_summary_statistics()
        output_str += f"Ligand generation results:\n{evaluator.get_summary_string()}\n\n"
        print("✓ Ligand generation evaluation complete")

    if mode == "inpaint_generation" or (mode == "all" and is_repainting_compatible):
        print("\n[3/3] Evaluating inpaint generation molecules...")
        evaluator.from_tensor(output_path / "molecules_inpaint_generation.pt")
        evaluator.evaluate(apply_transform=True, scorer=scorer, output_format='sdf')
        evaluator.to_html(output_path / "results_inpaint_generation.html")
        evaluator.to_tensor(output_path / "results_inpaint_generation.pt")
        evaluator.print_summary_statistics()
        output_str += (
            f"Inpaint generation results:\n{evaluator.get_summary_string()}\n"
        )
        print("✓ Inpaint generation evaluation complete")

    output_path.joinpath("summary.txt").write_text(output_str)
    print(f"\n✓ Summary saved to: {output_path / 'summary.txt'}")

def setup_model_and_data_module(checkpoint_path: Path, dataset_name: str, device="cpu"):
    """Load model and data module from local checkpoint file."""
    print(f"\n{'='*60}")
    print(f"MODEL AND DATA SETUP")
    print(f"{'='*60}\n")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Checkpoint size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\nLoading model to device: {device}")
    model = DiffusionHoppingModel.load_from_checkpoint(
        checkpoint_path, map_location=device, weights_only=False
    ).to(device)
    print(f"✓ Model loaded successfully")

    print(f"\nLoading dataset: {dataset_name}")
    data_module = get_datamodule(dataset_name, batch_size=32, shuffle=False)
    print("✓ Data module loaded successfully (shuffle=False for deterministic evaluation)")
    
    return model, data_module

def main():
    load_dotenv()  # Load environment variables from .env file
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    
    # Initialize Weights & Biases
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        print(f"✓ Logged in to Weights & Biases")
    else:
        print("WARNING: WANDB_API_KEY not set. W&B logging will be disabled.")
    
    parser = argparse.ArgumentParser(
        prog="eval_diffhopp.py",
        description="Evaluate ligand generation using a checkpoint file",
        epilog="Example: python eval_diffhopp.py checkpoints/gvp_conditional.ckpt --limit_samples 500",
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint file (e.g., checkpoints/gvp_conditional.ckpt)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate on",
        default="pdbbind_filtered",
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode to evaluate",
        choices=["ground_truth", "ligand_generation", "inpaint_generation", "all"],
        default="ligand_generation",
    )
    parser.add_argument(
        "--only_generation",
        action="store_true",
        help="Only generate molecules, do not evaluate them",
    )
    parser.add_argument(
        "--only_evaluation",
        action="store_true",
        help="Only evaluate molecules, do not generate them",
    )
    parser.add_argument(
        "--r",
        type=int,
        help="Number of resampling steps when using inpainting",
        default=10,
    )
    parser.add_argument(
        "--j",
        type=int,
        help="Jump length when using inpainting",
        default=10,
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        help="Limit the number of samples to evaluate",
        default=500,
    )
    parser.add_argument(
        "--molecules_per_pocket",
        type=int,
        help="Number of molecules to generate per pocket",
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for generation",
        default=32,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results (default: from EVALUATION_OUTPUT_DIR env or 'evaluation_local')",
        default=os.getenv("EVALUATION_OUTPUT_DIR", "evaluation_local"),
    )
    parser.add_argument(
        "--scorer",
        type=str,
        help="Scoring method: autodock_gpu",
        default="autodock_gpu",
        choices=["autodock_gpu"],
    )
    args = parser.parse_args()

    # Configuration
    mode = args.mode
    do_generation = not args.only_evaluation
    do_evaluation = not args.only_generation
    limit_samples = args.limit_samples
    molecules_per_pocket = args.molecules_per_pocket
    batch_size = args.batch_size

    # Setup paths
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint_name = checkpoint_path.stem  # filename without extension
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset
    output_path = Path(args.output_dir) / checkpoint_name / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Disable logging noise
    disable_obabel_and_rdkit_logging()

    # Print configuration
    print(f"\n{'='*60}")
    print(f"EVALUATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Mode: {mode}")
    print(f"Limit samples: {limit_samples}")
    print(f"Molecules per pocket: {molecules_per_pocket}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Output path: {output_path}")
    print(f"Do generation: {do_generation}")
    print(f"Do evaluation: {do_evaluation}")
    print(f"{'='*60}\n")

    # Initialize Weights & Biases project
    if WANDB_API_KEY and WANDB_PROJECT:
        try:
            # Get WANDB_DIR from environment or use output_path
            wandb_dir = os.getenv("WANDB_DIR", str(output_path))
            wandb.init(
                project=WANDB_PROJECT,
                dir=wandb_dir,
                config={
                    "checkpoint": checkpoint_name,
                    "dataset": dataset_name,
                    "mode": mode,
                    "limit_samples": limit_samples,
                    "molecules_per_pocket": molecules_per_pocket,
                    "batch_size": batch_size,
                    "scorer": args.scorer,
                    "device": device,
                },
                tags=["evaluation", checkpoint_name, args.scorer],
            )
            print(f"✓ W&B project initialized: {WANDB_PROJECT}")
            print(f"  Runs will be saved to: {wandb_dir}")
        except Exception as e:
            print(f"WARNING: Failed to initialize W&B project: {e}")
    else:
        if not WANDB_PROJECT:
            print("WARNING: WANDB_PROJECT not set. W&B logging will be disabled.")

    # Load model and data
    model, data_module = setup_model_and_data_module(
        checkpoint_path, dataset_name, device=device
    )

    # Create evaluator
    evaluator = Evaluator(output_path)
    evaluator.load_data_module(data_module)
    evaluator.load_model(model)

    # Run evaluation
    if do_generation:
        generate_molecules(
            evaluator,
            output_path,
            mode=mode,
            r=args.r,
            j=args.j,
            limit_samples=limit_samples,
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
        )
    
    if do_evaluation:
        evaluate_molecules(evaluator, output_path, mode=mode, scorer=args.scorer)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Finish the W&B run
    if WANDB_API_KEY and WANDB_PROJECT:
        wandb.finish()

if __name__ == "__main__":
    main()