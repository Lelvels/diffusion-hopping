#!/usr/bin/env python
"""
Evaluation script for W&B runs and author checkpoints.

This script evaluates models from:
1. W&B run directories (e.g., wandb/checkpoints/peachy-firebrand-37)
2. Author checkpoint files (e.g., author_checkpoints/gvp_conditional.ckpt)

It calculates comprehensive metrics including molecular properties, validity,
novelty, diversity, and docking scores.

Expected metrics:
- Train/Val Loss (computed via model inference)
- Novelty, Validity, Connectivity
- Lipinski, LogP, QED, SAScore
- Diversity (Tanimoto similarity-based)
- Vina/AutoDock-GPU scores

Example usage:
    # W&B run:
    python eval_wandb_run.py /path/to/wandb/checkpoints/peachy-firebrand-37 --limit_samples 500
    
    # Author checkpoint:
    python eval_wandb_run.py /path/to/author_checkpoints/gvp_conditional.ckpt --limit_samples 500
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from _util import get_datamodule
from diffusion_hopping.analysis.evaluate import Evaluator
from diffusion_hopping.analysis.metrics import (
    MolecularConnectivity,
    MolecularLipinski,
    MolecularLogP,
    MolecularNovelty,
    MolecularQEDValue,
    MolecularSAScore,
    MolecularValidity,
)
from diffusion_hopping.model import DiffusionHoppingModel
from diffusion_hopping.util import disable_obabel_and_rdkit_logging


def calculate_model_losses(model, data_module, device="cpu"):
    """Calculate train and validation losses by running model inference on datasets."""
    print(f"\nCalculating train/val losses via model inference...")
    
    model.eval()
    losses = {}
    
    # Setup data module for both train and val
    data_module.setup("fit")
    
    # Calculate train loss
    try:
        train_loader = data_module.train_dataloader()
        train_losses = []
        
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                result = model.validation_step(batch, 0)
                
                if isinstance(result, dict) and 'loss' in result:
                    train_losses.append(result['loss'].item())
                elif torch.is_tensor(result):
                    train_losses.append(result.item())
        
        if train_losses:
            losses['Train Loss'] = float(np.mean(train_losses))
            print(f"  ✓ Train loss: {losses['Train Loss']:.4f} (computed from {len(train_losses)} batches)")
    except Exception as e:
        print(f"  ⚠ Could not compute train loss: {e}")
    
    # Calculate validation loss
    try:
        val_loader = data_module.val_dataloader()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                result = model.validation_step(batch, 0)
                
                if isinstance(result, dict) and 'loss' in result:
                    val_losses.append(result['loss'].item())
                elif torch.is_tensor(result):
                    val_losses.append(result.item())
        
        if val_losses:
            losses['Val Loss'] = float(np.mean(val_losses))
            print(f"  ✓ Val loss: {losses['Val Loss']:.4f} (computed from {len(val_losses)} batches)")
    except Exception as e:
        print(f"  ⚠ Could not compute val loss: {e}")
    
    return losses


def calculate_diversity(molecules: List[Chem.Mol]) -> float:
    """Calculate molecular diversity using Tanimoto similarity of Morgan fingerprints."""
    if len(molecules) < 2:
        return 0.0
    
    # Generate Morgan fingerprints
    fps = []
    for mol in molecules:
        if mol is not None:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
            except Exception:
                continue
    
    if len(fps) < 2:
        return 0.0
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    # Diversity = 1 - average similarity
    avg_similarity = np.mean(similarities)
    diversity = 1.0 - avg_similarity
    
    return diversity


def load_wandb_run_config(run_dir: Path) -> dict:
    """Load W&B run configuration from wandb directory."""
    config = {}
    
    # Try to load from wandb-metadata.json
    metadata_file = run_dir / "wandb-metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            config.update(metadata.get('config', {}))
    
    # Try to load from config.yaml
    config_file = run_dir / "files" / "config.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
    
    return config


def setup_model_and_data_module(run_dir: Path, dataset_name: str, device="cpu"):
    """Load model and data module from W&B run directory."""
    print(f"\n{'='*60}")
    print(f"MODEL AND DATA SETUP FROM WANDB RUN")
    print(f"{'='*60}\n")
    
    # Find checkpoint file
    checkpoint_path = run_dir / "checkpoints" / "best_model.ckpt"
    if not checkpoint_path.exists():
        # Try alternative locations
        alt_paths = [
            run_dir / "best_model.ckpt",
            run_dir / "files" / "best_model.ckpt",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                checkpoint_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"Checkpoint 'best_model.ckpt' not found in {run_dir}/checkpoints/ "
                f"or alternative locations"
            )
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Checkpoint size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load W&B configuration
    config = load_wandb_run_config(run_dir)
    if config:
        print(f"\n✓ Loaded W&B configuration:")
        for key, value in list(config.items())[:10]:  # Show first 10 items
            print(f"  {key}: {value}")
        if len(config) > 10:
            print(f"  ... and {len(config) - 10} more")
    
    print(f"\nLoading model to device: {device}")
    model = DiffusionHoppingModel.load_from_checkpoint(
        checkpoint_path, map_location=device, weights_only=False
    ).to(device)
    print(f"✓ Model loaded successfully")

    print(f"\nLoading dataset: {dataset_name}")
    data_module = get_datamodule(dataset_name, batch_size=32, shuffle=False)
    print(f"✓ Data module loaded successfully (shuffle=False for deterministic evaluation)")
    
    return model, data_module, config

def evaluate_with_metrics(
    evaluator: Evaluator,
    output_path: Path,
    run_dir: Path,
    model,
    device: str = "cpu",
    limit_samples: int = 500,
    molecules_per_pocket: int = 10,
    batch_size: int = 32,
    scorer: str = "autodock_gpu"
):
    """Generate molecules and calculate comprehensive metrics."""
    
    print(f"\n{'='*60}")
    print(f"MOLECULAR GENERATION AND EVALUATION")
    print(f"{'='*60}\n")
    
    # Check if molecules already exist
    molecules_file = output_path / "molecules_generated.pt"
    results_file = output_path / "results.pt"
    
    if results_file.exists():
        # Skip both generation and evaluation - load final results
        print(f"✓ Found existing evaluation results: {results_file}")
        print(f"  Skipping generation and evaluation steps...")
        evaluator.from_tensor(results_file)
        print(f"✓ Loaded {len(evaluator._output)} evaluated molecules")
    elif molecules_file.exists():
        # Skip generation - load molecules and evaluate
        print(f"✓ Found existing generated molecules: {molecules_file}")
        print(f"  Skipping generation step...")
        evaluator.from_tensor(molecules_file)
        print(f"✓ Loaded {len(evaluator._output)} generated molecules")
        
        # Evaluate with docking scores
        print(f"\n[2/2] Calculating docking scores and molecular properties...")
        evaluator.evaluate(apply_transform=True, scorer=scorer, output_format='sdf')
        evaluator.to_html(output_path / "results.html")
        evaluator.to_tensor(output_path / "results.pt")
    else:
        # Generate molecules from scratch
        print(f"[1/2] Generating molecules...")
        print(f"  - Limit samples: {limit_samples}")
        print(f"  - Molecules per pocket: {molecules_per_pocket}")
        print(f"  - Batch size: {batch_size}")
        
        evaluator.generate_molecules(
            limit_samples=limit_samples,
            molecules_per_pocket=molecules_per_pocket,
            batch_size=batch_size,
        )
        evaluator.to_tensor(output_path / "molecules_generated.pt")
        print(f"✓ Molecules generated and saved")
        
        # Evaluate with docking scores
        print(f"\n[2/2] Calculating docking scores and molecular properties...")
        evaluator.evaluate(apply_transform=True, scorer=scorer, output_format='sdf')
        evaluator.to_html(output_path / "results.html")
        evaluator.to_tensor(output_path / "results.pt")
    
    # Calculate comprehensive metrics
    print(f"\n{'='*60}")
    print(f"CALCULATING COMPREHENSIVE METRICS")
    print(f"{'='*60}\n")
    
    metrics = {}
    
    # Calculate train/val losses via model inference
    loss_metrics = calculate_model_losses(model, evaluator.data_module, device=device)
    metrics.update(loss_metrics)
    
    # Extract molecules from evaluator output
    molecules = []
    if hasattr(evaluator, '_output') and evaluator._output is not None:
        for _, row in evaluator._output.iterrows():
            if 'molecule' in row and row['molecule'] is not None:
                molecules.append(row['molecule'])
    
    print(f"Total molecules generated: {len(molecules)}")
    
    
    if len(molecules) > 0:
        # Initialize metrics
        validity_metric = MolecularValidity()
        connectivity_metric = MolecularConnectivity()
        novelty_metric = MolecularNovelty(evaluator.data_module.get_train_smiles())
        qed_metric = MolecularQEDValue()
        sascore_metric = MolecularSAScore()
        logp_metric = MolecularLogP()
        lipinski_metric = MolecularLipinski()
        
        # Update metrics with molecules
        validity_metric.update(molecules)
        connectivity_metric.update(molecules)
        novelty_metric.update(molecules)
        qed_metric.update(molecules)
        sascore_metric.update(molecules)
        logp_metric.update(molecules)
        lipinski_metric.update(molecules)
        
        # Compute metrics
        metrics['Validity'] = float(validity_metric.compute())
        metrics['Connectivity'] = float(connectivity_metric.compute())
        metrics['Novelty'] = float(novelty_metric.compute())
        metrics['Lipinski'] = float(lipinski_metric.compute())
        metrics['LogP'] = float(logp_metric.compute())
        metrics['QED'] = float(qed_metric.compute())
        metrics['SAScore'] = float(sascore_metric.compute())
        
        # Calculate diversity
        valid_molecules = [mol for mol in molecules if mol is not None]
        metrics['Diversity'] = calculate_diversity(valid_molecules)
        
        # Get Vina/AutoDock scores
        if hasattr(evaluator, '_output') and 'AutoDockGPU' in evaluator._output.columns:
            all_scores = evaluator._output['AutoDockGPU']
            vina_scores = all_scores.dropna()
            
            total_molecules = len(all_scores)
            successful_docking = len(vina_scores)
            failed_docking = total_molecules - successful_docking
            
            print(f"\nAutoDock-GPU Docking Results:")
            print(f"  Total molecules: {total_molecules}")
            print(f"  Successfully docked: {successful_docking} ({successful_docking/total_molecules*100:.1f}%)")
            print(f"  Failed to dock: {failed_docking} ({failed_docking/total_molecules*100:.1f}%)")
            
            if len(vina_scores) > 0:
                metrics['Vina (mean)'] = float(vina_scores.mean())
                metrics['Vina (min)'] = float(vina_scores.min())
                metrics['Vina (docked)'] = successful_docking
                metrics['Vina (failed)'] = failed_docking
            else:
                print(f"  WARNING: No molecules were successfully docked!")
                metrics['Vina (docked)'] = 0
                metrics['Vina (failed)'] = total_molecules
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"FINAL METRICS")
    print(f"{'='*60}")
    for metric_name, value in metrics.items():
        if 'docked' in metric_name.lower() or 'failed' in metric_name.lower():
            # Print integer counts for docking stats
            print(f"{metric_name:20s}: {int(value)}")
        else:
            print(f"{metric_name:20s}: {value:.4f}")
    print(f"{'='*60}\n")
    
    # Save metrics to JSON
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Save detailed summary
    evaluator.print_summary_statistics()
    summary_text = f"Evaluation Summary\n"
    summary_text += f"{'='*60}\n\n"
    summary_text += f"Run Directory: {run_dir}\n"
    summary_text += f"Output Path: {output_path}\n\n"
    summary_text += f"Metrics:\n"
    for metric_name, value in metrics.items():
        summary_text += f"  {metric_name:20s}: {value:.4f}\n"
    summary_text += f"\n{evaluator.get_summary_string()}\n"
    
    (output_path / "summary.txt").write_text(summary_text)
    
    return metrics


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        prog="eval_wandb_run.py",
        description="Evaluate model from W&B run or author checkpoint",
        epilog=(
            "Examples:\n"
            "  W&B run: python eval_wandb_run.py /path/to/wandb/checkpoints/peachy-firebrand-37\n"
            "  Author checkpoint: python eval_wandb_run.py /path/to/author_checkpoints/gvp_conditional.ckpt"
        ),
    )
    parser.add_argument(
        "run_path",
        type=str,
        help="Path to W&B run directory or author checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate on",
        default="pdbbind_filtered",
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
        help="Output directory for results",
        default=None,
    )
    parser.add_argument(
        "--scorer",
        type=str,
        help="Scoring method",
        default="autodock_gpu",
        choices=["autodock_gpu"],
    )
    args = parser.parse_args()

    # Setup paths
    run_dir = Path(args.run_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"Path not found: {run_dir}")
    
    # Store original run_dir for W&B history lookup
    original_run_dir = run_dir
    
    # Determine output name based on whether it's an author checkpoint or W&B run
    author_checkpoint_dir = Path(os.getenv("DUNG_HOME", "")) / "Code/pharma_res/results/diff_hopp/author_checkpoints"
    
    if run_dir.parent == author_checkpoint_dir and run_dir.is_file() and run_dir.suffix == ".ckpt":
        # Author checkpoint: use filename without extension
        output_name = run_dir.stem
        checkpoint_path = run_dir
        run_dir = run_dir.parent  # Set run_dir to parent for config loading
    else:
        # W&B run directory: use directory name
        output_name = run_dir.name
        checkpoint_path = None  # Will be determined in setup function
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset
    
    # Determine output directory (just run name, no dataset subdirectory)
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        eval_base = os.getenv("EVALUATION_OUTPUT_DIR", "evaluation")
        output_path = Path(eval_base) / output_name
    
    output_path.mkdir(parents=True, exist_ok=True)

    # Disable logging noise
    disable_obabel_and_rdkit_logging()

    # Print configuration
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Input path: {args.run_path}")
    print(f"Output name: {output_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Limit samples: {args.limit_samples}")
    print(f"Molecules per pocket: {args.molecules_per_pocket}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Output path: {output_path}")
    print(f"Scorer: {args.scorer}")
    print(f"{'='*60}\n")

    # Load model and data
    if checkpoint_path:
        # Author checkpoint: load directly
        print(f"\nLoading author checkpoint: {checkpoint_path}")
        model = DiffusionHoppingModel.load_from_checkpoint(
            checkpoint_path, map_location=device, weights_only=False
        ).to(device)
        print(f"✓ Model loaded successfully")
        print(f"\nLoading dataset: {dataset_name}")
        data_module = get_datamodule(dataset_name, batch_size=32, shuffle=False)
        print(f"✓ Data module loaded successfully (shuffle=False for deterministic evaluation)")
        config = {}
    else:
        # W&B run: use existing setup function
        model, data_module, config = setup_model_and_data_module(
            run_dir, dataset_name, device=device
        )

    # Create evaluator
    evaluator = Evaluator(output_path)
    evaluator.load_data_module(data_module)
    evaluator.load_model(model)

    # Run comprehensive evaluation
    metrics = evaluate_with_metrics(
        evaluator,
        output_path,
        original_run_dir,
        model,
        device=device,
        limit_samples=args.limit_samples,
        molecules_per_pocket=args.molecules_per_pocket,
        batch_size=args.batch_size,
        scorer=args.scorer,
    )

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Metrics file: {output_path / 'metrics.json'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()