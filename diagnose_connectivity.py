#!/usr/bin/env python3
"""
Diagnostic script to analyze molecular connectivity issues in generated samples.
Checks bond counts, inter-atomic distances, and model architecture.
"""

import argparse
from pathlib import Path
import numpy as np
from rdkit import Chem
import torch


def analyze_sdf_file(sdf_path):
    """Analyze a single SDF file for connectivity issues."""
    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False, removeHs=False)
    
    if mol is None:
        return None
    
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    
    # Calculate pairwise distances
    conf = mol.GetConformer()
    positions = np.array([conf.GetAtomPosition(i) for i in range(num_atoms)])
    
    # Compute distance matrix
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    
    # Typical bond lengths are 1.0-2.0 Å for most organic bonds
    # OpenBabel uses similar thresholds
    close_pairs = np.sum(np.triu(distances < 2.0, k=1))  # Upper triangle, exclude diagonal
    
    # Get atom types
    atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    return {
        'filename': sdf_path.name,
        'num_atoms': num_atoms,
        'num_bonds': num_bonds,
        'close_pairs': close_pairs,  # Atoms within bonding distance
        'connectivity_ratio': num_bonds / max(num_atoms - 1, 1),  # Should be ~1.0 for connected
        'atom_types': atom_types,
        'min_distance': np.min(distances[distances > 0]),  # Exclude self-distances
        'mean_distance': np.mean(distances[np.triu_indices_from(distances, k=1)]),
    }


def check_model_architecture(checkpoint_path):
    """Check which architecture the model uses."""
    try:
        # Add safe globals
        from diffusion_hopping.model.enum import Architecture, Parametrization, SamplingMode
        from diffusion_hopping.data.featurization import ProteinLigandSimpleFeaturization
        from diffusion_hopping.data.transform import ChainSelectionTransform
        from diffusion_hopping.data.filter import QEDThresholdFilter
        from pathlib import PosixPath
        
        torch.serialization.add_safe_globals([
            Parametrization,
            Architecture,
            SamplingMode,
            ProteinLigandSimpleFeaturization,
            ChainSelectionTransform,
            PosixPath,
            QEDThresholdFilter,
        ])
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        arch = ckpt.get('hyper_parameters', {}).get('architecture', 'Not found')
        
        return {
            'architecture': str(arch),
            'parametrization': str(ckpt.get('hyper_parameters', {}).get('parametrization', 'Not found')),
            'edge_cutoff': ckpt.get('hyper_parameters', {}).get('edge_cutoff', 'Not found'),
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Diagnose molecular connectivity issues')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing generated SDF files')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (optional)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return
    
    # Analyze all SDF files
    sdf_files = sorted(results_dir.glob('*.sdf'))
    
    if not sdf_files:
        print(f"No SDF files found in {results_dir}")
        return
    
    print(f"Analyzing {len(sdf_files)} SDF files...\n")
    
    results = []
    for sdf_file in sdf_files:
        result = analyze_sdf_file(sdf_file)
        if result:
            results.append(result)
    
    # Print summary
    print("=" * 80)
    print("CONNECTIVITY ANALYSIS")
    print("=" * 80)
    print(f"{'Filename':<20} {'Atoms':>6} {'Bonds':>6} {'Close Pairs':>12} {'Conn. Ratio':>12} {'Min Dist':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['filename']:<20} {r['num_atoms']:>6} {r['num_bonds']:>6} {r['close_pairs']:>12} "
              f"{r['connectivity_ratio']:>12.2f} {r['min_distance']:>10.3f}")
    
    # Statistics
    bond_counts = [r['num_bonds'] for r in results]
    connectivity_ratios = [r['connectivity_ratio'] for r in results]
    
    print("-" * 80)
    print(f"{'STATISTICS':<20}")
    print(f"  Total molecules: {len(results)}")
    print(f"  Fully disconnected (0 bonds): {sum(1 for b in bond_counts if b == 0)}")
    print(f"  Poorly connected (<50% bonds): {sum(1 for r in connectivity_ratios if r < 0.5)}")
    print(f"  Well connected (>80% bonds): {sum(1 for r in connectivity_ratios if r > 0.8)}")
    print(f"  Mean bonds: {np.mean(bond_counts):.1f} ± {np.std(bond_counts):.1f}")
    print(f"  Mean connectivity ratio: {np.mean(connectivity_ratios):.2f} ± {np.std(connectivity_ratios):.2f}")
    
    # Check model architecture if checkpoint provided
    if args.checkpoint:
        print("\n" + "=" * 80)
        print("MODEL CONFIGURATION")
        print("=" * 80)
        model_info = check_model_architecture(Path(args.checkpoint))
        for key, value in model_info.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    zero_bond_count = sum(1 for b in bond_counts if b == 0)
    if zero_bond_count > 0:
        print(f"⚠️  WARNING: {zero_bond_count} molecules have ZERO bonds!")
        print("   This indicates severe coordinate errors during generation.")
        print("   Possible causes:")
        print("   1. Model architecture (EGNN has worse coordinate precision than GVP)")
        print("   2. Insufficient training")
        print("   3. OpenBabel bond inference thresholds too strict")
    
    poor_connectivity = sum(1 for r in connectivity_ratios if r < 0.5)
    if poor_connectivity > len(results) * 0.2:
        print(f"⚠️  WARNING: {poor_connectivity}/{len(results)} molecules are poorly connected!")
        print("   Connectivity ratio should be close to 1.0 for fully connected molecules.")
    
    if np.mean(connectivity_ratios) > 0.8:
        print("✓ Overall connectivity is good (mean ratio > 0.8)")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
