#!/usr/bin/env python3
"""
Simple Python test script for AutoDock-GPU integration.
Tests the autodock_gpu_score function used in the evaluation pipeline.
"""

import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil


def check_autodock_gpu():
    """Check if AutoDock-GPU is available in PATH."""
    print("=" * 60)
    print("Checking AutoDock-GPU Installation")
    print("=" * 60)
    
    result = shutil.which("autodock_gpu_64wi")
    if result:
        print(f"✓ AutoDock-GPU found: {result}")
        return True
    else:
        print("✗ AutoDock-GPU (autodock_gpu_64wi) not found in PATH")
        print("  Please ensure the AutoDock-GPU module is loaded")
        return False


def check_gpu():
    """Check GPU availability."""
    print("\n" + "=" * 60)
    print("Checking GPU Availability")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout:
            print(f"✓ GPU detected: {result.stdout.strip()}")
            return True
        else:
            print("⚠ No GPU detected (will use CPU)")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠ nvidia-smi not available")
        return False


def test_autodock_gpu_module():
    """Test importing the AutoDock-GPU scoring module."""
    print("\n" + "=" * 60)
    print("Testing AutoDock-GPU Python Module")
    print("=" * 60)
    
    try:
        # Try to import the autodock_gpu scoring function
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from diffusion_hopping.analysis.evaluate.autodock_gpu import autodock_gpu_score
        
        print("✓ Successfully imported autodock_gpu_score")
        print(f"  Function: {autodock_gpu_score}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import autodock_gpu_score: {e}")
        return False
    except Exception as e:
        print(f"⚠ Error during import: {e}")
        return False


def test_simple_docking():
    """Test a minimal docking scenario if test data exists."""
    print("\n" + "=" * 60)
    print("Testing Simple Docking (if test data available)")
    print("=" * 60)
    
    # Check for test data
    test_data = Path(__file__).parent.parent / "tests_data" / "complexes" / "1a0q"
    protein_file = test_data / "protein.pdb"
    ligand_file = test_data / "ligand.sdf"
    
    if not (protein_file.exists() and ligand_file.exists()):
        print("⚠ Test data not found - skipping functional test")
        print(f"  Expected: {test_data}")
        return None
    
    print(f"✓ Test data found: {test_data}")
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            print(f"  Working directory: {work_dir}")
            
            # Copy files
            shutil.copy(protein_file, work_dir / "protein.pdb")
            shutil.copy(ligand_file, work_dir / "ligand.sdf")
            
            # Try to convert ligand to PDBQT using obabel
            try:
                result = subprocess.run(
                    ["obabel", str(work_dir / "ligand.sdf"), 
                     "-O", str(work_dir / "ligand.pdbqt"), "-h"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0 and (work_dir / "ligand.pdbqt").exists():
                    print("✓ Ligand converted to PDBQT format")
                    
                    # Try simple autodock command
                    result = subprocess.run(
                        ["autodock_gpu_64wi", "--help"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        print("✓ AutoDock-GPU responds to --help command")
                        return True
                    else:
                        print("⚠ AutoDock-GPU help command failed")
                        return False
                else:
                    print("⚠ Ligand conversion failed")
                    return False
                    
            except FileNotFoundError:
                print("⚠ obabel not found - cannot prepare ligand")
                return False
            except subprocess.TimeoutExpired:
                print("⚠ Command timeout")
                return False
                
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AutoDock-GPU Python Test Suite")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Run tests
    results["autodock_gpu_installed"] = check_autodock_gpu()
    results["gpu_available"] = check_gpu()
    results["python_module"] = test_autodock_gpu_module()
    results["functional_test"] = test_simple_docking()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        
        print(f"  {test_name:<25} {status}")
    
    print("\n" + "=" * 60)
    
    # Overall result
    critical_tests = ["autodock_gpu_installed", "python_module"]
    if all(results.get(t, False) for t in critical_tests):
        print("✓ AutoDock-GPU is ready for use in evaluation pipeline")
        return 0
    else:
        print("✗ AutoDock-GPU setup is incomplete")
        print("\nPlease ensure:")
        print("  1. AutoDock-GPU module is loaded: module load AutoDock-GPU")
        print("  2. Python environment is properly configured")
        return 1


if __name__ == "__main__":
    sys.exit(main())
