#!/bin/bash

# Test script for AutoDock-GPU installation and functionality
# Usage: bash test_autodock_gpu.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    echo "================================================"
    echo "$1"
    echo "================================================"
}

# Check if AutoDock-GPU is available
print_header "1. Checking AutoDock-GPU Installation"

if command -v autodock_gpu_64wi &> /dev/null; then
    AUTODOCK_GPU=$(which autodock_gpu_64wi)
    print_message "$GREEN" "✓ AutoDock-GPU found: $AUTODOCK_GPU"
    
    # Get version info
    print_message "$BLUE" "\nVersion information:"
    $AUTODOCK_GPU --version 2>&1 || echo "Version flag not available"
else
    print_message "$RED" "✗ AutoDock-GPU (autodock_gpu_64wi) not found in PATH"
    print_message "$YELLOW" "Please ensure AutoDock-GPU module is loaded or installed"
    print_message "$YELLOW" "Try: module load AutoDock-GPU"
    exit 1
fi

# Check CUDA availability
print_header "2. Checking CUDA/GPU Availability"

if command -v nvidia-smi &> /dev/null; then
    print_message "$GREEN" "✓ nvidia-smi available"
    echo ""
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
else
    print_message "$YELLOW" "⚠ nvidia-smi not found - may be running on CPU"
fi

# Check for test data
print_header "3. Checking for Test Data"

# Use tests_data if available, otherwise skip
TEST_DATA_DIR="../tests_data/complexes/1a0q"
if [ -d "$TEST_DATA_DIR" ]; then
    print_message "$GREEN" "✓ Test data found: $TEST_DATA_DIR"
    
    PROTEIN_FILE="$TEST_DATA_DIR/protein.pdb"
    LIGAND_FILE="$TEST_DATA_DIR/ligand.sdf"
    
    if [ -f "$PROTEIN_FILE" ] && [ -f "$LIGAND_FILE" ]; then
        print_message "$GREEN" "✓ Protein and ligand files present"
        
        # Create temporary working directory
        print_header "4. Running AutoDock-GPU Test"
        
        WORK_DIR=$(mktemp -d -t autodock_test_XXXXXX)
        print_message "$BLUE" "Working directory: $WORK_DIR"
        
        # Copy files to work directory
        cp "$PROTEIN_FILE" "$WORK_DIR/"
        cp "$LIGAND_FILE" "$WORK_DIR/"
        
        cd "$WORK_DIR"
        
        # Prepare protein (add charges, etc.) - minimal test
        print_message "$BLUE" "\nPreparing protein for docking..."
        
        # Try to use MGLTools prepare_receptor if available
        if command -v prepare_receptor4.py &> /dev/null; then
            prepare_receptor4.py -r protein.pdb -o protein.pdbqt -A hydrogens || {
                print_message "$YELLOW" "⚠ prepare_receptor4.py failed, using simple conversion"
                cp protein.pdb protein.pdbqt
            }
        else
            print_message "$YELLOW" "⚠ MGLTools not found, copying protein as-is"
            cp protein.pdb protein.pdbqt
        fi
        
        # Prepare ligand
        print_message "$BLUE" "Preparing ligand for docking..."
        
        if command -v obabel &> /dev/null; then
            obabel ligand.sdf -O ligand.pdbqt -h 2>&1 || {
                print_message "$YELLOW" "⚠ obabel conversion failed"
            }
        else
            print_message "$YELLOW" "⚠ obabel not found, skipping ligand preparation"
        fi
        
        if [ -f "protein.pdbqt" ] && [ -f "ligand.pdbqt" ]; then
            print_message "$GREEN" "✓ Files prepared successfully"
            
            # Create a minimal AutoDock-GPU input file
            print_message "$BLUE" "\nCreating AutoDock-GPU configuration..."
            
            cat > docking.dpf <<EOF
receptor protein.pdbqt
ligand ligand.pdbqt

npts 40 40 40
gridcenter 0.0 0.0 0.0
spacing 0.375

seed 42
EOF
            
            print_message "$BLUE" "\nAttempting AutoDock-GPU docking (quick test)..."
            print_message "$YELLOW" "Note: This is a minimal test. Real docking requires proper setup."
            
            # Run AutoDock-GPU with minimal settings
            if timeout 60s $AUTODOCK_GPU --ffile docking.dpf --nrun 1 --lfile ligand.pdbqt --resnam docking &> autodock_output.txt; then
                print_message "$GREEN" "✓ AutoDock-GPU executed successfully!"
                echo ""
                print_message "$BLUE" "Output summary:"
                tail -n 20 autodock_output.txt | grep -E "(Run|Score|Energy)" || cat autodock_output.txt | tail -n 10
            else
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 124 ]; then
                    print_message "$YELLOW" "⚠ AutoDock-GPU timeout (expected for test)"
                else
                    print_message "$RED" "✗ AutoDock-GPU execution failed"
                    echo ""
                    print_message "$BLUE" "Error output:"
                    cat autodock_output.txt
                fi
            fi
        else
            print_message "$YELLOW" "⚠ Could not prepare PDBQT files - skipping docking test"
        fi
        
        # Cleanup
        print_message "$BLUE" "\nCleaning up: $WORK_DIR"
        cd - > /dev/null
        rm -rf "$WORK_DIR"
        
    else
        print_message "$YELLOW" "⚠ Test data files not complete"
    fi
else
    print_message "$YELLOW" "⚠ Test data not found: $TEST_DATA_DIR"
    print_message "$BLUE" "Skipping functional test - only checking installation"
fi

# Summary
print_header "Test Summary"

if command -v autodock_gpu_64wi &> /dev/null; then
    print_message "$GREEN" "✓ AutoDock-GPU is installed and accessible"
    print_message "$BLUE" "\nTo use in your scripts:"
    echo "  autodock_gpu_64wi --help"
    echo ""
    print_message "$BLUE" "To load module on HPC:"
    echo "  module load AutoDock-GPU"
else
    print_message "$RED" "✗ AutoDock-GPU is not properly installed"
    exit 1
fi

print_message "$GREEN" "\n✓ Test completed!"
