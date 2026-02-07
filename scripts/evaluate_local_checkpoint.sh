#!/bin/bash

# Evaluation script using LOCAL checkpoint files (no WandB required)
# Usage: ./evaluate_local_checkpoint.sh [checkpoint_name] [optional_arguments]

set -e  # Exit on error
cd "$(dirname "$0")/.."  # Change to project root

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print header
print_header() {
    echo ""
    print_message "$BLUE" "============================================================"
    print_message "$BLUE" "$1"
    print_message "$BLUE" "============================================================"
    echo ""
}

# Check if checkpoint is provided
if [ $# -lt 1 ]; then
    print_message "$RED" "Error: Missing checkpoint argument"
    echo ""
    echo "Usage: $0 <checkpoint> [options]"
    echo ""
    echo "Available checkpoints:"
    echo "  gvp_conditional       DiffHopp (default, for scaffold hopping)"
    echo "  gvp_unconditional     DiffHopp (for inpainting)"
    echo "  egnn_conditional      DiffHopp-EGNN (for scaffold hopping)"
    echo "  egnn_unconditional    DiffHopp-EGNN (for inpainting)"
    echo ""
    echo "Aliases:"
    echo "  gvp, diffhopp         → gvp_conditional"
    echo "  egnn, diffhopp-egnn   → egnn_conditional"
    echo ""
    echo "Options:"
    echo "  --dataset NAME        Dataset name (default: pdbbind_filtered)"
    echo "  --mode MODE           Evaluation mode: ground_truth, ligand_generation,"
    echo "                        inpaint_generation, or all (default: ligand_generation)"
    echo "  --limit_samples N     Number of samples to evaluate (default: 500)"
    echo "  --molecules N         Molecules per pocket (default: 10)"
    echo "  --batch_size N        Batch size for generation (default: 32)"
    echo "  --only_generation     Only generate molecules, skip evaluation"
    echo "  --only_evaluation     Only evaluate molecules, skip generation"
    echo "  --output_dir DIR      Output directory (default: evaluation_local)"
    echo "  --checkpoints_dir DIR Checkpoints directory (default: checkpoints)"
    echo ""
    echo "Examples:"
    echo "  $0 gvp_conditional"
    echo "  $0 egnn_conditional --limit_samples 500"
    echo "  $0 gvp_unconditional --mode inpaint_generation"
    echo "  $0 diffhopp --molecules 20 --batch_size 16"
    exit 1
fi

# Print header
print_header "Local Checkpoint Evaluation (500 Samples)"

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_message "$YELLOW" "⚠ Warning: No conda environment detected"
    print_message "$YELLOW" "Please activate the environment with: conda activate diffusion_hopping"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
elif [ "$CONDA_DEFAULT_ENV" != "diffusion_hopping" ] && [ "$CONDA_DEFAULT_ENV" != "turbohopp" ]; then
    print_message "$YELLOW" "⚠ Warning: Current environment is '$CONDA_DEFAULT_ENV'"
    print_message "$YELLOW" "Expected 'diffusion_hopping' or 'turbohopp'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_message "$GREEN" "✓ Conda environment: $CONDA_DEFAULT_ENV"
fi

# Check if checkpoints directory exists
CHECKPOINTS_DIR="checkpoints"
# Extract checkpoints_dir from arguments if provided
for arg in "$@"; do
    if [[ $arg == --checkpoints_dir=* ]]; then
        CHECKPOINTS_DIR="${arg#*=}"
    elif [[ $arg == "--checkpoints_dir" ]]; then
        # Handle --checkpoints_dir DIR format (next argument is the directory)
        shift
        CHECKPOINTS_DIR="$1"
    fi
done

if [ ! -d "$CHECKPOINTS_DIR" ]; then
    print_message "$RED" "✗ Checkpoints directory not found: $CHECKPOINTS_DIR"
    exit 1
else
    print_message "$GREEN" "✓ Checkpoints directory: $CHECKPOINTS_DIR"
    # List available checkpoints
    CKPT_COUNT=$(find "$CHECKPOINTS_DIR" -name "*.ckpt" | wc -l)
    if [ "$CKPT_COUNT" -eq 0 ]; then
        print_message "$RED" "✗ No checkpoint files found in $CHECKPOINTS_DIR"
        exit 1
    else
        print_message "$GREEN" "✓ Found $CKPT_COUNT checkpoint file(s)"
    fi
fi

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    print_message "$GREEN" "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    print_message "$YELLOW" "⚠ No GPU detected, will use CPU (slower)"
fi

# Check if AutoDock-GPU is available
if command -v autodock_gpu_64wi &> /dev/null; then
    print_message "$GREEN" "✓ AutoDock-GPU found in PATH"
else
    print_message "$YELLOW" "⚠ AutoDock-GPU not found in PATH"
    print_message "$YELLOW" "Make sure AutoDock-GPU module is loaded if running on HPC"
fi

echo ""
print_message "$BLUE" "Starting evaluation with local checkpoint..."
echo ""

# Run the evaluation script with all arguments
python evaluate_local_checkpoint.py "$@"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    print_header "Evaluation Completed Successfully!"
    print_message "$GREEN" "✓ All tasks completed"
    echo ""
    print_message "$BLUE" "Results are saved in the evaluation_local/ directory"
    echo ""
else
    echo ""
    print_header "Evaluation Failed"
    print_message "$RED" "✗ An error occurred during evaluation"
    echo ""
    exit 1
fi
