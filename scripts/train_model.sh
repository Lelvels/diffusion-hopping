#!/bin/bash -l
#SBATCH --job-name=tr_dh
#SBATCH --account=plgplgalphasyn3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output="/net/scratch/hscra/plgrid/plgkietho/maidung/Code/pharma_res/results/diff_hopp/logs/train_model_%j.log"
#SBATCH --error="/net/scratch/hscra/plgrid/plgkietho/maidung/Code/pharma_res/results/diff_hopp/logs/train_model_%j.err"

# 1. Virtual Environment Setup ---
export DUNG_HOME="/net/scratch/hscra/plgrid/plgkietho/maidung"
export VENV_PATH="$DUNG_HOME/diffusion_hopping_venv"

# 2. Load Helios-optimized ML environment ---
bash "$DUNG_HOME/Code/pharma_res/diffusion-hopping/scripts/load_modules.sh"

# 3. Load the modules required for AutoDock-GPU (if not already loaded by load_modules.sh)
echo "Loading modules..."
module load GCC/13.2.0 bzip2/1.0.8
module load GCC/13.2.0 AutoDock-GPU/1.5.3-CUDA-12.8.0
echo "Checking AutoDock-GPU installation..."
if command -v autodock_gpu &> /dev/null; then
    echo "✓ autodock_gpu binary found"
    autodock_gpu --version 2>&1 || echo "Warning: Could not retrieve version"
else
    echo "ERROR: autodock_gpu binary not found in PATH" >&2
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"
echo "✓ Activated virtual environment: $VENV_PATH"
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to activate virtual environment at $VENV_PATH" >&2
    exit 1
fi

# 4. Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python3 << 'EOF'
import torch
cuda_available = torch.cuda.is_available()
cuda_device_count = torch.cuda.device_count() if cuda_available else 0

if cuda_available:
    print(f"✓ CUDA is available, with {cuda_device_count} device(s) detected.")
    print(f"  Device count: {cuda_device_count}")
    for i in range(cuda_device_count):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("ERROR: CUDA is not available. AutoDock-GPU requires CUDA.")
    exit(1)
EOF

if [[ $? -ne 0 ]]; then
    echo "ERROR: CUDA check failed" >&2
    exit 1
fi

# 5. Navigate to the project directory
cd "$DUNG_HOME/Code/pharma_res/diffusion-hopping"
# 6. Validate required dependencies
echo ""
echo "Validating dependencies..."
python3 -c "import pytorch_lightning; print('✓ pytorch_lightning import successful')" 2>&1
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to import pytorch_lightning" >&2
    exit 1
fi

python3 -c "import wandb; print('✓ wandb import successful')" 2>&1
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to import wandb" >&2
    exit 1
fi

# 7. Parse command-line arguments
echo ""
echo "=== START TRAINING SCRIPT ==="
echo "Parsing training arguments..."

# Default values (matching train_model.py arguments)
DATASET_NAME="pdbbind_filtered"
ARCHITECTURE="gvp"
LR="0.001"
BATCH_SIZE="32"
SEED="42"
NUM_STEPS="70000"
T="1000"
NUM_LAYERS="6"
JOINT_FEATURES="256"
HIDDEN_FEATURES="512"
EDGE_CUTOFF="5.0"
ATTENTION="true"
CONDITION_ON_FG="true"
SAVE_EVERY_N_STEPS="5000"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --lr|--learning_rate)
            LR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --T)
            T="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --hidden_features)
            HIDDEN_FEATURES="$2"
            shift 2
            ;;
        --joint_features)
            JOINT_FEATURES="$2"
            shift 2
            ;;
        --edge_cutoff)
            EDGE_CUTOFF="$2"
            shift 2
            ;;
        --attention)
            ATTENTION="$2"
            shift 2
            ;;
        --condition_on_fg)
            CONDITION_ON_FG="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --save_every_n_steps)
            SAVE_EVERY_N_STEPS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash train_model.sh [options]"
            echo ""
            echo "Model Architecture:"
            echo "  --architecture ARCH           Architecture: gvp or egnn (default: gvp)"
            echo "  --num_layers N                Number of layers (default: 4)"
            echo "  --hidden_features N           Hidden features dimension (default: 128)"
            echo "  --joint_features N            Joint features dimension (default: 128)"
            echo "  --attention BOOL              Use attention layers (default: true)"
            echo ""
            echo "Training Configuration:"
            echo "  --dataset_name NAME           Dataset to train on (default: pdbbind_filtered)"
            echo "  --batch_size BS               Batch size (default: 32)"
            echo "  --lr LR                       Learning rate (default: 0.001)"
            echo "  --seed SEED                   Random seed (default: 42)"
            echo "  --save_every_n_steps N        Save checkpoint every N steps (default: 25000)"
            echo ""
            echo "Diffusion Model:"
            echo "  --T STEPS                     Total diffusion steps (default: 1000)"
            echo "  --num_steps STEPS             Training steps (default: 1000)"
            echo "  --edge_cutoff DIST            Edge cutoff distance (default: 10.0)"
            echo "  --condition_on_fg BOOL        Condition on functional groups (default: true)"
            echo ""
            echo "  --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash train_model.sh                                   # Default: GVP on pdbbind_filtered"
            echo "  bash train_model.sh --architecture egnn --batch_size 16"
            echo "  bash train_model.sh --lr 0.0001 --num_layers 6"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 8. Validate architecture
if [[ "$ARCHITECTURE" != "gvp" && "$ARCHITECTURE" != "egnn" ]]; then
    echo "ERROR: Invalid architecture: $ARCHITECTURE"
    echo "Valid architectures: gvp, egnn"
    exit 1
fi

# 9. Print training configuration
echo ""
echo "{'='*60}"
echo "TRAINING CONFIGURATION"
echo "{'='*60}"
echo "Dataset: $DATASET_NAME"
echo "Architecture: $ARCHITECTURE"
echo ""
echo "Model Hyperparameters:"
echo "  Layers: $NUM_LAYERS"
echo "  Hidden features: $HIDDEN_FEATURES"
echo "  Joint features: $JOINT_FEATURES"
echo "  Attention: $ATTENTION"
echo "  Condition on FG: $CONDITION_ON_FG"
echo ""
echo "Training Hyperparameters:"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE"
echo "  Seed: $SEED"
echo "  Save checkpoint every: $SAVE_EVERY_N_STEPS steps"
echo ""
echo "Diffusion Hyperparameters:"
echo "  T (diffusion steps): $T"
echo "  Num steps: $NUM_STEPS"
echo "  Edge cutoff: $EDGE_CUTOFF"
echo "{'='*60}"
echo ""

# 10. Check if dataset exists
DATASET_PATH="$DUNG_HOME/Code/pharma_res/data/$DATASET_NAME"
if [[ ! -d "$DATASET_PATH" ]]; then
    echo "ERROR: Dataset directory not found: $DATASET_PATH" >&2
    echo ""
    echo "Available options:"
    echo "  - pdbbind_filtered       (C-alpha representation)"
    echo "  - pdbbind_filtered_full  (full atom representation)"
    echo "  - pdbbind                (unfiltered, C-alpha)"
    echo "  - pdbbind_full           (unfiltered, full atom)"
    echo ""
    echo "Please ensure the dataset has been created using:"
    echo "  python create_dataset.py $DATASET_NAME"
    exit 1
fi

echo "✓ Dataset found at: $DATASET_PATH"

# 11. Build and run training arguments
echo "Starting training..."
echo ""

TRAIN_ARGS="--dataset_name $DATASET_NAME --architecture $ARCHITECTURE"
TRAIN_ARGS="$TRAIN_ARGS --batch_size $BATCH_SIZE --lr $LR --seed $SEED"
TRAIN_ARGS="$TRAIN_ARGS --num_steps $NUM_STEPS --T $T"
TRAIN_ARGS="$TRAIN_ARGS --num_layers $NUM_LAYERS --hidden_features $HIDDEN_FEATURES"
TRAIN_ARGS="$TRAIN_ARGS --joint_features $JOINT_FEATURES --edge_cutoff $EDGE_CUTOFF"
TRAIN_ARGS="$TRAIN_ARGS --attention $ATTENTION --condition_on_fg $CONDITION_ON_FG"
TRAIN_ARGS="$TRAIN_ARGS --save_every_n_steps $SAVE_EVERY_N_STEPS"

python3 train_model.py $TRAIN_ARGS

if [[ $? -eq 0 ]]; then
    echo ""
    echo "{'='*60}"
    echo "TRAINING COMPLETED SUCCESSFULLY!"
    echo "{'='*60}"
    echo ""
else
    echo ""
    echo "{'='*60}"
    echo "ERROR: Training failed!"
    echo "{'='*60}"
    exit 1
fi