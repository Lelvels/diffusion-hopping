#!/bin/bash -l
#SBATCH --job-name=eval_diffhopp_fast
#SBATCH --account=plgplgalphasyn3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output="/net/storage/pr3/plgrid/plggsball/plgkietho/maidung/Code/pharma_res/results/diff_hopp/logs/eval/eval_diffhopp_fast_%j.log"
#SBATCH --error="/net/storage/pr3/plgrid/plggsball/plgkietho/maidung/Code/pharma_res/results/diff_hopp/logs/eval/eval_diffhopp_fast_%j.err"

# 1. Virtual Environment Setup ---
echo "=== STARTING EVALUATION ENVIRONMENT SETUP ==="
export DUNG_HOME="/net/storage/pr3/plgrid/plggsball/plgkietho/maidung"
export VENV_PATH="$DUNG_HOME/diffusion_hopping_venv"

# 2. Load Helios-optimized ML environment ---
sh "$DUNG_HOME/Code/pharma_res/diffusion-hopping/scripts/load_modules.sh"

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
export pybin="/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin/python"
echo "✓ PYTHONPATH set to: $pybin, fuck default python path!!!"
echo "Python version: "
$pybin --version

echo "Checking CUDA availability..."
$pybin << 'EOF'
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