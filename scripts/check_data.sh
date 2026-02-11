#!/bin/bash -l
#SBATCH --job-name=check_data
#SBATCH --account=plgplgalphasyn3-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output="/net/storage/pr3/plgrid/plggsball/plgkietho/maidung/Code/pharma_res/results/diff_hopp/logs/check_data/check_data_%j.log"
#SBATCH --error="/net/storage/pr3/plgrid/plggsball/plgkietho/maidung/Code/pharma_res/results/diff_hopp/logs/check_data/check_data_%j.err"

# 1. Virtual Environment Setup ---
echo "=== STARTING EVALUATION ENVIRONMENT SETUP ==="
export DUNG_HOME="/net/storage/pr3/plgrid/plggsball/plgkietho/maidung"
export VENV_PATH="$DUNG_HOME/diffusion_hopping_venv"

# 2. Load Helios-optimized ML environment
sh "$DUNG_HOME/Code/pharma_res/diffusion-hopping/scripts/load_modules.sh"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"
echo "✓ Activated virtual environment: $VENV_PATH"
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to activate virtual environment at $VENV_PATH" >&2
    exit 1
fi

# 3. Load the modules required for AutoDock-GPU (if not already loaded by load_modules.sh)
echo "Loading modules..."
module load GCC/13.2.0 bzip2/1.0.8

export pybin="/net/software/aarch64/el9/Python/3.11.5-GCCcore-13.2.0/bin/python"
echo "✓ PYTHONPATH set to: $pybin, fuck default python path!!!"
echo "Python version: "
$pybin --version

# 5. Navigate to the project directory
cd "$DUNG_HOME/Code/pharma_res/diffusion-hopping"

# 6. Run the evaluation
# Ensure python is called from your activated venv (handled by load_modules.sh)
$pybin check_data.py pdbbind_filtered