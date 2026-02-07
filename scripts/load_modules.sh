#!/bin/bash

# --- 1. Load Helios-optimized ML environment ---
ml Python/3.11.5
ml ML-bundle/24.06a
ml GCC/13.2.0 AutoDock-GPU/1.5.3-CUDA-12.8.0
ml Boost/1.83.0
ml SWIG/4.1.1
ml CMake/3.29.3

# 2. Define Home and Scratch Paths
export DUNG_HOME="/net/scratch/hscra/plgrid/plgkietho/maidung"

# --- Redirect Cache & Temp to Scratch ---
mkdir -p "$DUNG_HOME/.pip_cache"
mkdir -p "$DUNG_HOME/.tmp"
export PIP_CACHE_DIR="$DUNG_HOME/.pip_cache"
export TMPDIR="$DUNG_HOME/.tmp"

# --- 3. Path Configuration ---
export PATH=$DUNG_HOME/my_software/bin:$PATH
export LD_LIBRARY_PATH=$DUNG_HOME/my_software/lib:$LD_LIBRARY_PATH
export BABEL_DATADIR=$DUNG_HOME/my_software/share/openbabel/3.1.1

# Data root for diffusion-hopping datasets
export DIFFUSION_HOPPING_DATA_ROOT="$DUNG_HOME/Code/pharma_res/data"

# --- 4. Synchronized Installation ---
# Note: Using --user and --force-reinstall to override broken system packages
pip install torch --no-cache-dir --force-reinstall --user \
  pytorch-lightning \
  torchmetrics \
  torch-geometric \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

# Install other dependencies (PIP_NO_CACHE_DIR handles the cache here)
pip install --no-cache-dir numpy pandas scipy networkx tqdm PyYAML click pytest
pip install --no-cache-dir rdkit meeko biopandas gemmi
pip install --no-cache-dir wandb sentry-sdk

# Open Babel installation
pip install --no-cache-dir openbabel \
  --global-option=build_ext \
  --global-option="-I$DUNG_HOME/my_software/include/openbabel3" \
  --global-option="-L$DUNG_HOME/my_software/lib"

echo "Environment setup complete without using $HOME cache."