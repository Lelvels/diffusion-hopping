#!/bin/bash

# --- 1. Load Helios-optimized ML environment ---
module load Python/3.11.5
module load GCC/13.2.0
module load CMake/3.29.3
module load Boost/1.83.0
module load SWIG/4.1.1
module load bzip2/1.0.8
module load ML-bundle/24.06a

# --- 2. Define Paths & Redirect Cache to Scratch ---
export DUNG_HOME="/net/scratch/hscra/plgrid/plgkietho/maidung"
export VENV_PATH="$DUNG_HOME/diffusion_hopping_venv"

mkdir -p "$DUNG_HOME/.pip_cache" "$DUNG_HOME/.tmp"
export PIP_CACHE_DIR="$DUNG_HOME/.pip_cache"
export TMPDIR="$DUNG_HOME/.tmp"
export PIP_NO_CACHE_DIR=1

# Research Tool Paths (Points to where you compiled autogrid4 and obabel)
export PATH="$DUNG_HOME/my_software/bin:$PATH"
export LD_LIBRARY_PATH="$DUNG_HOME/my_software/lib:$LD_LIBRARY_PATH"
export BABEL_DATADIR="$DUNG_HOME/my_software/share/openbabel/3.1.1"

# --- 3. Data root for research datasets ---
echo "=== START DIFFUSION-HOPPING ENVIRONMENT SETUP ==="
echo "DUNG_HOME: $DUNG_HOME"
echo "VENV_PATH: $VENV_PATH"
export DIFFUSION_HOPPING_DATA_ROOT="$DUNG_HOME/Code/pharma_res/data"

# --- 4. Virtual Environment Setup ---
cd "$DUNG_HOME"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# --- 5. Package Installation ---
echo "Installing packages individually..."
pip install --no-cache torch pytorch-lightning torchmetrics torch-geometric --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
pip install --no-cache numpy pandas scipy networkx tqdm PyYAML click pytest
pip install --no-cache rdkit meeko biopandas gemmi
pip install --no-cache wandb sentry-sdk python-dotenv

# --- 6. OpenBabel Python Bindings (Manual Source Build) ---
echo ""
echo "Checking OpenBabel Python bindings..."

# 6.1 Try to use existing installation first
if python -c "from openbabel import openbabel; print(f'Found OpenBabel version: {openbabel.OBReleaseVersion()}')" 2>/dev/null; then
    echo "✓ OpenBabel already installed and working. Skipping build."
else
    echo "OpenBabel not found or broken. Installing from source..."

    # 6.2 Define Paths
    OPENBABEL_PREFIX="$DUNG_HOME/my_software"
    OPENBABEL_LIB="$OPENBABEL_PREFIX/lib"
    OPENBABEL_INC="$OPENBABEL_PREFIX/include/openbabel3"

    # 6.3 Pre-flight checks
    if [ ! -f "$OPENBABEL_LIB/libopenbabel.so" ]; then
        if [ -f "$OPENBABEL_LIB/libopenbabel.so.7" ]; then
            echo "Creating symlink for linker..."
            ln -sf "$OPENBABEL_LIB/libopenbabel.so.7" "$OPENBABEL_LIB/libopenbabel.so"
        else
            echo "ERROR: libopenbabel.so not found at $OPENBABEL_LIB"
            exit 1
        fi
    fi

    if [ ! -d "$OPENBABEL_INC/openbabel" ]; then
        echo "ERROR: Headers not found in expected location."
        echo "Looking for: $OPENBABEL_INC/openbabel"
        exit 1
    fi

    # 6.4 Remove old versions
    pip uninstall -y openbabel-wheel openbabel 2>/dev/null || true

    # Install build dependencies
    echo "Installing build dependencies..."
    pip install --no-cache setuptools wheel

    # 6.5 Set build environment
    export CFLAGS="-I$OPENBABEL_INC"
    export CXXFLAGS="-I$OPENBABEL_INC"
    export LDFLAGS="-L$OPENBABEL_LIB -Wl,-rpath,$OPENBABEL_LIB"

    # 6.6 Download and Install from Source
    BUILD_DIR="$TMPDIR/ob_build"
    # Clean up previous failed builds
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    echo "Downloading OpenBabel source..."
    pip download --no-binary :all: --no-deps openbabel

    echo "Extracting source..."
    tar -xzf openbabel-*.tar.gz

    # Find the extracted directory name
    EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "openbabel-*" | head -1)
    if [ -z "$EXTRACTED_DIR" ]; then
        echo "ERROR: Could not find extracted openbabel directory"
        exit 1
    fi

    cd "$EXTRACTED_DIR"

    echo "Compiling with explicit paths..."
    # Run the build explicitly pointing to your custom software paths
    python3 setup.py build_ext \
        --include-dirs="$OPENBABEL_INC" \
        --library-dirs="$OPENBABEL_LIB" \
        --rpath="$OPENBABEL_LIB" \
        install

    # Cleanup
    cd "$DUNG_HOME"
    rm -rf "$BUILD_DIR"

    # 6.7 Verify
    echo "Verifying OpenBabel Python import..."
    # Ensure runtime linker knows where to look
    export LD_LIBRARY_PATH="$OPENBABEL_LIB:$LD_LIBRARY_PATH"

    python -c "from openbabel import openbabel; print(f'✓ OpenBabel Python bindings loaded successfully. Version: {openbabel.OBReleaseVersion()}')" || {
        echo "ERROR: OpenBabel import failed."
        echo ""
        echo "Debug info:"
        echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
        echo ""
        echo "Checking _openbabel.so dependencies:"
        ldd "$VENV_PATH/lib/python3.11/site-packages/openbabel/_openbabel.so" | grep openbabel || echo "No openbabel lib linked"
        exit 1
    }

    # --- 3.1 Weights & Biases configuration ---
    export WANDB_DIR="$DUNG_HOME/Code/pharma_res/results/diff_hopp"
    mkdir -p "$WANDB_DIR"
    echo "W&B runs will be saved to: $WANDB_DIR"
fi

echo ""
echo "Environment setup complete. Virtual environment activated at: $VENV_PATH"
echo "To activate this environment in future sessions, run: source $VENV_PATH/bin/activate"