#!/bin/bash

# Quick installation of autogrid4 using conda/mamba
# Run this on Helios after loading the modules

echo "================================================================"
echo "Installing MGLTools (includes autogrid4) via conda"
echo "================================================================"

# Create a dedicated conda environment (optional but recommended)
# Skip this if you want to install in base environment
# mamba create -n autodock python=3.11 -y
# conda activate autodock

# Install MGLTools which includes autogrid4
# Note: MGLTools is available on bioconda channel
mamba install -c bioconda mgltools -y || \
  conda install -c bioconda mgltools -y || {
    echo "Error: Failed to install MGLTools via conda"
    echo "Trying alternative: downloading precompiled AutoDock4..."
    
    # Alternative: Download precompiled binaries
    mkdir -p "$DUNG_HOME/my_software/bin"
    cd "$DUNG_HOME/my_software/bin"
    
    # Download Linux x86_64 binaries
    wget https://autodock.scripps.edu/downloads/autodock-registration/tars/dist426/autodocksuite-4.2.6-x86_64Linux2.tar
    tar -xf autodocksuite-4.2.6-x86_64Linux2.tar
    
    # Copy binaries to bin directory
    cp x86_64Linux2/autogrid4 .
    cp x86_64Linux2/autodock4 .
    chmod +x autogrid4 autodock4
    
    rm -rf x86_64Linux2 autodocksuite-4.2.6-x86_64Linux2.tar
    
    echo "✓ Precompiled binaries installed"
}

echo "================================================================"
echo "✓ Installation complete"
echo "================================================================"
echo "Testing autogrid4..."
which autogrid4 || echo "autogrid4 in: $DUNG_HOME/my_software/bin/autogrid4"
autogrid4 -h 2>&1 | head -n 3 || "$DUNG_HOME/my_software/bin/autogrid4" -h 2>&1 | head -n 3
echo "================================================================"
