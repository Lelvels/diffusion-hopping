#!/bin/bash

# Install AutoDock4 (includes autogrid4) on HPC
# This script should be run on the Helios cluster

set -e

INSTALL_DIR="$DUNG_HOME/my_software"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "================================================================"
echo "Installing AutoDock4 (includes autogrid4)"
echo "Installation directory: $INSTALL_DIR"
echo "================================================================"

# Download AutoDock4
AUTODOCK_VERSION="4.2.6"
AUTODOCK_URL="https://autodock.scripps.edu/downloads/autodock-registration/tars/dist426/autodocksuite-${AUTODOCK_VERSION}-src.tar.gz"

echo "Downloading AutoDock ${AUTODOCK_VERSION}..."
wget -O autodocksuite.tar.gz "$AUTODOCK_URL" || {
    echo "Error: Failed to download AutoDock"
    echo "You may need to manually download from: https://autodock.scripps.edu/download-autodock4/"
    exit 1
}

# Extract
echo "Extracting..."
tar -xzf autodocksuite.tar.gz
cd "autodocksuite-${AUTODOCK_VERSION}-src"

# Build autogrid4
echo "Building autogrid4..."
cd src/autogrid
./configure --prefix="$INSTALL_DIR"
make
make install

# Build autodock4
echo "Building autodock4..."
cd ../autodock
./configure --prefix="$INSTALL_DIR"
make
make install

echo "================================================================"
echo "✓ AutoDock4 installation complete"
echo "================================================================"
echo "autogrid4 location: $INSTALL_DIR/bin/autogrid4"
echo "autodock4 location: $INSTALL_DIR/bin/autodock4"
echo ""
echo "Verifying installation..."
"$INSTALL_DIR/bin/autogrid4" -h 2>&1 | head -n 5
echo "================================================================"

# Add to PATH (already in load_modules.sh)
echo "Note: Make sure $INSTALL_DIR/bin is in your PATH"
echo "This should already be configured in load_modules.sh"
