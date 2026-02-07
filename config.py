"""
Configuration file for data paths and other constants.

Environment Variable Configuration:
------------------------------------
Set DIFFUSION_HOPPING_DATA_ROOT to override the default data path:

For HPC (recommended):
    export DIFFUSION_HOPPING_DATA_ROOT="/net/scratch/hscra/plgrid/plgkietho/maidung/Code/pharma_res/data"

For local development:
    export DIFFUSION_HOPPING_DATA_ROOT="./data"

Or modify the default value in this file directly.
"""
import os
from pathlib import Path

# Default data root directory
# Modify this path based on your environment or set DIFFUSION_HOPPING_DATA_ROOT env variable
DEFAULT_DATA_ROOT = "/net/scratch/hscra/plgrid/plgkietho/maidung/Code/pharma_res/data"

DATA_ROOT = os.environ.get("DIFFUSION_HOPPING_DATA_ROOT", DEFAULT_DATA_ROOT)

# Convert to Path object for easier manipulation
DATA_ROOT = Path(DATA_ROOT)
