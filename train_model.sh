#!/bin/bash

# ============================================================================
# Training Configuration
# ============================================================================

# Dataset selection
DATASET="crossdocked"  # Options: crossdocked, pdbbind_filtered

# Model architecture
ARCHITECTURE="gvp"  # Options: gvp, egnn

# Training parameters
SEED=42
BATCH_SIZE=64
NUM_STEPS=10000
LEARNING_RATE=0.0001

# Diffusion parameters
T=500  # Number of diffusion timesteps

# Model architecture parameters
NUM_LAYERS=6        # 6 for both
HIDDEN_FEATURES=256 # 256 for both
JOINT_FEATURES=256  # 128 for pdbbind_filtered, 256 for crossdocked
EDGE_CUTOFF="(None, 5, 5)"  # Format: (ligand-ligand, ligand-protein, protein-protein)

# Additional options
ATTENTION="true"  # Options: true, false
CONDITION_ON_FG="false"  # Condition on functional groups: true, false

# ============================================================================
# Run Training
# ============================================================================

python train_model.py \
    --dataset_name $DATASET \
    --architecture $ARCHITECTURE \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --lr $LEARNING_RATE \
    --T $T \
    --num_layers $NUM_LAYERS \
    --hidden_features $HIDDEN_FEATURES \
    --joint_features $JOINT_FEATURES \
    --edge_cutoff "$EDGE_CUTOFF" \
    --attention $ATTENTION \
    --condition_on_fg $CONDITION_ON_FG
