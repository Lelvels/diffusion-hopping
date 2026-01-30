#!/bin/bash

# Evaluation script for DiffHopp model on CrossDocked dataset
# This script evaluates the trained model on the CrossDocked test set

MODEL_PATH="lightning_logs/sq59mors/checkpoints/final_model.ckpt"
DATASET="crossdocked"
RUN_ID="sq59mors"
MODE="all"  # Options: ground_truth, ligand_generation, inpaint_generation, all
MOLECULES_PER_POCKET=50    # Number of molecules to generate per protein pocket
BATCH_SIZE=512
# LIMIT_SAMPLES=100         # Limit samples for faster testing (remove for full evaluation)

echo "=========================================="
echo "DiffHopp Model Evaluation on CrossDocked"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "Molecules per pocket: $MOLECULES_PER_POCKET"
echo "Batch size: $BATCH_SIZE"
if [ -n "$LIMIT_SAMPLES" ]; then
    echo "Sample limit: $LIMIT_SAMPLES (for testing)"
fi
echo "=========================================="
echo ""

# Run evaluation
python evaluate_local.py $RUN_ID $DATASET \
    --checkpoint_path $MODEL_PATH \
    --mode $MODE \
    --molecules_per_pocket $MOLECULES_PER_POCKET \
    --batch_size $BATCH_SIZE \
    # --limit_samples $LIMIT_SAMPLES

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: evaluation/$RUN_ID/$DATASET/"
echo "=========================================="
