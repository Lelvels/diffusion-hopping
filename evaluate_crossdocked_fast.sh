#!/bin/bash

# Fast evaluation script for DiffHopp model on CrossDocked dataset
# Optimized for speed while maintaining statistical validity

MODEL_PATH="lightning_logs/sq59mors/checkpoints/final_model.ckpt"
DATASET="crossdocked"
RUN_ID="sq59mors_fast"
MODE="ligand_generation"  # Skip ground_truth and inpainting for speed
MOLECULES_PER_POCKET=10   # Reduced from 50 for faster evaluation
BATCH_SIZE=256            # Increased for better GPU utilization
LIMIT_SAMPLES=500         # Sample 500 test cases for good statistics

echo "=========================================="
echo "DiffHopp FAST Evaluation on CrossDocked"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "Molecules per pocket: $MOLECULES_PER_POCKET"
echo "Batch size: $BATCH_SIZE"
echo "Sample limit: $LIMIT_SAMPLES"
echo "=========================================="
echo ""
echo "Speed optimizations enabled:"
echo "  ✓ Ligand generation only (skip ground truth & inpainting)"
echo "  ✓ Reduced molecules per pocket (10 vs 50)"
echo "  ✓ Increased batch size (256 vs 128)"
echo "  ✓ Limited test samples (500 vs full set)"
echo ""
echo "To skip QVina scoring (major speedup):"
echo "  Add --only_generation flag to skip all metrics"
echo "=========================================="
echo ""

# Run evaluation
python evaluate_local.py $RUN_ID $DATASET \
    --checkpoint_path $MODEL_PATH \
    --mode $MODE \
    --molecules_per_pocket $MOLECULES_PER_POCKET \
    --batch_size $BATCH_SIZE \
    --limit_samples $LIMIT_SAMPLES

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: evaluation/$RUN_ID/$DATASET/"
echo "=========================================="
