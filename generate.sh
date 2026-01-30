#!/bin/bash

MODEL_PATH="lightning_logs/sq59mors/checkpoints/final_model.ckpt"
LIGAND_PATH="data/1a0q/ligand.sdf"
PROTEIN_PATH="data/1a0q/protein.pdb"
OUTPUT_DIR="results/1a0q_sq59mors"
NUM_SAMPLES=100

python generate_custom.py --input_molecule $LIGAND_PATH \
                        --input_protein $PROTEIN_PATH \
                        --output $OUTPUT_DIR \
                        --checkpoint_path $MODEL_PATH \
                        --num_samples $NUM_SAMPLES