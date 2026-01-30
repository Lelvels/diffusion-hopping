#!/bin/bash

MODEL_PATH="/home/labntc/Videos/Screencasts/Code/diffusion-hopping/lightning_logs/pm37nfjh/checkpoints/epoch=25-step=8970-val_loss=0.145.ckpt"
LIGAND_PATH="data/custom_data/4WKQ_ligand.pdb"
PROTEIN_PATH="data/custom_data/4WKQ.pdb"
OUTPUT_DIR="results/4WKQ"
NUM_SAMPLES=1000

python generate_custom.py --input_molecule $LIGAND_PATH \
                        --input_protein $PROTEIN_PATH \
                        --output $OUTPUT_DIR \
                        --checkpoint_path $MODEL_PATH \
                        --num_samples $NUM_SAMPLES