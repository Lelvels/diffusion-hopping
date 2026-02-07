#!/bin/bash
python evaluate_local_checkpoint.py gvp_conditional --scorer autodock_gpu --limit_samples 5 --molecules_per_pocket 2 --batch_size 2