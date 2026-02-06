# Local Checkpoint Evaluation - Quick Start Guide

This guide shows you how to evaluate the diffusion-hopping model using **local checkpoint files** (no WandB required).

## Available Checkpoints

The `checkpoints/` directory contains 4 pre-trained models:

| Checkpoint | Description | Use Case |
|------------|-------------|----------|
| `gvp_conditional.ckpt` | **DiffHopp** | Scaffold hopping (default) |
| `gvp_unconditional.ckpt` | **DiffHopp** | Inpainting/repainting |
| `egnn_conditional.ckpt` | **DiffHopp-EGNN** | Scaffold hopping |
| `egnn_unconditional.ckpt` | **DiffHopp-EGNN** | Inpainting/repainting |

## Prerequisites

### 1. Activate Conda Environment

```bash
conda activate diffusion_hopping
# or
conda activate turbohopp
```

### 2. Ensure gnina is Available

The script will automatically check for gnina in:
- System PATH
- `/mnt/SSD3/cong_nguyen/Code/pharmacy_code/gnina/build/gnina`

**No WandB setup required!**

## Basic Usage

### Evaluate with Default Settings (500 samples)

```bash
./evaluate_local_checkpoint.sh gvp_conditional
```

This will:
- Load the DiffHopp model from local checkpoint
- Evaluate 500 samples from pdbbind_filtered
- Generate 10 molecules per pocket
- Use ligand generation mode

### Use Aliases for Convenience

```bash
# These are equivalent to gvp_conditional
./evaluate_local_checkpoint.sh gvp
./evaluate_local_checkpoint.sh diffhopp

# These are equivalent to egnn_conditional
./evaluate_local_checkpoint.sh egnn
./evaluate_local_checkpoint.sh diffhopp-egnn
```

## Advanced Usage

### Evaluate DiffHopp-EGNN Model

```bash
./evaluate_local_checkpoint.sh egnn_conditional
```

### Evaluate with Inpainting Mode

```bash
./evaluate_local_checkpoint.sh gvp_unconditional --mode inpaint_generation
```

### Customize Number of Samples

```bash
./evaluate_local_checkpoint.sh gvp_conditional --limit_samples 100
```

### Generate More Molecules per Pocket

```bash
./evaluate_local_checkpoint.sh gvp_conditional --molecules 20
```

### Adjust Batch Size (for memory constraints)

```bash
./evaluate_local_checkpoint.sh gvp_conditional --batch_size 16
```

### Evaluate All Modes

```bash
./evaluate_local_checkpoint.sh gvp_conditional --mode all
```

### Only Generate (Skip Evaluation)

```bash
./evaluate_local_checkpoint.sh gvp_conditional --only_generation
```

### Only Evaluate (Skip Generation)

```bash
./evaluate_local_checkpoint.sh gvp_conditional --only_evaluation
```

### Custom Output Directory

```bash
./evaluate_local_checkpoint.sh gvp_conditional --output_dir my_results
```

## Combined Options

```bash
./evaluate_local_checkpoint.sh gvp_conditional \
    --limit_samples 250 \
    --molecules 15 \
    --batch_size 16 \
    --output_dir evaluation_custom
```

## Direct Python Usage

```bash
python evaluate_local_checkpoint.py <checkpoint> [options]
```

### Options

```
Required:
  checkpoint              Checkpoint name or file

Options:
  --checkpoints_dir DIR   Checkpoints directory (default: checkpoints)
  --dataset NAME          Dataset name (default: pdbbind_filtered)
  --mode MODE             Evaluation mode (default: ligand_generation)
                          Choices: ground_truth, ligand_generation, 
                                   inpaint_generation, all
  --limit_samples N       Number of samples (default: 500)
  --molecules_per_pocket N  Molecules per pocket (default: 10)
  --batch_size N          Batch size (default: 32)
  --r N                   Resampling steps for inpainting (default: 10)
  --j N                   Jump length for inpainting (default: 10)
  --only_generation       Only generate, skip evaluation
  --only_evaluation       Only evaluate, skip generation
  --output_dir DIR        Output directory (default: evaluation_local)
```

## Output Structure

Results are saved in:

```
evaluation_local/
└── <checkpoint_name>/
    └── pdbbind_filtered/
        ├── molecules_ground_truth.pt
        ├── molecules_ligand_generation.pt
        ├── molecules_inpaint_generation.pt
        ├── results_ground_truth.pt
        ├── results_ground_truth.html
        ├── results_ligand_generation.pt
        ├── results_ligand_generation.html
        ├── results_inpaint_generation.pt
        ├── results_inpaint_generation.html
        └── summary.txt
```

## Examples

### Quick Test (100 samples, DiffHopp)
```bash
./evaluate_local_checkpoint.sh gvp --limit_samples 100 --molecules 5
```

### Full Evaluation (500 samples, all modes)
```bash
./evaluate_local_checkpoint.sh gvp --mode all
```

### Compare DiffHopp vs DiffHopp-EGNN
```bash
# Evaluate DiffHopp
./evaluate_local_checkpoint.sh gvp_conditional --limit_samples 500

# Evaluate DiffHopp-EGNN
./evaluate_local_checkpoint.sh egnn_conditional --limit_samples 500
```

### High-Quality Evaluation (50 molecules per pocket)
```bash
./evaluate_local_checkpoint.sh gvp --molecules 50 --batch_size 16
```

### Memory-Efficient Evaluation
```bash
./evaluate_local_checkpoint.sh gvp --batch_size 8 --molecules 5
```

### Inpainting Evaluation
```bash
./evaluate_local_checkpoint.sh gvp_unconditional \
    --mode inpaint_generation \
    --r 10 \
    --j 10
```

## Comparison: WandB vs Local Checkpoints

| Feature | WandB Version | Local Checkpoint Version |
|---------|---------------|--------------------------|
| Script | `evaluate_500_samples.py` | `evaluate_local_checkpoint.py` |
| Requires WandB | ✓ Yes | ✗ No |
| Requires run_id | ✓ Yes | ✗ No |
| Checkpoint source | Download from cloud | Load from local file |
| Speed | Slower (download) | Faster (local) |
| Offline capable | ✗ No | ✓ Yes |

## Troubleshooting

### "Checkpoint not found" Error
Make sure the checkpoint file exists:
```bash
ls -lh checkpoints/*.ckpt
```

### "gnina not found" Error
Ensure gnina is built and in PATH:
```bash
export PATH=$PATH:/path/to/gnina/build
```

### CUDA Out of Memory
Reduce batch size:
```bash
./evaluate_local_checkpoint.sh gvp --batch_size 8
```

### Wrong Conda Environment
Activate the correct environment:
```bash
conda activate diffusion_hopping
```

## Viewing Results

After evaluation completes:

1. **Summary Statistics**: `evaluation_local/<checkpoint>/pdbbind_filtered/summary.txt`
2. **HTML Visualizations**: Open `results_*.html` files in a browser
3. **Detailed Results**: Load `results_*.pt` files in Python

```python
import torch
results = torch.load('evaluation_local/gvp_conditional/pdbbind_filtered/results_ligand_generation.pt')
```

## Performance Tips

### For Faster Evaluation
- Use `--limit_samples 100` for quick tests
- Reduce `--molecules_per_pocket` to 5
- Use `--mode ligand_generation` (default, fastest)

### For Memory Constraints
- Reduce `--batch_size` to 8 or 16
- Reduce `--molecules_per_pocket`
- Process in stages with `--only_generation` then `--only_evaluation`

### For Comprehensive Analysis
- Use `--mode all` to test all evaluation modes
- Increase `--molecules_per_pocket` to 20 or 50
- Use full test set by increasing `--limit_samples`
