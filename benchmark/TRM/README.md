# Tiny Recursive Model (TRM) Benchmark

A recursive transformer/MLP model for predicting ionic conductivity from tabular features (composition + lattice parameters).

## Architecture

The model treats each input feature (element counts, lattice parameters, space group number) as a position in a fixed-length sequence. Each position is projected into a hidden space via a per-position linear layer with learned positional embeddings.

The core uses a recursive reasoning structure:
- **H_cycles**: outer reasoning loops (only the final cycle receives gradient)
- **L_cycles**: inner loop iterations through the block stack
- **L_layers**: number of blocks in the stack

Two backbone variants:
- `transformer`: self-attention + FFN blocks
- `mlp`: FFN-only blocks (ablation to isolate the effect of attention)

## Usage

```bash
# Train with transformer backbone (default)
python train.py

# Train with MLP backbone
python train.py --backbone mlp

# 5-fold cross-validation
python train.py --cv

# Custom hyperparameters
python train.py --hidden_dim 128 --num_heads 8 --L_layers 3 --lr 0.0005

# CIF-only subset
python train.py --cif_only

# Round partial occupancies
python train.py --no_partial
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone` | transformer | "transformer" or "mlp" |
| `--hidden_dim` | 64 | Hidden dimension per position |
| `--num_heads` | 4 | Attention heads (transformer only) |
| `--L_layers` | 2 | Blocks per reasoning module |
| `--L_cycles` | 2 | Inner loop iterations |
| `--H_cycles` | 3 | Outer loop iterations |
| `--dropout` | 0.1 | Dropout rate |
| `--lr` | 1e-3 | Learning rate |
| `--wd` | 1e-4 | Weight decay |
| `--batch_size` | 32 | Batch size |
| `--epochs` | 500 | Max training epochs |
| `--patience` | 50 | Early stopping patience |
