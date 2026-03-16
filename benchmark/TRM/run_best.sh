#!/bin/bash
#SBATCH --job-name=trm-best
#SBATCH --output=best_%j.log
#SBATCH --error=best_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main
#
# Run TRM with best tuned hyperparameters (from Optuna) on full + CIF-only.
#
# Usage:
#   sbatch run_best.sh

source ~/miniconda3/bin/activate
conda activate obelix

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "=== TRM Best Hyperparameter Runs ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo ""

# Best transformer config (Optuna trial 1, CV MAE=0.951)
TRANS_ARGS="--backbone transformer --hidden_dim 128 --num_heads 8 --L_layers 3 --L_cycles 1 --H_cycles 4 --dropout 0.0 --lr 0.0007683818338208118 --wd 0.0002907055455650045 --batch_size 16 --warmup 150"

# Best MLP config (Optuna trial 34, CV MAE=1.098)
MLP_ARGS="--backbone mlp --hidden_dim 64 --num_heads 8 --L_layers 3 --L_cycles 3 --H_cycles 2 --dropout 0.1 --lr 0.0010249743813789322 --wd 0.00020360445373373938 --batch_size 16 --warmup 100"

# -- Full dataset --
echo "============================================================"
echo "=== Transformer (full dataset, 5-fold CV) ==="
echo "============================================================"
python3 -u train.py $TRANS_ARGS --cv --seed 42
echo ""

echo "============================================================"
echo "=== Transformer (full dataset, train/test) ==="
echo "============================================================"
python3 -u train.py $TRANS_ARGS --seed 42
echo ""

echo "============================================================"
echo "=== MLP (full dataset, 5-fold CV) ==="
echo "============================================================"
python3 -u train.py $MLP_ARGS --cv --seed 42
echo ""

echo "============================================================"
echo "=== MLP (full dataset, train/test) ==="
echo "============================================================"
python3 -u train.py $MLP_ARGS --seed 42
echo ""

# -- CIF-only --
echo "============================================================"
echo "=== Transformer (CIF-only, 5-fold CV) ==="
echo "============================================================"
python3 -u train.py $TRANS_ARGS --cif_only --cv --seed 42
echo ""

echo "============================================================"
echo "=== Transformer (CIF-only, train/test) ==="
echo "============================================================"
python3 -u train.py $TRANS_ARGS --cif_only --seed 42
echo ""

echo "============================================================"
echo "=== MLP (CIF-only, 5-fold CV) ==="
echo "============================================================"
python3 -u train.py $MLP_ARGS --cif_only --cv --seed 42
echo ""

echo "============================================================"
echo "=== MLP (CIF-only, train/test) ==="
echo "============================================================"
python3 -u train.py $MLP_ARGS --cif_only --seed 42
echo ""

echo "=== Done ==="
echo "Date: $(date)"
