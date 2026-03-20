#!/bin/bash
#SBATCH --job-name=trm-gnn-tune
#SBATCH --output=tune_gnn_%j.log
#SBATCH --error=tune_gnn_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=long
#
# Optuna hyperparameter tuning for GNN-TRM backbones.
# 50 trials × 3-fold CV × 1000 epochs per backbone.
#
# Usage:
#   sbatch run_tune_gnn.sh
#   sbatch run_tune_gnn.sh --backbone gnn_transformer   # one backbone only

source ~/miniconda3/bin/activate
conda activate obelix

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

pip install -q optuna torch_geometric

echo "=== GNN-TRM Hyperparameter Tuning ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo ""

python3 -u tune_gnn.py "$@" || echo "tune_gnn.py exited with code $?"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
