#!/bin/bash
#SBATCH --job-name=trm-tune
#SBATCH --output=tune_%j.log
#SBATCH --error=tune_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main
#
# Optuna hyperparameter tuning for TRM.
# 50 trials × 3-fold CV × 1000 epochs per backbone.
#
# Usage:
#   sbatch run_tune.sh
#   sbatch run_tune.sh --backbone transformer   # one backbone only

source ~/miniconda3/bin/activate
conda activate obelix

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

pip install -q optuna

echo "=== TRM Hyperparameter Tuning ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo ""

python3 -u tune.py "$@"

echo ""
echo "=== Sklearn baselines (RF + MLP, GridSearchCV) ==="
echo "Date: $(date)"
python3 -u ../tuning.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
