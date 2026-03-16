#!/bin/bash
#SBATCH --job-name=sklearn-baselines
#SBATCH --output=sklearn_%j.log
#SBATCH --error=sklearn_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=long-cpu
#
# Run sklearn baselines (RF + MLP GridSearchCV) only.

source ~/miniconda3/bin/activate
conda activate obelix

set -euo pipefail

SCRIPT_DIR="/home/mila/v/vivianoj/code/OBELiX/benchmark/TRM"
cd "$SCRIPT_DIR"

echo "=== Sklearn baselines (RF + MLP, GridSearchCV) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

python3 -u ../tuning.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
