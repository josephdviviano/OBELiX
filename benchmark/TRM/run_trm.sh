#!/bin/bash
#SBATCH --job-name=trm-obelix
#SBATCH --output=trm_%j.log
#SBATCH --error=trm_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main
#
# TRM benchmark for OBELiX ionic conductivity prediction.
# Runs both transformer and MLP backbones with 5-fold CV.
#
# Usage:
#   sbatch run_trm.sh                          # SLURM cluster
#   bash run_trm.sh                            # any machine with CUDA/CPU
#   bash run_trm.sh --epochs 200 --H_cycles 5  # override defaults
#
# Estimated runtime: ~30 min on GPU, ~1 hr on CPU

source ~/miniconda3/bin/activate
conda activate obelix

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "=== TRM Benchmark for OBELiX ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# -- Transformer backbone (5-fold CV) ----------------------------------------
echo "=== Transformer backbone (5-fold CV) ==="
python3 -u train.py --backbone transformer --cv --seed 42 "$@"
echo ""

# -- MLP backbone (5-fold CV) ------------------------------------------------
echo "=== MLP backbone (5-fold CV) ==="
python3 -u train.py --backbone mlp --cv --seed 42 "$@"
echo ""

# -- Transformer backbone (train → test) -------------------------------------
echo "=== Transformer backbone (train/test evaluation) ==="
python3 -u train.py --backbone transformer --seed 42 "$@"
echo ""

# -- MLP backbone (train → test) ---------------------------------------------
echo "=== MLP backbone (train/test evaluation) ==="
python3 -u train.py --backbone mlp --seed 42 "$@"
echo ""

echo "=== Done ==="
