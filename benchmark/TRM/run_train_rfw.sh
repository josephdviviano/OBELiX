#!/bin/bash
#SBATCH --job-name=rfw-eqv2
#SBATCH --output=rfw_%j.log
#SBATCH --error=rfw_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=long
#
# Train the Recurrent Foundation Wrapper (RFW) on ionic conductivity with
# a frozen EquiformerV3 backbone. Each Equiformer forward pass is expensive
# so keep epochs modest (50-100) and batch size small.
#
# Usage:
#   sbatch run_train_rfw.sh
#   sbatch run_train_rfw.sh --H_cycles 2 --L_cycles 1 --epochs 50
#   sbatch run_train_rfw.sh --use_checkpointing

source ~/miniconda3/bin/activate
conda activate obelix

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Ensure PyG extensions for EquiformerV3 (vendored code depends on them)
python -c "import torch_scatter, torch_sparse, torch_cluster" 2>/dev/null || {
    echo "Installing PyG extensions..."
    TORCH_V=$(python -c "import torch; print(f'{torch.__version__.split(\"+\")[0]}+cu{torch.version.cuda.replace(\".\", \"\")}')")
    pip install -q torch_scatter torch_sparse torch_cluster \
        -f "https://data.pyg.org/whl/torch-${TORCH_V}.html"
}

echo "=== RFW + EquiformerV3 Training ==="
echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "GPU:    $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo "Args:   $*"
echo ""

# Default config: Option B (TRM-faithful). Earlier runs disabled
# prediction_via_model to halve memory, but with the F1 refiner-gradient fix,
# Option A trains differently and is no longer the right "cheaper" fallback —
# it's now a distinct training mode. Default is the spiritually-correct Option B.
# Override per-job via `sbatch run_train_rfw.sh --no_prediction_via_model ...`.
python -u train_rfw.py \
    --H_cycles 3 \
    --L_cycles 2 \
    --epochs 100 \
    --warmup 6 \
    --batch_size 1 \
    --lora_rank 4 \
    --use_checkpointing \
    "$@" || echo "train_rfw.py exited with code $?"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
