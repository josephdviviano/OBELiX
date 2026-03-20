#!/bin/bash
#SBATCH --job-name=trm-best-gnn
#SBATCH --output=best_gnn_%j.log
#SBATCH --error=best_gnn_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=main
#
# Evaluate TRM GNN backbones (best Optuna configs) on train/test split.
#
# Usage:
#   sbatch run_best_gnn.sh

source ~/miniconda3/bin/activate
conda activate obelix

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

mkdir -p checkpoints
pip install -q torch_geometric

echo "=== TRM GNN — Test Evaluation ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")')"
echo ""

# Best gnn_transformer config (Optuna trial 47, CV MAE=1.264)
GNN_TRANS_ARGS="--backbone gnn_transformer --hidden_dim 128 --num_heads 2 --L_layers 3 --L_cycles 2 --H_cycles 3 --dropout 0.1 --lr 0.00988636022365111 --wd 0.006502472607464972 --batch_size 32 --warmup 50 --gnn_conv_layers 2 --n_gaussians 32 --gnn_cutoff 7.0 --pool_k 64"

# Best gnn_mlp config (Optuna trial 35, CV MAE=1.330)
GNN_MLP_ARGS="--backbone gnn_mlp --hidden_dim 128 --num_heads 8 --L_layers 3 --L_cycles 1 --H_cycles 2 --dropout 0.2 --lr 0.0019710965078528413 --wd 0.03741622309462485 --batch_size 32 --warmup 100 --gnn_conv_layers 4 --n_gaussians 32 --gnn_cutoff 5.0 --pool_k 32"

# Best gnn_gcn config (Optuna trial 27, CV MAE=1.293)
GNN_GCN_ARGS="--backbone gnn_gcn --hidden_dim 128 --num_heads 8 --L_layers 3 --L_cycles 3 --H_cycles 2 --dropout 0.4 --lr 0.0006002255062417215 --wd 0.00020633626432731784 --batch_size 16 --warmup 50 --gnn_conv_layers 5 --n_gaussians 32 --gnn_cutoff 6.0 --pool_k 128 --gcn_adj_k 8 --gcn_drop_edge 0.3 --no_gcn_gate_residual"

# -- GNN Transformer --
echo "============================================================"
echo "=== GNN Transformer (train/test) ==="
echo "============================================================"
python3 -u train.py $GNN_TRANS_ARGS --save_path checkpoints/gnn_transformer.pt --seed 42
echo ""

# -- GNN MLP --
echo "============================================================"
echo "=== GNN MLP (train/test) ==="
echo "============================================================"
python3 -u train.py $GNN_MLP_ARGS --save_path checkpoints/gnn_mlp.pt --seed 42
echo ""

# -- GNN GCN --
echo "============================================================"
echo "=== GNN GCN (train/test) ==="
echo "============================================================"
python3 -u train.py $GNN_GCN_ARGS --save_path checkpoints/gnn_gcn.pt --seed 42
echo ""

echo "=== Done ==="
echo "Date: $(date)"
