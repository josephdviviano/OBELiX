"""
Optuna hyperparameter tuning for GNN-TRM backbones.

Searches over GNN architecture, attention pooling, TRM reasoning, and optimizer
hyperparameters using Bayesian optimization (TPE) with median pruning.
Runs 3-fold CV per trial, reports intermediate val_mae for epoch-level pruning.

Usage:
    python tune_gnn.py                              # tune both GNN backbones
    python tune_gnn.py --backbone gnn_transformer   # tune one backbone
    python tune_gnn.py --n_trials 2 --epochs 50     # quick smoke test
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
import torch
from torch import nn
from sklearn.model_selection import KFold

from model import TRM, TRMConfig
from graph_data import load_gnn_data, make_pyg_loader
from train import train_epoch, evaluate, make_scheduler


def objective(trial, backbone, data_list, args):
    """Single Optuna trial: 3-fold CV with epoch-level pruning."""
    # -- GNN-specific hyperparameters --
    gnn_conv_layers = trial.suggest_int("gnn_conv_layers", 2, 8)
    n_gaussians = trial.suggest_categorical("n_gaussians", [32, 64, 128])
    gnn_cutoff = trial.suggest_float("gnn_cutoff", 4.0, 8.0, step=1.0)
    pool_k = trial.suggest_categorical("pool_k", [16, 32, 64, 128])

    # -- TRM reasoning hyperparameters --
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    L_layers = trial.suggest_int("L_layers", 1, 3)
    L_cycles = trial.suggest_int("L_cycles", 1, 3)
    H_cycles = trial.suggest_int("H_cycles", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.05)

    # -- GCN reasoning hyperparameters (only for gnn_gcn backbone) --
    is_gcn = backbone == "gnn_gcn"
    gcn_adj_k = trial.suggest_categorical("gcn_adj_k", [2, 4, 8]) if is_gcn else 8
    gcn_drop_edge = trial.suggest_float("gcn_drop_edge", 0.1, 0.5, step=0.1) if is_gcn else 0.3
    gcn_gate_residual = trial.suggest_categorical("gcn_gate_residual", [True, False]) if is_gcn else True

    # -- Optimizer hyperparameters --
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    warmup = trial.suggest_int("warmup", 50, 200, step=50)

    # Ensure hidden_dim is divisible by num_heads (needed for attention pooling)
    if hidden_dim % num_heads != 0:
        raise optuna.TrialPruned()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TRMConfig(
        num_features=0,  # not used for GNN
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        L_layers=L_layers,
        L_cycles=L_cycles,
        H_cycles=H_cycles,
        backbone=backbone,
        dropout=dropout,
        gnn_conv_layers=gnn_conv_layers,
        n_gaussians=n_gaussians,
        gnn_cutoff=gnn_cutoff,
        pool_k=pool_k,
        gcn_adj_k=gcn_adj_k,
        gcn_drop_edge=gcn_drop_edge,
        gcn_gate_residual=gcn_gate_residual,
    )

    n = len(data_list)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_best_vals = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n))):
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        model = TRM(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = make_scheduler(optimizer, warmup, args.epochs)
        criterion = nn.L1Loss()

        train_loader = make_pyg_loader(train_data, batch_size, shuffle=True)
        val_loader = make_pyg_loader(val_data, batch_size, shuffle=False)

        best_val = float("inf")
        patience_counter = 0

        for epoch in range(args.epochs):
            train_epoch(model, train_loader, optimizer, criterion, device, is_gnn=True)
            val_loss = evaluate(model, val_loader, criterion, device, is_gnn=True)
            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Report to Optuna for pruning
            step = fold * args.epochs + epoch
            trial.report(val_loss, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if patience_counter >= 100:
                break

        fold_best_vals.append(best_val)

    mean_cv = np.mean(fold_best_vals)
    return mean_cv


def tune_backbone(backbone, data_list, args):
    """Run Optuna study for one GNN backbone."""
    print(f"\n{'='*60}")
    print(f"Tuning {backbone} backbone — {args.n_trials} trials, "
          f"3-fold CV, {args.epochs} epochs")
    print(f"{'='*60}\n")

    db_path = Path(__file__).parent / f"optuna_{backbone}.db"
    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name=f"trm_{backbone}",
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
        ),
    )

    study.optimize(
        lambda trial: objective(trial, backbone, data_list, args),
        n_trials=args.n_trials,
        show_progress_bar=False,
        catch=(Exception,),  # log failures but don't kill the study
    )

    print(f"\n--- Best {backbone} trial ---")
    print(f"  CV MAE: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Save all trials
    df = study.trials_dataframe()
    out_dir = Path(__file__).parent
    df.to_csv(out_dir / f"optuna_{backbone}.csv", index=False)

    # Save best config
    best = {"backbone": backbone, "cv_mae": study.best_value, **study.best_params}
    best_path = out_dir / f"best_{backbone}.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"  Saved best config to {best_path}")

    return study


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for GNN-TRM")
    parser.add_argument("--backbone", choices=["gnn_transformer", "gnn_mlp", "gnn_gcn"],
                        default=None, help="Tune one backbone (default: all three)")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--gnn_cutoff", type=float, default=5.0,
                        help="Cutoff for initial graph construction (tuning overrides per trial)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load all CIF-matching training data
    # Use a generous cutoff for graph construction; the tuner will vary the
    # Gaussian expansion cutoff but graphs are built once with this cutoff.
    train_list, _ = load_gnn_data(cutoff=8.0)

    backbones = [args.backbone] if args.backbone else ["gnn_transformer", "gnn_mlp", "gnn_gcn"]

    for backbone in backbones:
        tune_backbone(backbone, train_list, args)


if __name__ == "__main__":
    main()
