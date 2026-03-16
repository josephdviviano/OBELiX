"""
Optuna hyperparameter tuning for TRM.

Searches over model architecture and optimizer hyperparameters using
Bayesian optimization (TPE) with median pruning. Runs 3-fold CV per trial,
reports intermediate val_mae for epoch-level pruning.

Usage:
    python tune.py                            # tune both backbones
    python tune.py --backbone transformer     # tune one backbone
    python tune.py --n_trials 30              # fewer trials
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from model import TRM, TRMConfig
from train import load_data, make_loader, train_epoch, evaluate, make_scheduler


def objective(trial, backbone, x, y, args):
    """Single Optuna trial: 3-fold CV with epoch-level pruning."""
    # -- Sample hyperparameters --
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    L_layers = trial.suggest_int("L_layers", 1, 3)
    L_cycles = trial.suggest_int("L_cycles", 1, 3)
    H_cycles = trial.suggest_int("H_cycles", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    warmup = trial.suggest_int("warmup", 50, 200, step=50)

    # Ensure hidden_dim is divisible by num_heads
    if hidden_dim % num_heads != 0:
        raise optuna.TrialPruned()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = x.shape[1]

    cfg = TRMConfig(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        L_layers=L_layers,
        L_cycles=L_cycles,
        H_cycles=H_cycles,
        backbone=backbone,
        dropout=dropout,
    )

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_best_vals = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        model = TRM(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = make_scheduler(optimizer, warmup, args.epochs)
        criterion = nn.L1Loss()

        train_loader = make_loader(x_train, y_train, batch_size)
        val_loader = make_loader(x_val, y_val, batch_size, shuffle=False)

        best_val = float("inf")

        for epoch in range(args.epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss

            # Report to Optuna for pruning every 50 epochs
            # Use a global step so pruning works across folds
            step = fold * args.epochs + epoch
            trial.report(val_loss, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        fold_best_vals.append(best_val)

    mean_cv = np.mean(fold_best_vals)
    return mean_cv


def tune_backbone(backbone, x, y, args):
    """Run Optuna study for one backbone."""
    print(f"\n{'='*60}")
    print(f"Tuning {backbone} backbone — {args.n_trials} trials, "
          f"3-fold CV, {args.epochs} epochs")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=200,  # don't prune before epoch 200
        ),
        study_name=f"trm_{backbone}",
    )

    study.optimize(
        lambda trial: objective(trial, backbone, x, y, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
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
    parser = argparse.ArgumentParser(description="Optuna tuning for TRM")
    parser.add_argument("--backbone", choices=["transformer", "mlp"],
                        default=None, help="Tune one backbone (default: both)")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--cif_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train, y_train, _, _, _ = load_data(cif_only=args.cif_only)

    backbones = [args.backbone] if args.backbone else ["transformer", "mlp"]

    for backbone in backbones:
        tune_backbone(backbone, x_train, y_train, args)


if __name__ == "__main__":
    main()
