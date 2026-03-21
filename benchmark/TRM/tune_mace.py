"""
Optuna hyperparameter tuning for the MACE-TRM backbone.

Searches over MACE architecture (max_ell, correlation, feature channels,
interaction block variant), TRM reasoning (L_layers, L_cycles, H_cycles),
and optimizer hyperparameters using Bayesian optimization (TPE) with median
pruning.  Runs 3-fold CV per trial.

Usage:
    python tune_mace.py                            # full tuning
    python tune_mace.py --n_trials 2 --epochs 50   # quick smoke test
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
from graph_data import load_gnn_data, make_pyg_loader, compute_avg_num_neighbors
from train import train_epoch, evaluate, make_scheduler


def objective(trial, data_list, avg_num_neighbors, args):
    """Single Optuna trial: 3-fold CV with epoch-level pruning."""
    # -- MACE architecture --
    mace_max_ell = trial.suggest_categorical("mace_max_ell", [1, 2])
    mace_correlation = trial.suggest_categorical("mace_correlation", [2, 3])
    mace_num_features = trial.suggest_categorical("mace_num_features", [32, 64, 128])
    mace_num_bessel = trial.suggest_categorical("mace_num_bessel", [4, 8, 16])
    mace_interaction = trial.suggest_categorical("mace_interaction", [
        "RealAgnosticResidualInteractionBlock",
        "RealAgnosticAttResidualInteractionBlock",
    ])

    # -- Graph construction --
    gnn_cutoff = trial.suggest_float("gnn_cutoff", 4.0, 8.0, step=1.0)

    # -- TRM reasoning --
    L_layers = trial.suggest_int("L_layers", 1, 2)
    L_cycles = trial.suggest_int("L_cycles", 1, 3)
    H_cycles = trial.suggest_int("H_cycles", 2, 4)

    # -- Optimizer --
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    warmup = trial.suggest_int("warmup", 50, 200, step=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TRMConfig(
        num_features=0,       # not used for MACE
        hidden_dim=64,        # not used for MACE
        backbone="mace",
        L_layers=L_layers,
        L_cycles=L_cycles,
        H_cycles=H_cycles,
        gnn_cutoff=gnn_cutoff,
        mace_max_ell=mace_max_ell,
        mace_correlation=mace_correlation,
        mace_num_features=mace_num_features,
        mace_num_bessel=mace_num_bessel,
        mace_interaction=mace_interaction,
        mace_avg_num_neighbors=avg_num_neighbors,
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
            train_epoch(model, train_loader, optimizer, criterion, device,
                        is_gnn=True)
            val_loss = evaluate(model, val_loader, criterion, device,
                                is_gnn=True)
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

    return np.mean(fold_best_vals)


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for MACE-TRM")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load graph data with edge vectors for MACE.
    # Use a generous cutoff for graph construction; the tuner varies the
    # effective cutoff per trial via the radial embedding, but graphs are
    # built once at this upper bound.
    print("Loading graph data with edge vectors...")
    train_list, _ = load_gnn_data(cutoff=8.0, include_vectors=True)
    avg_nn = compute_avg_num_neighbors(train_list)
    print(f"  {len(train_list)} training graphs, avg_num_neighbors={avg_nn:.1f}")

    print(f"\n{'=' * 60}")
    print(f"Tuning MACE backbone — {args.n_trials} trials, "
          f"3-fold CV, {args.epochs} epochs")
    print(f"{'=' * 60}\n")

    db_path = Path(__file__).parent / "optuna_mace.db"
    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name="trm_mace",
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=100,
        ),
    )

    study.optimize(
        lambda trial: objective(trial, train_list, avg_nn, args),
        n_trials=args.n_trials,
        show_progress_bar=False,
        catch=(Exception,),  # log failures but keep the study alive
    )

    print(f"\n--- Best MACE trial ---")
    print(f"  CV MAE: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Save all trials
    df = study.trials_dataframe()
    out_dir = Path(__file__).parent
    df.to_csv(out_dir / "optuna_mace.csv", index=False)

    # Save best config
    best = {"backbone": "mace", "cv_mae": study.best_value, **study.best_params}
    best_path = out_dir / "best_mace.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"  Saved best config to {best_path}")


if __name__ == "__main__":
    main()
