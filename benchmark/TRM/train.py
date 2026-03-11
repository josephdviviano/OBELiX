"""
Training script for the Tiny Recursive Model (TRM) benchmark.

Reuses the existing data pipeline (parse.py, processed.csv, train/test splits)
and evaluates with the same protocol as RF/MLP benchmarks: MAE on log10(conductivity).

Usage:
    python train.py                          # train transformer backbone
    python train.py --backbone mlp           # train MLP backbone
    python train.py --cv                     # run 5-fold cross-validation
"""

import argparse
import sys
from pathlib import Path

# Add parent dir so we can import parse.py from benchmark/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from parse import read_xy  # noqa: E402
from model import TRM, TRMConfig  # noqa: E402

BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_PATH / "data"


# -- Data loading (mirrors tuning.py) ----------------------------------------

def load_data(cif_only=False, partial=True):
    """Load and preprocess data, matching the RF/MLP pipeline."""
    xy = read_xy(DATA_PATH / "processed.csv", partial=partial)

    train_idx = [line.strip() for line in open(DATA_PATH / "train_idx.csv")][1:]
    test_idx = [line.strip() for line in open(DATA_PATH / "test_idx.csv")][1:]

    # Log-transform target
    xy["Ionic conductivity (S cm-1)"] = np.log10(
        xy["Ionic conductivity (S cm-1)"]
    )

    train_xy = xy.loc[train_idx].copy()
    test_xy = xy.loc[test_idx].copy()

    # StandardScaler on lattice params (last 8 cols before target, minus CIF)
    scaler = StandardScaler()
    lattice_cols = ["Space group number", "a", "b", "c", "alpha", "beta", "gamma"]
    scaler.fit(train_xy[lattice_cols])
    train_xy[lattice_cols] = scaler.transform(train_xy[lattice_cols])
    test_xy[lattice_cols] = scaler.transform(test_xy[lattice_cols])

    if cif_only:
        mask = train_xy["CIF"].isin(["Match", "Close Match"])
        train_xy = train_xy[mask]

    train_xy = train_xy.drop("CIF", axis=1)
    test_xy_full = test_xy.copy()
    test_xy = test_xy.drop("CIF", axis=1)

    x_train = train_xy.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train = train_xy.iloc[:, -1].to_numpy(dtype=np.float32)
    x_test = test_xy.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test = test_xy.iloc[:, -1].to_numpy(dtype=np.float32)

    return x_train, y_train, x_test, y_test, test_xy_full


def make_loader(x, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# -- Training loop ------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total_loss += criterion(pred, yb).item() * xb.shape[0]
    return total_loss / len(loader.dataset)


def train_model(cfg, x_train, y_train, x_val, y_val, args, device, verbose=True):
    """Train a single model, return best validation MAE."""
    model = TRM(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.L1Loss()  # MAE, matching existing benchmark scoring

    train_loader = make_loader(x_train, y_train, args.batch_size)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = model.state_dict()
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}  train_mae={train_loss:.4f}  val_mae={val_loss:.4f}")

        if patience_counter >= args.patience:
            if verbose:
                print(f"  early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_val


# -- Cross-validation ---------------------------------------------------------

def cross_validate(cfg, x, y, args, device, n_folds=5):
    """5-fold CV, returns mean ± std MAE."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
        print(f"Fold {fold+1}/{n_folds}")
        _, val_mae = train_model(
            cfg, x[train_idx], y[train_idx], x[val_idx], y[val_idx],
            args, device, verbose=False,
        )
        scores.append(val_mae)
        print(f"  val_mae = {val_mae:.4f}")

    scores = np.array(scores)
    print(f"CV result: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TRM benchmark for OBELiX")
    parser.add_argument("--backbone", choices=["transformer", "mlp"], default="transformer")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--L_layers", type=int, default=2)
    parser.add_argument("--L_cycles", type=int, default=2)
    parser.add_argument("--H_cycles", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--cif_only", action="store_true")
    parser.add_argument("--no_partial", action="store_true")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_test, y_test, test_xy_full = load_data(
        cif_only=args.cif_only, partial=not args.no_partial,
    )

    num_features = x_train.shape[1]
    cfg = TRMConfig(
        num_features=num_features,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        L_layers=args.L_layers,
        L_cycles=args.L_cycles,
        H_cycles=args.H_cycles,
        backbone=args.backbone,
        dropout=args.dropout,
    )

    print(f"Config: backbone={cfg.backbone}, hidden={cfg.hidden_dim}, "
          f"heads={cfg.num_heads}, L_layers={cfg.L_layers}, "
          f"L_cycles={cfg.L_cycles}, H_cycles={cfg.H_cycles}")
    print(f"Data: {x_train.shape[0]} train, {x_test.shape[0]} test, "
          f"{num_features} features")

    if args.cv:
        cross_validate(cfg, x_train, y_train, args, device)
    else:
        print("\nTraining on full train set, evaluating on test set:")
        model, _ = train_model(
            cfg, x_train, y_train, x_test, y_test, args, device,
        )

        # Final test evaluation
        test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)
        test_mae = evaluate(model, test_loader, nn.L1Loss(), device)
        print(f"\nTest MAE (log10 scale): {test_mae:.4f}")

        # Also report CIF-only test MAE if running on whole dataset
        if not args.cif_only:
            cif_mask = test_xy_full["CIF"].isin(["Match", "Close Match"])
            if cif_mask.any():
                test_cif = test_xy_full[cif_mask].drop("CIF", axis=1)
                x_tc = test_cif.iloc[:, :-1].to_numpy(dtype=np.float32)
                y_tc = test_cif.iloc[:, -1].to_numpy(dtype=np.float32)
                cif_loader = make_loader(x_tc, y_tc, args.batch_size, shuffle=False)
                cif_mae = evaluate(model, cif_loader, nn.L1Loss(), device)
                print(f"Test MAE (CIF-only, log10 scale): {cif_mae:.4f}")


if __name__ == "__main__":
    main()
