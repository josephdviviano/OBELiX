"""
Training script for the Tiny Recursive Model (TRM) benchmark.

Reuses the existing data pipeline (parse.py, processed.csv, train/test splits)
and evaluates with the same protocol as RF/MLP benchmarks: MAE on log10(conductivity).

Usage:
    python train.py                              # train transformer backbone
    python train.py --backbone mlp               # train MLP backbone
    python train.py --backbone gnn_transformer   # train GNN + transformer reasoning
    python train.py --backbone gnn_mlp           # train GNN + MLP reasoning
    python train.py --cv                         # run 5-fold cross-validation
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

def _unpack_batch(batch, device, is_gnn=False):
    """Unpack a batch into (input, target) depending on backbone type."""
    if is_gnn:
        batch = batch.to(device)
        # reshape(-1) instead of squeeze(-1) so single-sample batches
        # stay 1-d rather than collapsing to a 0-d scalar tensor
        return batch, batch.y.reshape(-1)
    else:
        xb, yb = batch
        return xb.to(device), yb.to(device)


def train_epoch(model, loader, optimizer, criterion, device, is_gnn=False):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        xb, yb = _unpack_batch(batch, device, is_gnn)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = yb.shape[0]
        total_loss += loss.item() * bs
        n_samples += bs
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_gnn=False):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        xb, yb = _unpack_batch(batch, device, is_gnn)
        pred = model(xb)
        bs = yb.shape[0]
        total_loss += criterion(pred, yb).item() * bs
        n_samples += bs
    return total_loss / n_samples


def make_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup followed by cosine decay to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(cfg, x_train, y_train, x_val, y_val, args, device, verbose=True):
    """Train for all epochs, return model at best validation MAE.

    For tabular backbones: x_train/x_val are numpy arrays, y_train/y_val are numpy arrays.
    For GNN/MACE backbones: x_train/x_val are lists of PyG Data, y_train/y_val are ignored (targets in Data.y).
    """
    is_gnn = cfg.backbone.startswith("gnn_") or cfg.backbone == "mace"
    model = TRM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = make_scheduler(optimizer, args.warmup, args.epochs)
    criterion = nn.L1Loss()

    if is_gnn:
        from graph_data import make_pyg_loader
        train_loader = make_pyg_loader(x_train, args.batch_size, shuffle=True)
        val_loader = make_pyg_loader(x_val, args.batch_size, shuffle=False)
    else:
        train_loader = make_loader(x_train, y_train, args.batch_size)
        val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = model.state_dict()
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, is_gnn)
        val_loss = evaluate(model, val_loader, criterion, device, is_gnn)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if verbose and (epoch + 1) % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:4d}  train_mae={train_loss:.4f}  val_mae={val_loss:.4f}  lr={lr:.2e}")

    if verbose:
        print(f"  best epoch: {best_epoch}  best val_mae: {best_val:.4f}")

    model.load_state_dict(best_state)
    return model, best_val


# -- Cross-validation ---------------------------------------------------------

def cross_validate(cfg, x, y, args, device, n_folds=5):
    """5-fold CV, returns mean ± std MAE.

    For GNN/MACE backbones, x is a list of PyG Data objects and y is None.
    """
    is_gnn = cfg.backbone.startswith("gnn_") or cfg.backbone == "mace"
    n = len(x) if is_gnn else x.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n))):
        print(f"Fold {fold+1}/{n_folds}")
        if is_gnn:
            x_tr = [x[i] for i in train_idx]
            x_va = [x[i] for i in val_idx]
            _, val_mae = train_model(cfg, x_tr, None, x_va, None, args, device, verbose=False)
        else:
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
    parser.add_argument("--backbone", choices=["transformer", "mlp", "gcn",
                                                "gnn_transformer", "gnn_mlp", "gnn_gcn",
                                                "mace"],
                        default="transformer")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--L_layers", type=int, default=2)
    parser.add_argument("--L_cycles", type=int, default=2)
    parser.add_argument("--H_cycles", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--cif_only", action="store_true")
    parser.add_argument("--no_partial", action="store_true")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation")
    parser.add_argument("--seed", type=int, default=42)
    # GNN-specific args
    parser.add_argument("--gnn_conv_layers", type=int, default=3)
    parser.add_argument("--n_gaussians", type=int, default=64)
    parser.add_argument("--gnn_cutoff", type=float, default=5.0)
    parser.add_argument("--pool_k", type=int, default=32)
    # GCN reasoning block args
    parser.add_argument("--gcn_adj_k", type=int, default=8)
    parser.add_argument("--gcn_drop_edge", type=float, default=0.3)
    parser.add_argument("--gcn_gate_residual", action="store_true", default=True)
    parser.add_argument("--no_gcn_gate_residual", dest="gcn_gate_residual", action="store_false")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Save trained model checkpoint (state_dict + config) to this path")
    # MACE-specific args
    parser.add_argument("--mace_max_ell", type=int, default=2,
                        help="Max spherical harmonics order (1 or 2)")
    parser.add_argument("--mace_correlation", type=int, default=2,
                        help="Body order for symmetric contraction (2=3-body, 3=4-body)")
    parser.add_argument("--mace_num_features", type=int, default=128,
                        help="Feature channels per angular momentum order")
    parser.add_argument("--mace_num_bessel", type=int, default=8,
                        help="Number of Bessel radial basis functions")
    parser.add_argument("--mace_interaction", type=str,
                        default="RealAgnosticResidualInteractionBlock",
                        help="MACE interaction block class name")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_mace = args.backbone == "mace"
    is_gnn = args.backbone.startswith("gnn_") or is_mace

    avg_nn = 10.0  # default; auto-computed below for MACE
    if is_gnn:
        from graph_data import load_gnn_data, make_pyg_loader
        train_data, test_data = load_gnn_data(
            cutoff=args.gnn_cutoff, include_vectors=is_mace,
        )
        num_features = 0  # not used for GNN/MACE
        if is_mace:
            from graph_data import compute_avg_num_neighbors
            avg_nn = compute_avg_num_neighbors(train_data)
            print(f"  avg_num_neighbors: {avg_nn:.1f}")
    else:
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
        gnn_conv_layers=args.gnn_conv_layers,
        n_gaussians=args.n_gaussians,
        gnn_cutoff=args.gnn_cutoff,
        pool_k=args.pool_k,
        gcn_adj_k=args.gcn_adj_k,
        gcn_drop_edge=args.gcn_drop_edge,
        gcn_gate_residual=args.gcn_gate_residual,
        mace_max_ell=args.mace_max_ell,
        mace_correlation=args.mace_correlation,
        mace_num_features=args.mace_num_features,
        mace_num_bessel=args.mace_num_bessel,
        mace_interaction=args.mace_interaction,
        mace_avg_num_neighbors=avg_nn,
    )

    print(f"Config: backbone={cfg.backbone}, hidden={cfg.hidden_dim}, "
          f"heads={cfg.num_heads}, L_layers={cfg.L_layers}, "
          f"L_cycles={cfg.L_cycles}, H_cycles={cfg.H_cycles}")
    if is_mace:
        print(f"MACE: max_ell={cfg.mace_max_ell}, correlation={cfg.mace_correlation}, "
              f"features={cfg.mace_num_features}, bessel={cfg.mace_num_bessel}")
        print(f"  interaction={cfg.mace_interaction}")
        print(f"Data: {len(train_data)} train, {len(test_data)} test graphs")
    elif is_gnn:
        print(f"GNN: conv_layers={cfg.gnn_conv_layers}, gaussians={cfg.n_gaussians}, "
              f"cutoff={cfg.gnn_cutoff}, pool_k={cfg.pool_k}")
        print(f"Data: {len(train_data)} train, {len(test_data)} test graphs")
    else:
        print(f"Data: {x_train.shape[0]} train, {x_test.shape[0]} test, "
              f"{num_features} features")

    if args.cv:
        if is_gnn:
            cross_validate(cfg, train_data, None, args, device)
        else:
            cross_validate(cfg, x_train, y_train, args, device)
    else:
        print("\nTraining on full train set, evaluating on test set:")
        if is_gnn:
            model, _ = train_model(cfg, train_data, None, test_data, None, args, device)
            test_loader = make_pyg_loader(test_data, args.batch_size, shuffle=False)
            test_mae = evaluate(model, test_loader, nn.L1Loss(), device, is_gnn=True)
        else:
            model, _ = train_model(cfg, x_train, y_train, x_test, y_test, args, device)
            test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)
            test_mae = evaluate(model, test_loader, nn.L1Loss(), device)

        print(f"\nTest MAE (log10 scale): {test_mae:.4f}")

        if args.save_path:
            from dataclasses import asdict
            torch.save({"config": asdict(cfg), "state_dict": model.state_dict()},
                        args.save_path)
            print(f"Saved checkpoint to {args.save_path}")

        # Also report CIF-only test MAE if running tabular on whole dataset
        if not is_gnn and not args.cif_only:
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
