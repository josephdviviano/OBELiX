"""
Training script for the Recurrent Foundation Wrapper (RFW) on ionic
conductivity prediction, using a frozen EquiformerV3 backbone.

Usage:
    python train_rfw.py                         # defaults (H=3, L=2)
    python train_rfw.py --H_cycles 2 --L_cycles 1 --epochs 50
    python train_rfw.py --cv                    # 5-fold CV
    python train_rfw.py --use_checkpointing     # grad checkpoint on L-cycles

Because each Equiformer forward pass is ~0.5-2s on V100, training is
slow (minutes per epoch, not seconds). Plan accordingly.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

from graph_data import load_gnn_data
from rfw.wrapper import RecurrentFoundationWrapper, RFWConfig
from rfw_eqv2.adapter import EqV2Adapter, _load_equiformer_v3

BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_PATH / "data"


# -- Training loop ------------------------------------------------------------

def make_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup → cosine decay to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, criterion, device,
                grad_clip: float = 1.0):
    model.train()
    total_loss = 0.0
    n = 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for batch in loader:
        batch = batch.to(device)
        target = batch.y.reshape(-1)
        pred = model(batch)
        loss = criterion(pred, target)

        # Skip bad batches rather than poison training with a NaN update
        if not torch.isfinite(loss):
            print(f"  [warn] non-finite loss={loss.item()}; skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        bs = target.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        target = batch.y.reshape(-1)
        pred = model(batch)
        bs = target.shape[0]
        total_loss += criterion(pred, target).item() * bs
        n += bs
    return total_loss / n


def build_model(args, device):
    """Build a frozen Equiformer wrapped by RFW."""
    print(f"Loading EquiformerV3 ({args.checkpoint})...")
    frozen = _load_equiformer_v3(args.checkpoint, device=device)
    print(f"  Frozen params: {sum(p.numel() for p in frozen.parameters()):,}")

    adapter = EqV2Adapter(frozen, lora_rank=args.lora_rank)
    cfg = RFWConfig(
        state_dim=args.state_dim,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        lora_rank=args.lora_rank,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_ffn_expansion=args.ffn_expansion,
        dropout=args.dropout,
        prediction_via_model=not args.no_prediction_via_model,
        use_l_cycle_checkpointing=args.use_checkpointing,
    )
    wrapper = RecurrentFoundationWrapper(
        frozen, adapter, feature_dim=frozen.num_channels, cfg=cfg,
    ).to(device)
    print(wrapper.param_summary())
    return wrapper, cfg


def train_model(args, train_data, val_data, device, verbose=True):
    wrapper, cfg = build_model(args, device)

    # Only pass trainable params to the optimizer
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = make_scheduler(optimizer, args.warmup, args.epochs)
    criterion = nn.L1Loss()

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(wrapper, train_loader, optimizer, criterion,
                                 device, grad_clip=args.grad_clip)
        val_loss = evaluate(wrapper, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            # Keep on device — we reload onto the same device below
            best_state = {
                k: v.detach().clone()
                for k, v in wrapper.state_dict().items()
                if "frozen_model" not in k  # don't save the frozen backbone
            }
            best_epoch = epoch + 1

        if verbose and (epoch + 1) % 5 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:4d}  "
                  f"train_mae={train_loss:.4f}  val_mae={val_loss:.4f}  "
                  f"lr={lr:.2e}")

    if verbose:
        print(f"  best epoch {best_epoch}  best val_mae={best_val:.4f}")

    if best_state is not None:
        wrapper.load_state_dict(best_state, strict=False)
    return wrapper, best_val


def cross_validate(args, data, device, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(data)))):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        tr = [data[i] for i in tr_idx]
        va = [data[i] for i in va_idx]
        _, val_mae = train_model(args, tr, va, device, verbose=False)
        scores.append(val_mae)
        print(f"  val_mae = {val_mae:.4f}")
    scores = np.array(scores)
    print(f"\nCV result: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RFW + EquiformerV3 training")
    # Architecture
    parser.add_argument("--state_dim", type=int, default=64)
    parser.add_argument("--H_cycles", type=int, default=3)
    parser.add_argument("--L_cycles", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="LoRA inner rank. 4 → ~1.85M trainable, 8 → ~3.7M, 16 → ~7.4M")
    parser.add_argument("--cross_attn_heads", type=int, default=4)
    parser.add_argument("--ffn_expansion", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no_prediction_via_model", action="store_true",
                        help="Use cheap MLP for P update instead of a 2nd model pass")
    parser.add_argument("--use_checkpointing", action="store_true",
                        help="Gradient checkpoint on L-cycle refiner")
    # Pretrained
    parser.add_argument("--checkpoint", type=str,
                        default="omat24-mptrj-salex_gradient.pt",
                        help="HuggingFace checkpoint filename")
    # Data
    parser.add_argument("--cutoff", type=float, default=12.0,
                        help="Cutoff for graph construction (Å)")
    # Optimization
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Small by default because each example is expensive "
                             "and Equiformer tensors are O(N_atoms × coeffs × channels)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient norm clipping (0 to disable)")
    # Driver
    parser.add_argument("--cv", action="store_true", help="Run 5-fold CV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CIF graphs
    print("\nLoading CIFs...")
    train_data, test_data = load_gnn_data(cutoff=args.cutoff)
    print(f"Train: {len(train_data)}  Test: {len(test_data)}")

    print(f"\nRFW config: H={args.H_cycles} L={args.L_cycles} "
          f"D={args.state_dim} rank={args.lora_rank} "
          f"P_via_model={not args.no_prediction_via_model} "
          f"checkpoint={args.use_checkpointing}")

    if args.cv:
        cross_validate(args, train_data, device)
    else:
        print("\nTraining on full train set, evaluating on test set:")
        wrapper, _ = train_model(args, train_data, test_data, device)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        test_mae = evaluate(wrapper, test_loader, nn.L1Loss(), device)
        print(f"\nTest MAE (log10 scale): {test_mae:.4f}")

        if args.save_path:
            trainable_state = {
                k: v.cpu() for k, v in wrapper.state_dict().items()
                if "frozen_model" not in k
            }
            torch.save({
                "config": vars(args),
                "trainable_state": trainable_state,
            }, args.save_path)
            print(f"Saved trainable state to {args.save_path}")


if __name__ == "__main__":
    main()
