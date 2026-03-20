"""
Post-hoc visualization of attention pooling weights for GNN-TRM models.

Given a trained checkpoint and a CIF file, shows which atoms the model
considers most important for its ionic conductivity prediction.

Usage:
    python visualize_attention.py --checkpoint model.pt --cif path/to/structure.cif
    python visualize_attention.py --checkpoint model.pt --sid SAMPLE_ID
    python visualize_attention.py --checkpoint model.pt --sid SAMPLE_ID --out importance.png
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import TRM, TRMConfig
from graph_data import structure_to_data

BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_PATH / "data"
CIF_DIR = DATA_PATH / "randomized_cifs"


def load_model(checkpoint_path, device="cpu"):
    """Load a trained GNN-TRM model from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = TRMConfig(**ckpt["config"])
    if not cfg.backbone.startswith("gnn_"):
        raise ValueError(f"Attention visualization requires a GNN backbone, got '{cfg.backbone}'")
    model = TRM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def get_atom_importance(model, data, device="cpu"):
    """Run inference on a single structure and extract per-atom importance.

    Returns:
        prediction: scalar predicted log10(conductivity)
        importance: (N_atoms,) array of attention-based importance scores
        attn_weights: (K, N_atoms) raw attention weight matrix
    """
    from torch_geometric.loader import DataLoader

    loader = DataLoader([data], batch_size=1)
    batch = next(iter(loader)).to(device)

    with torch.no_grad():
        prediction = model(batch).item()

    # Extract attention weights stored by AttentionPooling
    attn = model.attn_pool.last_attn_weights  # (1, K, max_nodes)
    n_atoms = model.attn_pool.last_node_counts[0].item()

    # Trim padding and squeeze batch dim
    attn_weights = attn[0, :, :n_atoms].cpu().numpy()  # (K, N_atoms)

    # Per-atom importance: sum of attention received across all K queries
    importance = attn_weights.sum(axis=0)  # (N_atoms,)
    # Normalize to [0, 1]
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min())

    return prediction, importance, attn_weights


def plot_structure_importance(structure, importance, prediction=None,
                              title=None, out_path=None):
    """3D scatter plot of crystal structure colored by attention importance.

    Args:
        structure: pymatgen Structure
        importance: (N_atoms,) array in [0, 1]
        prediction: optional predicted value to show in title
        title: optional custom title
        out_path: save to file instead of showing
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    coords = structure.cart_coords
    elements = [str(site.specie) for site in structure]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.YlOrRd

    # Size proportional to importance (min 30, max 300)
    sizes = 30 + 270 * importance

    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=importance, cmap=cmap, norm=norm,
        s=sizes, alpha=0.85, edgecolors="k", linewidths=0.5,
    )

    # Label atoms with element symbol
    for i, (coord, elem) in enumerate(zip(coords, elements)):
        ax.text(coord[0], coord[1], coord[2], f" {elem}",
                fontsize=7, alpha=0.7)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label="Attention importance")

    if title is None:
        title = "Atom importance (attention pooling weights)"
    if prediction is not None:
        title += f"\npred log10(σ) = {prediction:.3f}"
    ax.set_title(title)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()
    plt.close()


def print_importance_table(structure, importance):
    """Print a ranked table of atom importance scores."""
    elements = [str(site.specie) for site in structure]
    ranked = sorted(enumerate(importance), key=lambda x: -x[1])

    print(f"\n{'Rank':>4}  {'Idx':>4}  {'Elem':>5}  {'Importance':>10}  {'Frac coords'}")
    print("-" * 65)
    for rank, (idx, score) in enumerate(ranked, 1):
        fc = structure[idx].frac_coords
        print(f"{rank:4d}  {idx:4d}  {elements[idx]:>5}  {score:10.4f}  "
              f"({fc[0]:.3f}, {fc[1]:.3f}, {fc[2]:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Visualize GNN-TRM attention importance")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cif", help="Path to CIF file")
    group.add_argument("--sid", help="Sample ID (loads from data/randomized_cifs/)")
    parser.add_argument("--out", default=None, help="Output plot path (default: show interactively)")
    parser.add_argument("--cutoff", type=float, default=None,
                        help="Neighbor cutoff in Å (default: use checkpoint config)")
    parser.add_argument("--no_plot", action="store_true", help="Skip plot, only print table")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    print(f"Loaded {cfg.backbone} model (pool_k={cfg.pool_k}, hidden={cfg.hidden_dim})")

    cutoff = args.cutoff or cfg.gnn_cutoff

    # Load structure
    from pymatgen.core import Structure
    if args.cif:
        cif_path = Path(args.cif)
    else:
        cif_path = CIF_DIR / f"{args.sid}.cif"
    if not cif_path.exists():
        print(f"Error: CIF file not found: {cif_path}")
        sys.exit(1)

    structure = Structure.from_file(str(cif_path))
    print(f"Structure: {structure.composition.reduced_formula}, {len(structure)} atoms")

    # Build graph and run inference
    data = structure_to_data(structure, target=0.0, cutoff=cutoff)
    prediction, importance, attn_weights = get_atom_importance(model, data, device)

    print(f"Predicted log10(σ) = {prediction:.4f}")
    print_importance_table(structure, importance)

    if not args.no_plot:
        sid_label = args.sid or cif_path.stem
        plot_structure_importance(
            structure, importance, prediction=prediction,
            title=f"{sid_label} — {structure.composition.reduced_formula}",
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
