"""
Crystal graph construction and PyG dataset for the GNN-TRM backbone.

Converts CIF files to PyG Data objects with:
  - Node features: atomic numbers (LongTensor, for embedding lookup)
  - Edge index: from structure.get_all_neighbors(cutoff)
  - Edge attributes: interatomic distances (float)
  - Target: log10(ionic conductivity)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymatgen.core import Structure


BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_PATH / "data"
CIF_DIR = DATA_PATH / "randomized_cifs"


def structure_to_data(structure, target, cutoff=5.0):
    """Convert a pymatgen Structure to a PyG Data object.

    Handles disordered sites by using the majority species.
    """
    # Node features: atomic numbers
    atomic_nums = []
    for site in structure:
        if site.is_ordered:
            atomic_nums.append(site.specie.Z)
        else:
            # Use majority species for disordered sites
            species = site.species
            majority = max(species.keys(), key=lambda s: species[s])
            atomic_nums.append(majority.Z)
    z = torch.tensor(atomic_nums, dtype=torch.long)

    # Edges from neighbor list
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)
    src, dst, dists = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            # neighbor is (Site, distance, index, image)
            j = neighbor[2]
            d = neighbor[1]
            src.append(i)
            dst.append(j)
            dists.append(d)

    if len(src) == 0:
        # Fallback: self-loops if no neighbors found (shouldn't happen with reasonable cutoff)
        n = len(atomic_nums)
        src = list(range(n))
        dst = list(range(n))
        dists = [0.0] * n

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(dists, dtype=torch.float).unsqueeze(-1)

    y = torch.tensor([target], dtype=torch.float)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float).unsqueeze(0)

    return Data(z=z, pos=pos, cell=cell,
                edge_index=edge_index, edge_attr=edge_attr, y=y,
                num_nodes=len(atomic_nums))


def load_gnn_data(data_path=None, cutoff=5.0):
    """Load CIF-matching entries and build PyG datasets for train/test.

    Returns:
        (train_list, test_list): lists of PyG Data objects
    """
    if data_path is None:
        data_path = DATA_PATH

    csv_path = data_path / "processed.csv"
    df = pd.read_csv(csv_path, index_col="ID")

    # Filter to entries with CIF files
    cif_mask = df["CIF"].isin(["Match", "Close Match"])
    df_cif = df[cif_mask]

    # Load train/test splits
    train_idx = [line.strip() for line in open(data_path / "train_idx.csv")][1:]
    test_idx = [line.strip() for line in open(data_path / "test_idx.csv")][1:]

    # Filter to CIF-available entries
    train_ids = [idx for idx in train_idx if idx in df_cif.index]
    test_ids = [idx for idx in test_idx if idx in df_cif.index]

    cif_dir = data_path / "randomized_cifs"

    def build_dataset(ids):
        data_list = []
        skipped = 0
        for sid in ids:
            cif_path = cif_dir / f"{sid}.cif"
            if not cif_path.exists():
                skipped += 1
                continue
            try:
                structure = Structure.from_file(str(cif_path))
                target = np.log10(df_cif.loc[sid, "Ionic conductivity (S cm-1)"])
                data = structure_to_data(structure, target, cutoff=cutoff)
                data.sid = sid
                data_list.append(data)
            except Exception as e:
                print(f"Warning: skipping {sid}: {e}")
                skipped += 1
        if skipped > 0:
            print(f"  Skipped {skipped} entries")
        return data_list

    print(f"Building train dataset ({len(train_ids)} CIF-matching entries)...")
    train_list = build_dataset(train_ids)
    print(f"Building test dataset ({len(test_ids)} CIF-matching entries)...")
    test_list = build_dataset(test_ids)

    print(f"Loaded {len(train_list)} train, {len(test_list)} test graphs")
    return train_list, test_list


def make_pyg_loader(data_list, batch_size, shuffle=True):
    """Create a PyG DataLoader from a list of Data objects."""
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Smoke test: load a few CIFs and batch them
    train_list, test_list = load_gnn_data()
    if len(train_list) > 0:
        loader = make_pyg_loader(train_list[:5], batch_size=5)
        batch = next(iter(loader))
        print(f"\nSmoke test batch:")
        print(f"  z shape: {batch.z.shape}")
        print(f"  edge_index shape: {batch.edge_index.shape}")
        print(f"  edge_attr shape: {batch.edge_attr.shape}")
        print(f"  batch vector shape: {batch.batch.shape}")
        print(f"  y shape: {batch.y.shape}")
