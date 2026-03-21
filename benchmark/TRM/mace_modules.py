"""
MACE equivariant modules for the TRM backbone.

Provides MACE-based encoder and reasoning blocks that integrate with the TRM
recursive architecture. Features remain per-node throughout the entire recurrent
computation (no pseudo-nodes or fixed-size pooling in the reasoning core).

The key insight: reusing shared-weight MACE blocks across TRM's H×L recurrence
cycles yields arbitrarily high effective body-order correlations while keeping
the parameter count small. A standard 2-layer MACE captures 4-body correlations;
the same blocks applied recursively (e.g. 3 H-cycles × 2 L-cycles × 1 layer =
6 passes) capture much higher-order correlations with identical parameters.

Dependencies:
    mace-torch >= 0.3.15
    e3nn == 0.4.4 (pinned by mace-torch)

Note:
    e3nn 0.4.4 may require TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 with PyTorch ≥ 2.6.

References:
    Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields", NeurIPS 2022.
"""

import torch
from torch import nn
from e3nn import o3
from mace.modules.blocks import (
    EquivariantProductBasisBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)

# Registry of supported interaction block classes.
MACE_INTERACTION_CLASSES = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
}


def hidden_irreps_from_config(num_features, max_ell):
    """Build hidden irreps from feature count and max angular momentum.

    Args:
        num_features: feature channels per angular momentum order
        max_ell: maximum spherical harmonics order (0, 1, or 2)

    Returns:
        o3.Irreps, e.g. ``128x0e + 128x1o + 128x2e`` for num_features=128, max_ell=2

    The parity follows the standard convention: (-1)^L, giving 0e, 1o, 2e, ...
    """
    return o3.Irreps(
        [(num_features, (l, (-1) ** l)) for l in range(max_ell + 1)]
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class MACEEncoder(nn.Module):
    """MACE-style initial encoding for crystal graphs.

    Converts atomic numbers and graph topology into:
      - per-node equivariant features (scalar-only at this stage; higher-L
        components become active after the first MACE interaction block)
      - per-edge radial basis features (Bessel + polynomial cutoff)
      - per-edge angular features (real spherical harmonics of edge direction)

    Args:
        num_features: equivariant feature channels per angular momentum
        max_ell: maximum spherical harmonics order (0, 1, or 2)
        num_bessel: number of Bessel radial basis functions
        polynomial_cutoff: order of the polynomial envelope cutoff
        cutoff: radial cutoff distance (Angstroms) — must match graph construction
        max_atomic_num: maximum atomic number for one-hot encoding
    """

    def __init__(
        self,
        num_features,
        max_ell,
        num_bessel,
        polynomial_cutoff,
        cutoff,
        max_atomic_num,
    ):
        super().__init__()
        self.max_atomic_num = max_atomic_num
        self.num_elements = max_atomic_num + 1

        hidden_irreps = hidden_irreps_from_config(num_features, max_ell)
        node_attrs_irreps = o3.Irreps(f"{self.num_elements}x0e")

        # One-hot → equivariant node features (only L=0 populated initially,
        # since the o3.Linear cannot create higher-L from pure-scalar input).
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attrs_irreps,
            irreps_out=hidden_irreps,
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=cutoff,
            num_bessel=num_bessel,
            num_polynomial_cutoff=polynomial_cutoff,
        )

        self.sh_irreps = o3.Irreps.spherical_harmonics(max_ell)

    def forward(self, z, edge_index, edge_vectors, batch):
        """
        Args:
            z: (N,) atomic numbers (LongTensor)
            edge_index: (2, E) edge indices
            edge_vectors: (E, 3) displacement vectors from sender to receiver
            batch: (N,) graph-assignment vector

        Returns:
            node_feats:  (N, irreps_dim) initial node features
            node_attrs:  (N, num_elements) one-hot element identity
            edge_feats:  (E, num_bessel) radial basis features
            edge_attrs:  (E, sh_dim) spherical harmonic features
        """
        # One-hot element identity
        node_attrs = torch.nn.functional.one_hot(
            z.clamp(max=self.max_atomic_num),
            num_classes=self.num_elements,
        ).float()

        # Node embedding (higher-L channels are zero at this stage)
        node_feats = self.node_embedding(node_attrs)

        # Radial features (Bessel basis × polynomial cutoff envelope)
        edge_lengths = edge_vectors.norm(dim=-1, keepdim=True)  # (E, 1)
        edge_feats, _ = self.radial_embedding(
            edge_lengths, node_attrs, edge_index, z
        )

        # Angular features (real spherical harmonics of unit edge direction)
        edge_dirs = edge_vectors / edge_lengths.clamp(min=1e-8)
        edge_attrs = o3.spherical_harmonics(
            self.sh_irreps, edge_dirs, normalize=True, normalization="component"
        )

        return node_feats, node_attrs, edge_feats, edge_attrs


# ---------------------------------------------------------------------------
# Single MACE layer (interaction + product basis)
# ---------------------------------------------------------------------------


class MACEBlock(nn.Module):
    """One MACE layer: equivariant message passing + symmetric contraction.

    Composes an InteractionBlock (constructs many-body messages via tensor
    products of node features with spherical harmonics, aggregated over
    neighbors) with an EquivariantProductBasisBlock (applies the ACE symmetric
    contraction to build higher body-order features).

    Input and output are both flat per-node feature tensors of shape
    ``(N, irreps_dim)``, making this a drop-in equivariant graph convolution.

    Args:
        hidden_irreps: equivariant feature irreps
        sh_irreps: spherical harmonics irreps for edges
        num_elements: number of chemical element types
        num_bessel: radial basis size (determines edge_feats irreps)
        correlation: body order for symmetric contraction (2 → 3-body, 3 → 4-body)
        avg_num_neighbors: mean neighbor count, used for message normalization
        interaction_cls: name of the MACE interaction block class
        radial_mlp: hidden layer widths for the radial weight-generating MLP
    """

    def __init__(
        self,
        hidden_irreps,
        sh_irreps,
        num_elements,
        num_bessel,
        correlation,
        avg_num_neighbors,
        interaction_cls="RealAgnosticResidualInteractionBlock",
        radial_mlp=(64, 64, 64),
    ):
        super().__init__()

        node_attrs_irreps = o3.Irreps(f"{num_elements}x0e")
        edge_feats_irreps = o3.Irreps(f"{num_bessel}x0e")

        InteractionClass = MACE_INTERACTION_CLASSES[interaction_cls]

        self.interaction = InteractionClass(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=hidden_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=list(radial_mlp),
        )

        self.product = EquivariantProductBasisBlock(
            node_feats_irreps=hidden_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=True,
        )

    def forward(self, node_feats, node_attrs, edge_attrs, edge_feats, edge_index):
        """
        Args:
            node_feats: (N, irreps_dim) per-node equivariant features
            node_attrs: (N, num_elements) one-hot element identity
            edge_attrs: (E, sh_dim) spherical harmonics
            edge_feats: (E, num_bessel) radial basis
            edge_index: (2, E) edge indices

        Returns:
            (N, irreps_dim) updated per-node features
        """
        msg, sc = self.interaction(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
        )
        return self.product(msg, sc, node_attrs)


# ---------------------------------------------------------------------------
# Reasoning module (wraps multiple MACE blocks for the TRM recurrent core)
# ---------------------------------------------------------------------------


class MACEReasoningModule(nn.Module):
    """MACE-based reasoning module for the TRM recurrent core.

    Replaces :class:`ReasoningModule` when ``backbone="mace"``.  Graph structure
    is stored via :meth:`set_graph` once per batch, then reused across all
    H-cycles and L-cycles of the recurrence loop.

    The ``forward(hidden, injection)`` signature matches the standard
    ReasoningModule interface so the TRM recursive loop can call it uniformly.

    Args:
        num_features: equivariant feature channels per angular momentum
        max_ell: maximum spherical harmonics order
        num_bessel: Bessel radial basis size
        correlation: body order for symmetric contraction
        avg_num_neighbors: average neighbor count for message normalization
        interaction_cls: MACE interaction block class name
        radial_mlp: radial MLP hidden sizes
        num_layers: number of MACEBlock layers per reasoning pass
        max_atomic_num: maximum atomic number
    """

    def __init__(
        self,
        num_features,
        max_ell,
        num_bessel,
        correlation,
        avg_num_neighbors,
        interaction_cls,
        radial_mlp,
        num_layers,
        max_atomic_num,
    ):
        super().__init__()
        hidden_irreps = hidden_irreps_from_config(num_features, max_ell)
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_elements = max_atomic_num + 1

        self.blocks = nn.ModuleList([
            MACEBlock(
                hidden_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                num_elements=num_elements,
                num_bessel=num_bessel,
                correlation=correlation,
                avg_num_neighbors=avg_num_neighbors,
                interaction_cls=interaction_cls,
                radial_mlp=radial_mlp,
            )
            for _ in range(num_layers)
        ])
        self._graph = None

    def set_graph(self, node_attrs, edge_attrs, edge_feats, edge_index):
        """Store graph structure for the current batch.

        Must be called once before the recurrence loop begins.  The stored
        tensors are reused across all H-cycles and L-cycles (the graph topology
        does not change during recurrence — only the node features evolve).
        """
        self._graph = (node_attrs, edge_attrs, edge_feats, edge_index)

    def forward(self, hidden, injection):
        """Apply injection then MACE blocks to per-node features.

        Args:
            hidden: (N, irreps_dim) current per-node state
            injection: (N, irreps_dim) injected signal (pred + encoded)

        Returns:
            (N, irreps_dim) updated per-node features
        """
        hidden = hidden + injection
        node_attrs, edge_attrs, edge_feats, edge_index = self._graph
        for block in self.blocks:
            hidden = block(hidden, node_attrs, edge_attrs, edge_feats, edge_index)
        return hidden
