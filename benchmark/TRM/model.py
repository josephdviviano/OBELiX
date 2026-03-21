"""
Tiny Recursive Model (TRM) for ionic conductivity prediction.

Backbone variants:
  - "transformer": self-attention + FFN blocks (learns inter-feature relationships)
  - "mlp": FFN-only blocks (recursive baseline without attention)
  - "gcn": tabular input → GCN reasoning blocks (learnable sparse adjacency)
  - "gnn_transformer": GNN encoder + attention pooling → transformer reasoning
  - "gnn_mlp": GNN encoder + attention pooling → MLP reasoning
  - "gnn_gcn": GNN encoder + attention pooling → GCN reasoning blocks
  - "mace": MACE equivariant encoder + per-node MACE reasoning blocks

All share the same recursive structure: H outer cycles × L inner cycles through
a stack of L_layers blocks, with persistent memory (prediction + reasoning state).
Only the final H cycle receives gradient.

The MACE backbone uses equivariant message passing (spherical harmonics, tensor
products, symmetric contractions) throughout the recurrent core.  Features remain
per-node with no pseudo-node aggregation — batching uses PyG's native flat
concatenation.  Requires mace-torch and e3nn.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


# -- Config ------------------------------------------------------------------

@dataclass
class TRMConfig:
    num_features: int  # number of input feature slots (ignored for GNN backbones)
    hidden_dim: int = 64  # hidden dimension per position
    num_heads: int = 4  # attention heads (transformer backbone only)
    ffn_expansion: float = 2.0  # FFN hidden size = hidden_dim * expansion
    L_layers: int = 2  # blocks per reasoning module
    L_cycles: int = 2  # inner loop iterations
    H_cycles: int = 3  # outer loop iterations (only last gets grad)
    backbone: Literal["transformer", "mlp", "gcn", "gnn_transformer", "gnn_mlp", "gnn_gcn", "mace"] = "transformer"
    dropout: float = 0.1
    # GCN reasoning block parameters
    gcn_adj_k: int = 8           # top-k neighbors in learnable adjacency
    gcn_drop_edge: float = 0.3   # DropEdge probability during training
    gcn_gate_residual: bool = True  # gated residual vs simple residual
    # GNN-specific parameters
    gnn_conv_layers: int = 3
    n_gaussians: int = 64
    gnn_cutoff: float = 5.0
    pool_k: int = 32  # number of attention pooling queries
    max_atomic_num: int = 100
    # MACE parameters (used when backbone="mace")
    mace_max_ell: int = 2               # max spherical harmonics order (0, 1, or 2)
    mace_correlation: int = 2           # body order for symmetric contraction
    mace_num_features: int = 128        # feature channels per angular momentum
    mace_num_bessel: int = 8            # Bessel radial basis functions
    mace_polynomial_cutoff: int = 5     # polynomial cutoff order
    mace_radial_mlp: tuple = (64, 64, 64)  # radial MLP hidden sizes
    mace_interaction: str = "RealAgnosticResidualInteractionBlock"
    mace_avg_num_neighbors: float = 10.0  # average neighbors per node


# -- Building blocks ---------------------------------------------------------

class FFN(nn.Module):
    """Two-layer feed-forward with GELU activation."""
    def __init__(self, dim: int, expansion: float, dropout: float):
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → MHA → residual → LayerNorm → FFN → residual."""
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = FFN(cfg.hidden_dim, cfg.ffn_expansion, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class MLPBlock(nn.Module):
    """Pre-norm MLP block: LayerNorm → FFN → residual."""
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = FFN(cfg.hidden_dim, cfg.ffn_expansion, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class LearnableAdjacency(nn.Module):
    """Learnable sparse adjacency matrix shared across all GCN layers."""
    def __init__(self, S: int, k: int, drop_edge: float):
        super().__init__()
        self.adj_logits = nn.Parameter(torch.randn(S, S) * 0.02)
        self.k = min(k, S)
        self.drop_edge = drop_edge

    def forward(self, training: bool) -> torch.Tensor:
        # Row-wise softmax → soft adjacency
        A = torch.softmax(self.adj_logits, dim=-1)
        # Top-k sparsification
        topk_vals, topk_idx = A.topk(self.k, dim=-1)
        sparse_A = torch.zeros_like(A)
        sparse_A.scatter_(-1, topk_idx, topk_vals)
        # DropEdge during training
        if training and self.drop_edge > 0:
            mask = torch.bernoulli(torch.full_like(sparse_A, 1.0 - self.drop_edge))
            sparse_A = sparse_A * mask
        # Re-normalize rows
        row_sum = sparse_A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return sparse_A / row_sum


class GCNBlock(nn.Module):
    """GCN reasoning block with gated residual and FFN sub-layer.

    forward(x):
        h = LayerNorm(x)
        h = ReLU(A_sparse @ h @ W)       # GCN aggregation
        gate = sigmoid(Linear([x; h]))    # gated residual
        x = x + gate * h
        x = x + FFN(LayerNorm(x))         # standard FFN sub-layer
        return x
    """
    def __init__(self, cfg: TRMConfig, adj: LearnableAdjacency):
        super().__init__()
        D = cfg.hidden_dim
        self.adj = adj
        self.norm1 = nn.LayerNorm(D)
        self.W = nn.Linear(D, D)
        self.gate_residual = cfg.gcn_gate_residual
        if self.gate_residual:
            self.gate_proj = nn.Linear(2 * D, D)
        self.norm2 = nn.LayerNorm(D)
        self.ffn = FFN(D, cfg.ffn_expansion, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.adj(self.training)              # (S, S)
        h = self.norm1(x)                        # (B, S, D)
        h = torch.relu(torch.matmul(A, h))       # A @ h: (B, S, D)
        h = self.W(h)                            # linear transform
        if self.gate_residual:
            gate = torch.sigmoid(self.gate_proj(torch.cat([x, h], dim=-1)))
            x = x + gate * h
        else:
            x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


# -- Recursive core ----------------------------------------------------------

class ReasoningModule(nn.Module):
    """Stack of blocks with injection: hidden = hidden + injection, then L_layers blocks."""
    def __init__(self, cfg: TRMConfig, S: int):
        super().__init__()
        use_gcn = cfg.backbone in ("gcn", "gnn_gcn")
        use_attn = cfg.backbone in ("transformer", "gnn_transformer")

        if use_gcn:
            self.adj = LearnableAdjacency(S, cfg.gcn_adj_k, cfg.gcn_drop_edge)
            self.layers = nn.ModuleList([GCNBlock(cfg, self.adj) for _ in range(cfg.L_layers)])
        elif use_attn:
            self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.L_layers)])
        else:
            self.layers = nn.ModuleList([MLPBlock(cfg) for _ in range(cfg.L_layers)])

    def forward(self, hidden: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        hidden = hidden + injection
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


# -- Memory -------------------------------------------------------------------

@dataclass
class TRMMemory:
    prediction: torch.Tensor   # (B, S, D) — current prediction state
    reasoning: torch.Tensor    # (B, S, D) — reasoning scratch state


# -- Full model ---------------------------------------------------------------

class TRM(nn.Module):
    """
    Tiny Recursive Model for tabular or graph-based regression.

    Tabular input: (B, num_features) flat feature vector.
    Graph input: PyG Batch object (for gnn_* backbones).
    Output: (B,) scalar prediction (log10 ionic conductivity).
    """
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.cfg = cfg
        self.is_gnn = cfg.backbone.startswith("gnn_")
        self.is_mace = cfg.backbone == "mace"
        D = cfg.hidden_dim

        if self.is_mace:
            from mace_modules import MACEEncoder, MACEReasoningModule
            self.mace_encoder = MACEEncoder(
                num_features=cfg.mace_num_features,
                max_ell=cfg.mace_max_ell,
                num_bessel=cfg.mace_num_bessel,
                polynomial_cutoff=cfg.mace_polynomial_cutoff,
                cutoff=cfg.gnn_cutoff,
                max_atomic_num=cfg.max_atomic_num,
            )
            self.core = MACEReasoningModule(
                num_features=cfg.mace_num_features,
                max_ell=cfg.mace_max_ell,
                num_bessel=cfg.mace_num_bessel,
                correlation=cfg.mace_correlation,
                avg_num_neighbors=cfg.mace_avg_num_neighbors,
                interaction_cls=cfg.mace_interaction,
                radial_mlp=cfg.mace_radial_mlp,
                num_layers=cfg.L_layers,
                max_atomic_num=cfg.max_atomic_num,
            )
            head_dim = cfg.mace_num_features
        elif self.is_gnn:
            from gnn_encoder import GNNEncoder, AttentionPooling
            self.gnn_encoder = GNNEncoder(
                hidden_dim=D,
                n_conv_layers=cfg.gnn_conv_layers,
                n_gaussians=cfg.n_gaussians,
                cutoff=cfg.gnn_cutoff,
                max_atomic_num=cfg.max_atomic_num,
            )
            self.attn_pool = AttentionPooling(
                hidden_dim=D,
                n_queries=cfg.pool_k,
                num_heads=cfg.num_heads,
            )
            S = cfg.pool_k
            self.pred_init = nn.Parameter(torch.randn(1, S, D) * 0.02)
            self.Z_init = nn.Parameter(torch.randn(1, S, D) * 0.02)
            self.core = ReasoningModule(cfg, S)
            head_dim = D
        else:
            S = cfg.num_features
            # Per-position projection: each feature slot gets its own Linear(1 → D)
            self.input_proj = nn.ModuleList([nn.Linear(1, D) for _ in range(S)])
            self.pos_embed = nn.Parameter(torch.randn(1, S, D) * 0.02)
            self.pred_init = nn.Parameter(torch.randn(1, S, D) * 0.02)
            self.Z_init = nn.Parameter(torch.randn(1, S, D) * 0.02)
            self.core = ReasoningModule(cfg, S)
            head_dim = D

        # Regression head: pool over positions/nodes, project to scalar
        self.head_norm = nn.LayerNorm(head_dim)
        self.head = nn.Linear(head_dim, 1)

    def _encode_tabular(self, x: torch.Tensor) -> torch.Tensor:
        """Project (B, S) flat features → (B, S, D) hidden sequence."""
        parts = [proj(x[:, i : i + 1]) for i, proj in enumerate(self.input_proj)]
        return torch.stack(parts, dim=1) + self.pos_embed

    def _encode_graph(self, batch) -> torch.Tensor:
        """Encode PyG Batch → (B, K, D) via GNN + attention pooling."""
        node_emb = self.gnn_encoder(batch.z, batch.edge_index, batch.edge_attr,
                                    batch.batch)
        return self.attn_pool(node_emb, batch.batch)

    def _init_memory(self, B: int) -> TRMMemory:
        return TRMMemory(
            prediction=self.pred_init.expand(B, -1, -1).clone(),
            reasoning=self.Z_init.expand(B, -1, -1).clone(),
        )

    def _recursive_reasoning(self, encoded: torch.Tensor, B: int) -> torch.Tensor:
        """Run H_cycles of recursive reasoning, return final prediction tensor."""
        mem = self._init_memory(B)

        # H_cycles - 1 without gradient (saves memory, matches reference TRM)
        with torch.no_grad():
            for _ in range(self.cfg.H_cycles - 1):
                inj = mem.prediction + encoded
                for _ in range(self.cfg.L_cycles):
                    mem.reasoning = self.core(mem.reasoning, inj)
                mem.prediction = self.core(mem.prediction, mem.reasoning)
            mem = TRMMemory(
                prediction=mem.prediction.detach(),
                reasoning=mem.reasoning.detach(),
            )

        # Final H cycle with gradient
        inj = mem.prediction + encoded
        for _ in range(self.cfg.L_cycles):
            mem.reasoning = self.core(mem.reasoning, inj)
        pred = self.core(mem.prediction, mem.reasoning)

        # Pool over sequence positions → scalar
        pooled = pred.mean(dim=1)
        return self.head(self.head_norm(pooled)).squeeze(-1)

    def _forward_mace(self, batch) -> torch.Tensor:
        """Full forward pass for MACE backbone.

        Features stay per-node throughout the entire recurrent computation.
        Uses PyG flat batching (no padding needed).

        Args:
            batch: PyG Batch with z, edge_index, edge_vectors, batch, y
        Returns:
            (B,) scalar predictions
        """
        from torch_geometric.nn import global_mean_pool

        # Encode graph → per-node features + edge descriptors
        node_feats, node_attrs, edge_feats, edge_attrs = self.mace_encoder(
            batch.z, batch.edge_index, batch.edge_vectors, batch.batch
        )

        # Store graph structure for the reasoning module (fixed during recurrence)
        self.core.set_graph(node_attrs, edge_attrs, edge_feats, batch.edge_index)

        encoded = node_feats  # (N, irreps_dim) — only L=0 populated initially

        # Initialize memory: prediction from encoder, reasoning from zeros
        pred = encoded.clone()
        Z = torch.zeros_like(encoded)

        # H_cycles - 1 without gradient (saves memory, matches reference TRM)
        with torch.no_grad():
            for _ in range(self.cfg.H_cycles - 1):
                inj = pred + encoded
                for _ in range(self.cfg.L_cycles):
                    Z = self.core(Z, inj)
                pred = self.core(pred, Z)
            pred = pred.detach()
            Z = Z.detach()

        # Final H cycle with gradient
        inj = pred + encoded
        for _ in range(self.cfg.L_cycles):
            Z = self.core(Z, inj)
        pred = self.core(pred, Z)

        # Readout: extract L=0 scalars → mean pool over nodes → linear head
        scalar_pred = pred[:, :self.cfg.mace_num_features]  # (N, num_features)
        pooled = global_mean_pool(scalar_pred, batch.batch)  # (B, num_features)
        return self.head(self.head_norm(pooled)).squeeze(-1)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: (B, num_features) for tabular, PyG Batch for GNN/MACE backbones.
        Returns:
            (B,) — predicted log10(ionic conductivity).
        """
        if self.is_mace:
            return self._forward_mace(x)
        if self.is_gnn:
            encoded = self._encode_graph(x)
            B = encoded.shape[0]
        else:
            B = x.shape[0]
            encoded = self._encode_tabular(x)

        return self._recursive_reasoning(encoded, B)
