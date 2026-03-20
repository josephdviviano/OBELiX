"""
GNN encoder and attention pooling for the GNN-TRM backbone.

- GaussianExpansion: expand scalar distances to Gaussian basis vectors
- GNNEncoder: atom embedding + CGConv layers → per-node embeddings
- AttentionPooling: learnable queries cross-attend to variable-length node sets
"""

import torch
from torch import nn
from torch_geometric.nn import CGConv


class GaussianExpansion(nn.Module):
    """Expand scalar distances into Gaussian basis functions."""

    def __init__(self, n_gaussians: int = 64, cutoff: float = 5.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_gaussians)
        self.register_buffer("centers", centers)
        # Width: spacing between centers
        self.width = (cutoff / n_gaussians) * 0.5

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """distances: (E, 1) → (E, n_gaussians)"""
        return torch.exp(-((distances - self.centers) ** 2) / (2 * self.width**2))


class GNNEncoder(nn.Module):
    """CGCNN-style encoder: atom embeddings + CGConv layers.

    Args:
        hidden_dim: embedding / hidden dimension
        n_conv_layers: number of CGConv layers
        n_gaussians: Gaussian expansion size for edge features
        cutoff: distance cutoff for Gaussian expansion
        max_atomic_num: max atomic number for embedding table
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_conv_layers: int = 3,
        n_gaussians: int = 64,
        cutoff: float = 5.0,
        max_atomic_num: int = 100,
    ):
        super().__init__()
        self.atom_embed = nn.Embedding(max_atomic_num + 1, hidden_dim)
        self.gauss = GaussianExpansion(n_gaussians, cutoff)
        self.edge_proj = nn.Linear(n_gaussians, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_conv_layers):
            self.convs.append(CGConv(hidden_dim, dim=hidden_dim, batch_norm=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, z, edge_index, edge_attr, batch):
        """
        Args:
            z: (N,) atomic numbers
            edge_index: (2, E) edge indices
            edge_attr: (E, 1) distances
            batch: (N,) batch assignment vector

        Returns:
            (N, D) node embeddings
        """
        x = self.atom_embed(z)
        edge_feat = self.edge_proj(self.gauss(edge_attr))

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_feat)
            x = bn(x)
            x = torch.relu(x)

        return x


class AttentionPooling(nn.Module):
    """Cross-attention pooling: K learnable queries attend to variable-length node sets.

    Uses PyG batch vector to separate graphs in a batched setting.

    Args:
        hidden_dim: dimension of node embeddings and queries
        n_queries: number of learnable query vectors (K)
        num_heads: number of attention heads
    """

    def __init__(self, hidden_dim: int = 64, n_queries: int = 32, num_heads: int = 4):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, batch):
        """
        Args:
            x: (total_nodes, D) node embeddings from GNNEncoder
            batch: (total_nodes,) batch assignment

        Returns:
            (B, K, D) pooled graph representations
        """
        # Split nodes by graph and pad for batched cross-attention
        batch_size = batch.max().item() + 1
        device = x.device

        # Find max number of nodes in this batch
        counts = torch.bincount(batch, minlength=batch_size)
        max_nodes = counts.max().item()

        # Pad node embeddings into (B, max_nodes, D) with mask
        D = x.shape[1]
        padded = torch.zeros(batch_size, max_nodes, D, device=device)
        mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)

        for i in range(batch_size):
            node_mask = batch == i
            nodes_i = x[node_mask]
            n_i = nodes_i.shape[0]
            padded[i, :n_i] = nodes_i
            mask[i, :n_i] = False  # False = attend, True = ignore

        # Cross-attention: queries attend to node embeddings
        Q = self.queries.expand(batch_size, -1, -1)
        out, attn_weights = self.cross_attn(
            Q, padded, padded, key_padding_mask=mask
        )
        out = self.norm(out)

        # Store for post-hoc interpretability: (B, K, max_nodes) head-averaged
        self.last_attn_weights = attn_weights
        self.last_node_counts = counts  # (B,) actual atom count per graph

        return out  # (B, K, D)
