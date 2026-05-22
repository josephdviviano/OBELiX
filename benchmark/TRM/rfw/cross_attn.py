"""
Cross-attention update for the reasoning state Z.

During L-cycles, Z is refined by cross-attending to the frozen model's
layer features. Z is the query; layer features (concatenated across all
captured layers, with layer-positional encoding) are keys/values.

This module is domain-agnostic. The caller provides:
  - Z: (B, D) — reasoning state
  - layer_features: (B, num_layers, N_max_tokens, F) — captured layer outputs
    for each graph, stacked along the layer axis and padded along the token axis
  - key_padding_mask: (B, N_max_tokens) — True where padded (per token only;
    the same mask applies at every layer, so we broadcast it internally)

The feature projection (F → D) is owned by this module so Z stays in
D-space regardless of foundation model.
"""

import torch
from torch import nn


class CrossAttentionRefiner(nn.Module):
    """Z update: Z ← Z + LN(CrossAttn(Q=Z, K=V=layer_features))."""

    def __init__(
        self,
        state_dim: int,          # D
        feature_dim: int,        # F — raw foundation model feature dim
        num_heads: int = 4,
        ffn_expansion: float = 2.0,
        dropout: float = 0.0,
        max_layers: int = 32,    # for layer positional embedding
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim

        # Project raw layer features → state_dim so attention is dimensionally consistent
        self.kv_proj = nn.Linear(feature_dim, state_dim)

        # Layer positional embedding — disambiguates which layer a token came from
        self.layer_pos = nn.Parameter(torch.randn(max_layers, state_dim) * 0.02)

        self.norm_q = nn.LayerNorm(state_dim)
        self.norm_kv = nn.LayerNorm(state_dim)
        self.cross_attn = nn.MultiheadAttention(
            state_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(state_dim)
        hidden = int(state_dim * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, state_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        Z: torch.Tensor,                           # (B, D)
        layer_features: torch.Tensor,              # (B, num_layers, N_max, F)
        key_padding_mask: torch.Tensor | None,     # (B, N_max) — True where padded
    ) -> torch.Tensor:
        """
        Args:
            Z:                 (B, D)
            layer_features:    (B, L, N_max, F) — padded per-token features at each layer
            key_padding_mask:  (B, N_max) — True for padding positions in the token axis
        Returns:
            Z_new: (B, D)
        """
        B, L, N_max, F = layer_features.shape
        assert L <= self.layer_pos.shape[0], (
            f"num_layers {L} exceeds max_layers {self.layer_pos.shape[0]}"
        )

        # Project F → D
        kv = self.kv_proj(layer_features)  # (B, L, N_max, D)

        # Add layer positional embedding — broadcast across tokens
        kv = kv + self.layer_pos[:L].view(1, L, 1, self.state_dim)

        # Flatten layers × tokens
        kv = kv.reshape(B, L * N_max, self.state_dim)          # (B, L*N_max, D)

        # Build key padding mask — the same per-token mask, repeated for each layer
        if key_padding_mask is not None:
            # (B, N_max) → (B, L, N_max) → (B, L*N_max)
            kpm = key_padding_mask.unsqueeze(1).expand(B, L, N_max).reshape(B, L * N_max)
        else:
            kpm = None

        # Cross-attention with Z as query
        q = self.norm_q(Z).unsqueeze(1)        # (B, 1, D)
        k = v = self.norm_kv(kv)                # (B, L*N_max, D)

        attn_out, _ = self.cross_attn(q, k, v, key_padding_mask=kpm, need_weights=False)
        Z_new = Z + attn_out.squeeze(1)         # residual connection

        # Position-wise FFN
        Z_new = Z_new + self.ffn(self.norm_ffn(Z_new))

        return Z_new

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
