"""
Tiny Recursive Model (TRM) for ionic conductivity prediction.

Two backbone variants:
  - "transformer": self-attention + FFN blocks (learns inter-feature relationships)
  - "mlp": FFN-only blocks (recursive baseline without attention)

Both share the same recursive structure: H outer cycles × L inner cycles through
a stack of L_layers blocks, with persistent memory (prediction + reasoning state).
Only the final H cycle receives gradient.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


# -- Config ------------------------------------------------------------------

@dataclass
class TRMConfig:
    num_features: int  # number of input feature slots
    hidden_dim: int = 64  # hidden dimension per position
    num_heads: int = 4  # attention heads (transformer backbone only)
    ffn_expansion: float = 2.0  # FFN hidden size = hidden_dim * expansion
    L_layers: int = 2  # blocks per reasoning module
    L_cycles: int = 2  # inner loop iterations
    H_cycles: int = 3  # outer loop iterations (only last gets grad)
    backbone: Literal["transformer", "mlp"] = "transformer"
    dropout: float = 0.1


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


# -- Recursive core ----------------------------------------------------------

class ReasoningModule(nn.Module):
    """Stack of blocks with injection: hidden = hidden + injection, then L_layers blocks."""
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        Block = TransformerBlock if cfg.backbone == "transformer" else MLPBlock
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_layers)])

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
    Tiny Recursive Model for tabular regression.

    Input: (B, num_features) flat feature vector.
    Output: (B,) scalar prediction (log10 ionic conductivity).
    """
    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.cfg = cfg
        S, D = cfg.num_features, cfg.hidden_dim

        # Per-position projection: each feature slot gets its own Linear(1 → D)
        self.input_proj = nn.ModuleList([nn.Linear(1, D) for _ in range(S)])
        self.pos_embed = nn.Parameter(torch.randn(1, S, D) * 0.02)

        # Learnable memory initialisation
        self.pred_init = nn.Parameter(torch.randn(1, S, D) * 0.02)
        self.Z_init = nn.Parameter(torch.randn(1, S, D) * 0.02)

        # Shared reasoning module (applied recursively)
        self.core = ReasoningModule(cfg)

        # Regression head: pool over sequence, project to scalar
        self.head_norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project (B, S) flat features → (B, S, D) hidden sequence."""
        # Each feature independently projected then summed with positional embedding
        parts = [proj(x[:, i : i + 1]) for i, proj in enumerate(self.input_proj)]
        return torch.stack(parts, dim=1) + self.pos_embed

    def _init_memory(self, B: int) -> TRMMemory:
        return TRMMemory(
            prediction=self.pred_init.expand(B, -1, -1).clone(),
            reasoning=self.Z_init.expand(B, -1, -1).clone(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_features) — scaled tabular features.
        Returns:
            (B,) — predicted log10(ionic conductivity).
        """
        B = x.shape[0]
        encoded = self._encode(x)
        mem = self._init_memory(B)

        # H_cycles - 1 without gradient (saves memory, matches reference TRM)
        with torch.no_grad():
            for _ in range(self.cfg.H_cycles - 1):
                inj = mem.prediction + encoded
                for _ in range(self.cfg.L_cycles):
                    mem.reasoning = self.core(mem.reasoning, inj)
                mem.prediction = self.core(mem.prediction, mem.reasoning)
            # Detach before final grad-enabled cycle
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
