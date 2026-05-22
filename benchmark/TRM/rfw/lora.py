"""
Cycle-conditional LoRA adapter.

Generates a low-rank weight update ΔW = α · A(s) · B(s) at each call,
where (A, B) are produced by hypernetworks conditioned on the recursive
state s = concat(P, Z). The update is added to the output of a frozen
linear-like operation.

This module does NOT touch the frozen layer's weights — it is applied
externally as a parallel residual:

    frozen_out = frozen_layer(input)
    delta     = (input @ A(s)) @ B(s) * alpha   # low-rank residual
    output    = frozen_out + delta

The state-conditional A and B are what makes the adapter behave
differently across recursive cycles even though the frozen layer is
unchanged.
"""

import math

import torch
from torch import nn


class CycleConditionalLoRA(nn.Module):
    """
    Args:
        in_features:  input dim of the frozen op being adapted
        out_features: output dim of the frozen op being adapted
        rank:         LoRA inner rank (small, e.g. 4-16)
        state_dim:    dim of the conditioning state (typically 2*D where
                      D is the reasoning state dim and we pass concat(P, Z))
        alpha:        LoRA scaling (default rank, gives effective scale 1)
        init_scale:   stddev of hypernet output init; small to start near identity
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        state_dim: int,
        alpha: float | None = None,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / rank

        # LayerNorm on the conditioning state — defends against the recursive
        # accumulation of P and Z (which grow unboundedly across H-cycles).
        # Without this, ||A(s)|| scales linearly with ||state|| and the LoRA
        # delta can blow up under Option B's feedback loop. HRM-Text MagicNorm
        # spirit: PreNorm on the *input* to a learned module that lives inside
        # a recurrence. Init still yields zero delta at step 0 because hyper_B
        # is zero-initialized.
        self.state_norm = nn.LayerNorm(state_dim)

        # Hypernetworks: state → flat weights, then reshape
        # A is (in_features, rank); B is (rank, out_features)
        self.hyper_A = nn.Linear(state_dim, in_features * rank)
        self.hyper_B = nn.Linear(state_dim, rank * out_features)

        # Init small so the LoRA delta starts near zero (frozen model
        # behaviour preserved at step 0)
        nn.init.normal_(self.hyper_A.weight, std=init_scale)
        nn.init.zeros_(self.hyper_A.bias)
        nn.init.zeros_(self.hyper_B.weight)
        nn.init.zeros_(self.hyper_B.bias)
        # B init = 0 → ΔW = 0 at start regardless of A; classic LoRA init

    def generate_AB(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, state_dim) — conditioning state
        Returns:
            A: (B, in_features, rank)
            B: (B, rank, out_features)
        """
        # Normalize before the hypernet projection — see __init__ for rationale.
        state = self.state_norm(state)
        bs = state.shape[0]
        A = self.hyper_A(state).view(bs, self.in_features, self.rank)
        B = self.hyper_B(state).view(bs, self.rank, self.out_features)
        return A, B

    def forward(self, x: torch.Tensor, state: torch.Tensor,
                group_index: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the LoRA delta ΔW · x for input x.

        Args:
            x:           (N, in_features) — input being processed by the frozen op.
                         For per-token operations, N is total tokens across the batch.
            state:       (B, state_dim) — per-sample conditioning state.
            group_index: (N,) long — maps each row of x to its sample in [0, B).
                         If None, assumes N == B (one row per sample).
        Returns:
            delta: (N, out_features) — low-rank residual to add to frozen output.
        """
        A, B = self.generate_AB(state)  # (B, in, r), (B, r, out)

        if group_index is None:
            assert x.shape[0] == A.shape[0], (
                f"x batch ({x.shape[0]}) != state batch ({A.shape[0]}) "
                "and no group_index given"
            )
            # (B, in_features) @ (B, in, r) → (B, r)  via batched matmul
            xA = torch.bmm(x.unsqueeze(1), A).squeeze(1)        # (B, r)
            delta = torch.bmm(xA.unsqueeze(1), B).squeeze(1)    # (B, out_features)
        else:
            # Scatter A and B per row using group_index
            A_per_row = A[group_index]   # (N, in, r)
            B_per_row = B[group_index]   # (N, r, out)
            # x: (N, in_features) → (N, 1, in) @ (N, in, r) → (N, 1, r) → (N, r)
            xA = torch.bmm(x.unsqueeze(1), A_per_row).squeeze(1)
            delta = torch.bmm(xA.unsqueeze(1), B_per_row).squeeze(1)

        return delta * self.scaling

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
