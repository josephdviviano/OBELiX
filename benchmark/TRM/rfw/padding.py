"""
Generic padding / pooling utilities for the Recurrent Foundation Wrapper.

These were originally in rfw_eqv2 but are domain-agnostic — any adapter
producing ragged per-token features benefits from them. Keeping them in the
generic package also removes a layering violation: rfw/test_smoke.py used to
import from rfw_eqv2/.
"""

import torch


def pad_ragged(
    stacked: torch.Tensor,        # (L, N_total, F) layer-first tokens
    group_index: torch.Tensor,    # (N_total,) which sample each token belongs to
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad (L, N_total, F) into (B, L, N_max, F) + (B, N_max) mask.

    Args:
        stacked:     (L, N_total, F) captured features, layer-major.
        group_index: (N_total,) long — NON-DECREASING index of the group
                     each token belongs to. For PyG this is `batch.batch`.
        num_groups:  B — total number of groups (samples).

    Returns:
        padded: (B, L, N_max, F)
        mask:   (B, N_max), True where padded (per PyTorch attention convention).
    """
    assert (group_index[1:] >= group_index[:-1]).all(), (
        "pad_ragged requires group_index to be non-decreasing; "
        "argsort first if not."
    )
    L, N_total, F = stacked.shape
    device = stacked.device

    counts = torch.bincount(group_index, minlength=num_groups)
    N_max = int(counts.max().item())

    offsets = counts.cumsum(0) - counts
    intra_idx = torch.arange(N_total, device=device) - offsets[group_index]

    padded = torch.zeros(num_groups, L, N_max, F,
                         dtype=stacked.dtype, device=device)
    mask = torch.ones(num_groups, N_max, dtype=torch.bool, device=device)

    # (L, N, F) → (N, L, F) for the scatter
    stacked_n_first = stacked.transpose(0, 1)
    padded[group_index, :, intra_idx, :] = stacked_n_first
    mask[group_index, intra_idx] = False

    return padded, mask


def mean_pool_grouped(
    x: torch.Tensor,              # (N_total, F)
    group_index: torch.Tensor,    # (N_total,)
    num_groups: int,
) -> torch.Tensor:
    """Mean of per-token features within each group. Returns (B, F)."""
    F = x.shape[1]
    out = torch.zeros(num_groups, F, dtype=x.dtype, device=x.device)
    out.index_add_(0, group_index, x)
    counts = (
        torch.bincount(group_index, minlength=num_groups)
        .clamp(min=1)
        .to(x.dtype)
    )
    return out / counts.unsqueeze(-1)
