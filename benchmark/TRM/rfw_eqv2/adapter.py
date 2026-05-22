"""
EquiformerV3 adapter for the Recurrent Foundation Wrapper.

Provides the Equiformer-specific plumbing:
  - Lists LoRA insertion sites: input embedding + attention output + FFN output
    at every transformer block
  - Installs forward hooks that add cycle-conditional LoRA deltas to the L=0
    (scalar) channel of the frozen attention/FFN outputs. Only the L=0 slice is
    touched; higher-L (vector/tensor) coefficients are left unchanged, which
    preserves SO(3) equivariance.
  - Runs the frozen model, capturing per-block outputs as layer features
  - Extracts the L=0 scalar channel per atom per layer for the cross-attention
    refiner (lmax can go up to 4 so extracting L=0 keeps invariance simple)
  - Pads per-atom features into a fixed (B, L, N_max, F) tensor with a mask

Feature shape convention: EquiformerV3 internal tensors are
  (N_atoms, num_coeffs=(lmax+1)^2, num_channels)
The L=0 scalar component is at coefficient index 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from rfw.wrapper import FoundationAdapter, LoRASite
from rfw.padding import pad_ragged, mean_pool_grouped


def _load_equiformer_v3(checkpoint_name: str = "omat24-mptrj-salex_gradient.pt",
                        device: str = "cpu") -> nn.Module:
    """Download & load a pre-trained EquiformerV3, freeze all params.

    Extracted to a helper so tests can stub it and training scripts can call
    it directly. Lives here (in rfw_eqv2) because it is Equiformer-specific.
    """
    vendor_root = Path(__file__).resolve().parent.parent / "vendor_equiformer_v3"
    vendor_src = vendor_root / "src"
    if str(vendor_src) not in sys.path:
        sys.path.insert(0, str(vendor_src))
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    # Purge stale fairchem imports so the vendored src takes priority
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("fairchem"):
            del sys.modules[mod_name]

    from experimental.models.equiformer_v3 import equiformer_v3 as _eqv3  # noqa: F401
    from huggingface_hub import hf_hub_download
    from fairchem.core.common.registry import registry

    ckpt_path = hf_hub_download(
        repo_id="mirror-physics/equiformer_v3",
        filename=f"checkpoint/{checkpoint_name}",
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_config = dict(checkpoint["config"]["model"])
    model_config.pop("name", None)
    # Feature extraction only — disable force/stress heads so forward pass
    # doesn't call .requires_grad_ on positions
    model_config["regress_forces"] = False
    model_config["regress_stress"] = False
    model_config["direct_prediction"] = True
    # Disable Equiformer's built-in per-block gradient checkpointing. It is
    # incompatible with our forward hooks: the hooks fire on both the initial
    # forward AND the backward recompute pass, which breaks autograd's metadata
    # tracking (CheckpointError: "Recomputed values have different metadata").
    # Memory savings instead come from smaller batch sizes and the
    # `--no_prediction_via_model` flag (skips the second grad-bearing pass).
    if "gradient_checkpointing_block_list" in model_config:
        model_config["gradient_checkpointing_block_list"] = [0] * len(
            model_config["gradient_checkpointing_block_list"])

    model_cls = registry.get_model_class("equiformer_v3")
    model = model_cls(**model_config)

    state_dict = checkpoint["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)


def _pyg_to_fairchem_batch(pyg_batch, device="cpu"):
    """Translate our PyG Data batch (z, pos, cell, edge_index, ...) to
    the field names/format the fairchem EquiformerV3 expects.

    Small fairchem-required tensors (natoms/tags/pbc/fixed) are constructed
    directly on `device` to avoid creating them on CPU and then transferring
    — that would add ~4-6 small CPU→GPU sync points per graph per call.
    """
    from torch_geometric.data import Data, Batch

    # If it's a list of Data, collate first
    if isinstance(pyg_batch, list):
        pyg_batch = Batch.from_data_list(pyg_batch)

    # Build a fresh fairchem-conformant batch by iterating through the
    # per-graph Data objects so natoms/tags/pbc are set per-graph correctly
    graphs = pyg_batch.to_data_list()
    fairchem_graphs = []
    for g in graphs:
        N = g.z.shape[0]
        cell = g.cell if g.cell.dim() == 3 else g.cell.unsqueeze(0)
        fairchem_graphs.append(Data(
            atomic_numbers=g.z.to(torch.long),
            pos=g.pos,
            cell=cell,
            natoms=torch.tensor([N], dtype=torch.long, device=device),
            tags=torch.zeros(N, dtype=torch.long, device=device),
            pbc=torch.tensor([[True, True, True]], dtype=torch.bool, device=device),
            fixed=torch.zeros(N, dtype=torch.long, device=device),
            y=g.y,
        ))
    return Batch.from_data_list(fairchem_graphs).to(device)


class EqV2Adapter(FoundationAdapter):
    """
    Domain glue for EquiformerV3.

    LoRA sites (per block b in 0..num_blocks-1):
      - "block_{b}_attn_out": after attention output, before residual
      - "block_{b}_ffn_out":  after FFN output, before residual

    Hooks:
      - On `model.blocks[b].attn`  : add LoRA delta to the output tensor
      - On `model.blocks[b].ffn`   : add LoRA delta to the output tensor
      - On `model.blocks[b]` itself: capture the block's final output as
        the per-layer feature for L-cycle cross-attention
    """

    def __init__(self, frozen_model: nn.Module, lora_rank: int = 8):
        self.frozen_model = frozen_model
        self.num_blocks = len(frozen_model.blocks)
        # Channel dim is shared across all sites (Equiformer has a single
        # num_channels throughout its transformer blocks)
        self.num_channels = frozen_model.num_channels
        self.lora_rank = lora_rank

        # Populated by set_loras / set_state / install_hooks
        self._loras: dict = {}
        self._current_state: torch.Tensor | None = None
        self._captured_features: list[torch.Tensor] = []
        self._current_batch_vec: torch.Tensor | None = None

        self._hook_handles: list = []

        # Cache of the most recent PyG-to-fairchem conversion, keyed by id(batch).
        # Avoids ~5-20ms re-conversion per Equiformer pass; with Option B at H=3
        # that's up to 6 calls per training step on the same batch.
        self._fairchem_cache_key: int | None = None
        self._cached_fairchem_batch = None

    # -- FoundationAdapter interface ------------------------------------------

    def lora_sites(self) -> list[LoRASite]:
        sites = []
        for b in range(self.num_blocks):
            sites.append(LoRASite(
                name=f"block_{b}_attn_out",
                in_features=self.num_channels,
                out_features=self.num_channels,
            ))
            sites.append(LoRASite(
                name=f"block_{b}_ffn_out",
                in_features=self.num_channels,
                out_features=self.num_channels,
            ))
        return sites

    def install_hooks(self, frozen_model: nn.Module) -> None:
        """Remove any previous hooks first (idempotent), then register:
          1. attn output hook → adds LoRA delta
          2. ffn output hook → adds LoRA delta
          3. block output hook → captures per-layer features

        Also clears hooks installed by a *sibling* RFW adapter on the same
        frozen model — without this, instantiating a second adapter on a shared
        frozen model would leave the first adapter's hooks active, producing
        double LoRA application and corrupted captures.
        """
        # Drop our own previously-installed hooks
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

        # Drop any sibling RFW adapter's hooks on this model
        sibling_handles = getattr(frozen_model, "_rfw_hook_handles", [])
        for h in sibling_handles:
            try:
                h.remove()
            except Exception:
                pass

        new_handles = []
        for b, block in enumerate(frozen_model.blocks):
            new_handles.append(
                block.ga.register_forward_hook(self._make_lora_hook(f"block_{b}_attn_out"))
            )
            new_handles.append(
                block.ffn.register_forward_hook(self._make_lora_hook(f"block_{b}_ffn_out"))
            )
            new_handles.append(
                block.register_forward_hook(self._make_capture_hook(b))
            )
        self._hook_handles = new_handles
        # Marker so a future sibling adapter can clear us in turn
        frozen_model._rfw_hook_handles = new_handles

    def set_state(self, P: torch.Tensor, Z: torch.Tensor) -> None:
        self._current_state = torch.cat([P, Z], dim=-1)

    def set_loras(self, loras: dict) -> None:
        self._loras = loras

    def run_model(
        self,
        frozen_model: nn.Module,
        batch,
        *,
        want_layer_features: bool = True,
    ) -> dict:
        """Run Equiformer and return layer features + pooled per-graph features."""
        # Invalidate stale state before re-populating. If the LoRA hook fires
        # without these being freshly set (e.g. someone bypasses run_model),
        # the in-hook asserts will catch it instead of producing wrong-but-not-
        # crashing output via stale per-atom indexing.
        self._current_batch_vec = None
        self._captured_features = [None] * self.num_blocks

        # Reuse the converted fairchem batch if the caller is the same PyG batch.
        # In Option B at H=3 we call run_model up to 6× per training step on
        # the same input batch — caching saves the per-call Python overhead.
        cache_key = id(batch)
        if self._fairchem_cache_key != cache_key or self._cached_fairchem_batch is None:
            device = next(frozen_model.parameters()).device
            self._cached_fairchem_batch = _pyg_to_fairchem_batch(batch, device=device)
            self._fairchem_cache_key = cache_key
        fairchem_batch = self._cached_fairchem_batch
        self._current_batch_vec = fairchem_batch.batch  # (N_atoms,) long

        _ = frozen_model(fairchem_batch)
        # After the forward pass, self._captured_features[b] holds the block's
        # output — shape (N_atoms, num_coeffs, num_channels). Extract L=0.

        B = fairchem_batch.num_graphs

        # Per-graph pooled — always built; mean over atoms of the final layer.
        final = self._captured_features[-1][:, 0, :]   # (N_atoms, C) L=0 scalars
        pooled = mean_pool_grouped(
            final, group_index=self._current_batch_vec, num_groups=B,
        )

        out = {"per_graph_pooled": pooled}

        if want_layer_features:
            # Stack to (num_blocks, N_atoms, C), then pad per-graph.
            l0_per_layer = [feat[:, 0, :] for feat in self._captured_features]
            all_layers = torch.stack(l0_per_layer, dim=0)  # (L, N_atoms, C)
            padded, mask = pad_ragged(
                all_layers,
                group_index=self._current_batch_vec,
                num_groups=B,
            )
            # padded: (B, L, N_max, C), mask: (B, N_max) with True for padding
            out["layer_features"] = padded
            out["key_padding_mask"] = mask

        # Release the per-block raw captures. The derived tensors above
        # (pooled, optionally padded) already reference whatever activations
        # autograd needs, so the captures themselves are now garbage.
        self._captured_features = []

        return out

    # -- hook factories -------------------------------------------------------

    def _make_lora_hook(self, site_name: str):
        def hook(module, inputs, output):
            # output: (N_atoms, num_coeffs, num_channels) — Equiformer SO3 tensor
            if self._current_state is None or site_name not in self._loras:
                return output

            lora = self._loras[site_name]
            bv = self._current_batch_vec                       # (N_atoms,)
            # Equivariance-preserving LoRA: compute a scalar delta from the L=0
            # channel per atom, and add it ONLY to the L=0 channel. Higher-L
            # coefficients are untouched so SO(3) equivariance is preserved.
            scalars = output[:, 0, :]                          # (N_atoms, C)

            # Defensive asserts — catch silent porting/staleness bugs early.
            assert scalars.shape[-1] == lora.in_features, (
                f"{site_name}: frozen output channels {scalars.shape[-1]} != "
                f"LoRASite.in_features {lora.in_features}"
            )
            assert bv is not None, (
                f"{site_name}: LoRA hook fired without _current_batch_vec set; "
                "did frozen_model run outside run_model()?"
            )
            assert bv.shape[0] == scalars.shape[0], (
                f"{site_name}: stale batch_vec ({bv.shape[0]}) does not match "
                f"current atom count ({scalars.shape[0]})"
            )

            delta = lora(scalars, self._current_state, bv)     # (N_atoms, C)
            # Equivariance-preserving merge that avoids cloning the L>0 slice.
            # The L=0 slice gets a new tensor (scalars + delta), but the
            # remaining (lmax+1)^2 - 1 SH coefficients (≈24/25 of the tensor)
            # pass through unchanged. Saves ~96% of the autograd memory the
            # full clone would consume.
            l0_new = (scalars + delta).unsqueeze(1)            # (N_atoms, 1, C)
            return torch.cat([l0_new, output[:, 1:, :]], dim=1)
        return hook

    def _make_capture_hook(self, block_idx: int):
        def hook(module, inputs, output):
            # output is the block's final SO3 tensor (N_atoms, num_coeffs, C)
            self._captured_features[block_idx] = output
        return hook


# -- backward-compat aliases -------------------------------------------------
# The padding utilities now live in `rfw.padding` (domain-neutral names).
# These aliases preserve the old `_pad_per_graph` / `_mean_pool_per_graph`
# import path used by `rfw/test_smoke.py` and any external callers.

def _pad_per_graph(stacked, batch_vec, num_graphs):
    return pad_ragged(stacked, group_index=batch_vec, num_groups=num_graphs)


def _mean_pool_per_graph(x, batch_vec, num_graphs):
    return mean_pool_grouped(x, group_index=batch_vec, num_groups=num_graphs)
