# PLAN: RFW (Recurrent Foundation Wrapper) Optimization & Cleanup

## Context

The RFW implementation at `benchmark/TRM/rfw/` and `benchmark/TRM/rfw_eqv2/` has passed three independent code reviews (correctness, efficiency, portability). The design is TRM-faithful and autograd-correct, but there are meaningful wins on memory, parameter budget, and portability before we commit to expensive training runs.

This plan captures those findings as an implementation roadmap. Items are prioritized by impact and ordered so that correctness fixes land first, quick efficiency wins next, and portability/cleanup last.

## What's confirmed correct — DO NOT change

- TRM-faithful gradient semantics: H-1 outer cycles run under `torch.no_grad()`, the final H-cycle accumulates gradients through LoRAs, cross-attn, p_update, and head. `P.detach()`/`Z.detach()` before the final cycle is correct.
- LoRA preserves SO(3) equivariance by only modifying the L=0 (scalar) channel; higher-L coefficients are untouched.
- Hook ordering: capture hook on `block` fires AFTER the LoRA hooks on `block.ga`/`block.ffn`, so captured features already include LoRA deltas (intended behavior).
- Cross-attention mask polarity: `True = padding` matches `nn.MultiheadAttention`'s `key_padding_mask` convention.
- `output.clone()` + slice-assign pattern is autograd-safe.
- Both model passes (option B) in the final H-cycle accumulate grads correctly into the LoRA hypernets and refiner.
- Frozen Equiformer parameters receive no gradient.
- `P_init` / `Z_init` correctly receive no gradient when `H_cycles >= 2` — this is TRM's 1-step gradient approximation.

## Priority 1: Correctness fixes

### 1.1 Hook cleanup leak across multiple adapters

**File**: `benchmark/TRM/rfw_eqv2/adapter.py:168-187`

**Problem**: `install_hooks` only removes hooks tracked in `self._hook_handles`. If two `EqV2Adapter` instances ever share a frozen model (e.g., via `copy.deepcopy` reuse, or during multi-fold CV that reuses a model ref), both sets of hooks fire — each looking up its own `_loras` dict — producing double LoRA application and corrupted state.

**Fix**:

```python
def install_hooks(self, frozen_model: nn.Module) -> None:
    # Remove OUR hooks from prior installs
    for h in self._hook_handles:
        h.remove()
    self._hook_handles.clear()

    # Also remove any other RFW hooks installed on this model by a sibling adapter
    # — identifiable by a marker attribute we set below.
    stale = getattr(frozen_model, "_rfw_hook_handles", [])
    for h in stale:
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
    # Expose to siblings so they can clear us if they install next
    frozen_model._rfw_hook_handles = new_handles
```

**Test**: construct two adapters on the same frozen_model, run forward on the second, verify only one adapter's hooks fire (check via its LoRAs having grad, the other's not).

### 1.2 Sortedness assumption in `_pad_per_graph`

**File**: `benchmark/TRM/rfw_eqv2/adapter.py:257-288`

**Problem**: `intra_idx = torch.arange(N_atoms) - offsets[batch_vec]` assumes `batch_vec` is sorted (contiguous per graph). PyG's `Batch.from_data_list` guarantees this, but any future atom shuffling/filtering would silently produce wrong indices.

**Fix**: add an assert at the top of `_pad_per_graph`:

```python
assert (batch_vec[1:] >= batch_vec[:-1]).all(), (
    "_pad_per_graph requires batch_vec to be non-decreasing. "
    "If atoms were shuffled, argsort batch_vec first."
)
```

Tiny cost on CPU; negligible on GPU. Catches a class of silent data corruption.

## Priority 2: Quick efficiency wins

### 2.1 Eliminate `output.clone()` in the LoRA hook

**File**: `benchmark/TRM/rfw_eqv2/adapter.py:230-246`

**Problem**: Each LoRA hook clones the entire `(N_atoms, 25, 128)` SO3 tensor to avoid mutating the frozen module's returned tensor. With 14 hooks per forward × 6 passes per training step, this is ~110 MB/step of wasted bandwidth and autograd memory at batch=8, N≈100.

**Fix** — use `torch.cat` so only the L=0 slice gets a new tensor; L>0 passes through unchanged:

```python
def _make_lora_hook(self, site_name: str):
    def hook(module, inputs, output):
        if self._current_state is None or site_name not in self._loras:
            return output
        lora = self._loras[site_name]
        bv = self._current_batch_vec
        # Equivariance-preserving: delta only on L=0
        scalars = output[:, 0, :]                           # (N_atoms, C)
        delta = lora(scalars, self._current_state, bv)      # (N_atoms, C)
        l0_new = (scalars + delta).unsqueeze(1)             # (N_atoms, 1, C)
        return torch.cat([l0_new, output[:, 1:, :]], dim=1)
    return hook
```

This avoids cloning L>0 coefficients (24/25 of the tensor) — ~96% of the wasted memory eliminated. Verify with a memory profiler that peak autograd memory drops.

### 2.2 Cache `_pyg_to_fairchem_batch` per forward call

**File**: `benchmark/TRM/rfw_eqv2/adapter.py:195-226`

**Problem**: `run_model` calls `_pyg_to_fairchem_batch` on every invocation — 6× per step (H=3 × 2 passes per H-cycle with option B). The conversion iterates through `to_data_list` and rebuilds a Batch — pure Python overhead, 5-20 ms each.

**Fix** — cache within a single `forward()` call. Add a cache key so it's reset per batch:

```python
def run_model(self, frozen_model, batch):
    # Build the fairchem-format batch once per outer `forward()` call
    cache_key = id(batch)
    if getattr(self, "_fairchem_cache_key", None) != cache_key:
        device = next(frozen_model.parameters()).device
        self._cached_fairchem_batch = _pyg_to_fairchem_batch(batch, device=device)
        self._fairchem_cache_key = cache_key
    fairchem_batch = self._cached_fairchem_batch
    self._current_batch_vec = fairchem_batch.batch
    ...  # rest unchanged
```

Reset the cache key to `None` at the start of each `RecurrentFoundationWrapper.forward()` (or just rely on the `id()` changing per new Batch object).

### 2.3 Lower default `lora_rank` to 4

**File**: `benchmark/TRM/rfw/wrapper.py:120`

**Problem**: At `lora_rank=8`, each LoRA site has ~264K params × 14 sites = 3.7M trainable. The user's stated budget is ~2M.

**Fix**: change `lora_rank` default from 8 to 4 in `RFWConfig`. Brings total LoRA params to ~1.85M. User can still override for expressiveness experiments.

```python
@dataclass
class RFWConfig:
    ...
    lora_rank: int = 4   # was 8; rank-4 hits the ~2M trainable target
    ...
```

Document the tradeoff in the docstring: rank 4 ≈ 1.85M, rank 8 ≈ 3.7M, rank 16 ≈ 7.4M.

### 2.4 Make `prediction_via_model=False` easier to access for memory-bound runs

**File**: `benchmark/TRM/rfw/wrapper.py:124`

**Problem**: With option B on and `H_cycles=3`, we do 6 Equiformer forwards per step (2 grad, 4 no-grad). Option A does 3 forwards. For memory-bound batch sizes, this is the single biggest lever.

**Fix**: don't change the default (option B is spiritually correct), but:

- Add a clearer docstring explaining the tradeoff
- Expose a `--prediction_via_model` / `--no_prediction_via_model` flag in `train_rfw.py` (when we create it)
- In the SLURM script, if batch_size > threshold, suggest `--no_prediction_via_model`

## Priority 3: Portability / code-organization cleanup

### 3.1 Move generic utilities to `rfw/padding.py`

**Files**:
- Create: `benchmark/TRM/rfw/padding.py`
- Modify: `benchmark/TRM/rfw_eqv2/adapter.py` (remove `_pad_per_graph`, `_mean_pool_per_graph`, import from new location)
- Modify: `benchmark/TRM/rfw/test_smoke.py` (import from new location — fixes the layering violation where `rfw/test_smoke.py` imports from `rfw_eqv2/`)

**New file contents**:

```python
"""Generic padding / pooling utilities for the Recurrent Foundation Wrapper.

These were originally in rfw_eqv2 but are domain-agnostic — any adapter
producing ragged per-token features benefits from them.
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
        "group_index must be non-decreasing; argsort first if not"
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

    stacked_n_first = stacked.transpose(0, 1)  # (N_total, L, F)
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
    counts = torch.bincount(group_index, minlength=num_groups).clamp(min=1).to(x.dtype)
    return out / counts.unsqueeze(-1)
```

Keep thin aliases in `rfw_eqv2/adapter.py` during transition to avoid breaking imports:

```python
from rfw.padding import pad_ragged as _pad_per_graph          # deprecated alias
from rfw.padding import mean_pool_grouped as _mean_pool_per_graph
```

### 3.2 Populate `rfw/__init__.py` with public exports

**File**: `benchmark/TRM/rfw/__init__.py`

**Currently**: empty. Users must import from submodules.

**Fix**:

```python
"""Recurrent Foundation Wrapper — a generic recurrent adapter for
frozen foundation models, spiritually faithful to TRM."""

from .wrapper import (
    RecurrentFoundationWrapper,
    RFWConfig,
    FoundationAdapter,
    LoRASite,
)
from .lora import CycleConditionalLoRA
from .cross_attn import CrossAttentionRefiner
from .padding import pad_ragged, mean_pool_grouped

__all__ = [
    "RecurrentFoundationWrapper",
    "RFWConfig",
    "FoundationAdapter",
    "LoRASite",
    "CycleConditionalLoRA",
    "CrossAttentionRefiner",
    "pad_ragged",
    "mean_pool_grouped",
]
```

### 3.3 Generalize `_batch_size` in wrapper

**File**: `benchmark/TRM/rfw/wrapper.py:256-266`

**Currently**:

```python
def _batch_size(self, batch):
    if hasattr(batch, "num_graphs"):
        return batch.num_graphs
    if isinstance(batch, dict) and "batch_size" in batch:
        return batch["batch_size"]
    raise ValueError(...)
```

**Fix** — add a Tensor branch and delegate to the adapter as ultimate fallback:

```python
def _batch_size(self, batch):
    # Try adapter first — it knows its own domain
    if hasattr(self.adapter, "batch_size"):
        return self.adapter.batch_size(batch)
    # Fallbacks for common cases
    if hasattr(batch, "num_graphs"):
        return batch.num_graphs
    if isinstance(batch, dict) and "batch_size" in batch:
        return batch["batch_size"]
    if isinstance(batch, torch.Tensor) and batch.ndim >= 2:
        return batch.shape[0]
    raise ValueError(
        "Cannot determine batch size. Implement `batch_size(batch) -> int` "
        "on your FoundationAdapter subclass, or pass a recognized batch type."
    )
```

Update `FoundationAdapter` docstring to note `batch_size` as an optional override.

### 3.4 Rename PyG-isms in rfw/ to domain-neutral terms

**Files**: `rfw/wrapper.py`, `rfw/lora.py`, `rfw/cross_attn.py`

Search-and-replace in `rfw/` only (leave `rfw_eqv2/` alone — crystals DO have graphs):

- `graph_index` → `group_index` (in `lora.py:82`, docstrings, and call sites)
- `per_graph_pooled` → `per_sample_pooled` (in `wrapper.py` return dict key and all references)
- "per-graph" / "per graph" → "per-sample" in docstrings
- "num_graphs" → "batch size" in docstrings (the actual `.num_graphs` attribute access in `_batch_size` stays, since it's a real PyG contract)

`rfw_eqv2/adapter.py` will need to update its dict keys when calling `run_model` to match the renamed keys.

Leave "tokens" (already neutral) and "LoRASite" (fine) alone.

### 3.5 Decouple adapter state from PyTorch hook mechanism (deferred)

**Files**: `rfw/wrapper.py`, `rfw_eqv2/adapter.py`

**Problem**: Every adapter duplicates boilerplate — `_loras`, `_current_state`, `_captured_features`, `_current_batch_vec`, `_hook_handles` instance variables, plus `_make_lora_hook` and `_make_capture_hook` factory methods. A third-party author implementing `ESMAdapter` would have to reproduce ~70% of this.

**Longer-term fix** (defer until after first training results): refactor the FoundationAdapter contract so the adapter only declares *what* and the wrapper handles *how*:

```python
@dataclass
class HookSpec:
    site_name: str          # matches a LoRASite name
    module_selector: Callable[[nn.Module], nn.Module]  # e.g. lambda m: m.blocks[3].ga
    combine_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    # receives (frozen_output, lora_delta) — default is identity + broadcast add


class FoundationAdapter(ABC):
    @abstractmethod
    def lora_sites(self) -> list[LoRASite]: ...

    @abstractmethod
    def hook_specs(self) -> list[HookSpec]: ...

    @abstractmethod
    def capture_specs(self) -> list[tuple[str, Callable]]:
        """(name, module_selector) pairs for features to capture."""
        ...

    @abstractmethod
    def prepare_batch(self, batch) -> Any:
        """Convert user-land batch to foundation model input format."""
        ...

    @abstractmethod
    def pool_per_sample(self, features, batch_meta) -> torch.Tensor: ...
```

The wrapper then owns `_current_state`, hook registration, capture buffers, and state setting. The adapter becomes a pure description of *what to do*, not a stateful object.

**Skip for now** — only do this if/when we actually add a second domain adapter (e.g., ESMAdapter). It's a refactor for portability that doesn't help immediate results.

## Priority 4: Deferred items (only if issues arise)

These are flagged from review but NOT worth implementing pre-training:

- **Factorized LoRA** (hypernet params scale badly with foundation hidden dim). Only matters for models ≥1280 hidden dim (ESM2-650M+). Current test target is Equiformer at 128 hidden dim — no urgency.
- **Sparse / jagged attention** to replace `_pad_per_graph` in cross-attn. Only matters if N_atom variance within a batch becomes pathological. With crystal batches this is unlikely.
- **Cache `generate_AB` once per model call** so 14 hooks reuse the same A/B. Marginal save (84M FLOPs out of billions) — not worth the complexity.
- **External `torch.utils.checkpoint` around the whole Equiformer forward**. Defer until we actually OOM at our target batch sizes. Needs care around hook determinism.
- **Bump `max_layers` default from 32 to 96**. Only matters if we wrap a 33+ layer model. Equiformer is 7 layers — currently a non-issue. Document in docstring.

## Verification

Each priority-1 and priority-2 change should be verified by running the existing smoke tests:

```bash
cd /home/mila/v/vivianoj/code/OBELiX/benchmark/TRM
# Generic wrapper tests (fast, CPU)
/home/mila/v/vivianoj/miniconda3/envs/obelix/bin/python -m rfw.test_smoke

# End-to-end with real EquiformerV3 (slow, CPU)
/home/mila/v/vivianoj/miniconda3/envs/obelix/bin/python -m rfw_eqv2.test_smoke
```

After priority 2.1 (clone elimination), add a memory check: run the end-to-end smoke test with `torch.cuda.memory_stats()` (when GPU available) or `resource.getrusage` (on CPU) before and after, and confirm peak memory drops.

After priority 2.3 (rank change), print `wrapper.param_summary()` and verify total trainable is ~1.85M.

## Implementation order

Execute in priority order, committing after each numbered item:

1. P1.1 — hook cleanup leak (~15 lines, 1 new test)
2. P1.2 — sortedness assert (~3 lines)
3. P2.1 — eliminate `output.clone()` (~10 lines, rerun smoke test)
4. P2.2 — cache fairchem batch (~10 lines, rerun smoke test)
5. P2.3 — default `lora_rank=4` (1 line)
6. P2.4 — doc/CLI surfacing of `prediction_via_model` (no code change yet; defer to when `train_rfw.py` exists)
7. P3.1 — move utilities to `rfw/padding.py` (~100 lines shuffled, update imports)
8. P3.2 — populate `rfw/__init__.py` (~20 lines)
9. P3.3 — generalize `_batch_size` (~10 lines)
10. P3.4 — PyG-ism renames in `rfw/` (search-and-replace, verify both smoke tests still pass)

Stop and reassess after P3. P3.5 and P4 items are deferred until training results indicate they matter.

## Files touched summary

| File | Touched by |
|---|---|
| `rfw/__init__.py` | P3.2 |
| `rfw/wrapper.py` | P2.3, P2.4, P3.3, P3.4 |
| `rfw/lora.py` | P3.4 |
| `rfw/cross_attn.py` | P3.4 (rename only, if any) |
| `rfw/padding.py` | P3.1 (new file) |
| `rfw/test_smoke.py` | P3.1 (import path change) |
| `rfw_eqv2/adapter.py` | P1.1, P1.2, P2.1, P2.2, P3.1 (imports) |
| `rfw_eqv2/test_smoke.py` | no change expected |

All changes should keep both smoke tests green at every step.
