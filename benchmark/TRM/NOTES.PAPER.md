# Paper Notes — Recurrent Foundation Wrapper (RFW)

Notes accumulated while building and debugging the RFW prototype on ionic
conductivity. Meant as a working document for framing the paper.

## Core thesis

**For a matched trainable-parameter budget, a recurrent wrapper around a
frozen foundation model is more expressive than a static adapter (e.g.
standard LoRA).** Recursion converts parameter count into effective depth:
a 4M-param wrapper applied H cycles is closer in expressive capacity to
`H × rank` than to just `rank`, because each cycle's reshape of the frozen
backbone is conditioned on state accumulated in prior cycles.

Same intuition that motivated the original Tiny Recursive Model (TRM) paper,
carried over to the adapter setting: recursion is cheap expressiveness.

## The swappability story

With a single frozen backbone on disk (tens of GB for a 7B LLM), ship:

- `wrapper_chemistry.pt` — a few MB, trained for ionic conductivity
- `wrapper_protein_binding.pt` — same shape, different task
- `wrapper_stability.pt` — same shape, different task

Swap at inference by loading a different wrapper state dict. Backbone stays
resident across tasks.

Compelling because:
1. **Deployment footprint**: hosting N tasks = 1 × backbone + N × (tiny
   wrapper). For a 7B model with 4M-param wrappers, 14 GB + 40 MB per task.
   Adapter-soup territory.
2. **Task isolation**: each task lives in its own wrapper, no catastrophic
   forgetting. Consistent with LoRA-MoE and adapter-mixture work.
3. **Composability potential**: because the recurrence state evolves,
   wrappers could in principle be chained or blended mid-inference — "apply
   protein-binding for cycles 1-2, then stability for cycle 3." Speculative;
   don't claim in v1, but natural follow-up.

## Architecture schematic

Dimensions for our Equiformer run: B = batch, N = atoms in a graph
(variable, ≤ N_max after padding), K = 7 blocks, C = 128 channels,
M = 25 SH coefficients (lmax=4), D = 64 state dim, r = 8 LoRA rank.
Total trainable: ~3.75 M. Frozen Equiformer: 30.3 M.

### Top-level data flow (one H-cycle)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  Inputs                            │  Persistent state                    ║
║    batch : PyG Batch               │    P : (B, D)   prediction state     ║
║            ├─ z     : (ΣN,)        │    Z : (B, D)   reasoning state      ║
║            ├─ pos   : (ΣN, 3)      │                                      ║
║            ├─ cell  : (B, 3, 3)    │                                      ║
║            └─ batch : (ΣN,)        │                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [1]  Hypernets generate LoRAs from concat(P, Z) : (B, 2D)                 │
│                                                                           │
│      for each of 14 sites (one per {attn, ffn} × 7 blocks):               │
│          hyper_A : Linear(2D → C·r)   →   A : (B, C, r)  ≈  0.13 M / site │
│          hyper_B : Linear(2D → r·C)   →   B : (B, r, C)  ≈  0.13 M / site │
│      Total LoRA hypernet params: 14 × 2 × 0.13 M  ≈  3.7 M                │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [2]  Frozen EquiformerV3 forward pass with LoRAs installed via hooks      │
│                                                                           │
│      block_0  →  block_1  →  ...  →  block_6                              │
│         │           │                   │                                 │
│         ├── captures L=0 of each block's output: (ΣN, C) per block        │
│         │                                                                 │
│         ├── hook on block_b.ga  (attention):                              │
│         │     output : (ΣN, M, C)                                         │
│         │     scalar = output[:, 0, :]            (ΣN, C)                 │
│         │     delta  = scalar @ A @ B · scaling   (ΣN, C)                 │
│         │     output[:, 0, :] += delta            ← equivariant           │
│         │                                                                 │
│         └── hook on block_b.ffn : same mechanism                          │
│                                                                           │
│      After all 7 blocks: stacked L=0 features (B, K=7, N_max, C)          │
│      + key_padding_mask (B, N_max)                                        │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [3]  L-cycles: refine Z via cross-attention (LoRAs + features FROZEN here)│
│                                                                           │
│      for _ in range(L_cycles):                                            │
│          kv = Linear(C → D)(layer_features)       (B, K·N_max, D)         │
│          kv += layer_positional_embedding[:K]     (B, K·N_max, D)         │
│          Z  += CrossAttn(Q=Z, K=V=kv, mask)       (B, D)                  │
│          Z  += FFN(LayerNorm(Z))                                          │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [4]  Update P (two options):                                              │
│                                                                           │
│      prediction_via_model=True:                                           │
│          regenerate LoRAs with new (P, Z)                                 │
│          run Equiformer again (2nd grad-bearing pass!)                    │
│          pool final L=0 per graph  →  (B, C)                              │
│          P += Linear(C → D)(LayerNorm(pooled))                            │
│                                                                           │
│      prediction_via_model=False (cheaper):                                │
│          P += Linear(C → D)(LayerNorm(first-pass pooled))                 │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
            (repeat H cycles; H-1 under no_grad, final with grad)
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ [5]  Readout                                                              │
│      prediction = Linear(D → 1)(LayerNorm(P))     (B,)                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### LoRA site placement inside each Equiformer block

```
TransBlockV3.forward(x):
    x : (N, 25, 128)        ← SO3 tensor, 25 SH coeffs × 128 channels

    ┌─ attn_out = self.ga(norm_1(x), ...)            (N, 25, 128)
    │      ╔═══════════════════════════════════════╗
    │      ║  HOOK: LoRA "block_{b}_attn_out"      ║  ← SITE 1 per block
    │      ║    scalar = attn_out[:, 0, :]  (N,128)║
    │      ║    delta  = state-conditional lo-rank ║
    │      ║    attn_out[:, 0, :] += delta         ║
    │      ╚═══════════════════════════════════════╝
    │      x = x + attn_out                           (N, 25, 128)

    ├─ ffn_out = self.ffn(norm_2(x))                  (N, 25, 128)
    │      ╔═══════════════════════════════════════╗
    │      ║  HOOK: LoRA "block_{b}_ffn_out"       ║  ← SITE 2 per block
    │      ║    scalar = ffn_out[:, 0, :]   (N,128)║
    │      ║    delta  = state-conditional lo-rank ║
    │      ║    ffn_out[:, 0, :] += delta          ║
    │      ╚═══════════════════════════════════════╝
    │      x = x + ffn_out                            (N, 25, 128)

    └─ HOOK: capture x → self._captured_features[b]  (for L-cycle)

    return x
```

**Equivariance note**: LoRA deltas are computed from the L=0 scalar
channel (rotationally invariant) and added only to the L=0 channel.
Higher-L components (vectors, tensors) are never touched, so SO(3)
equivariance of the frozen Equiformer is preserved through every
LoRA-modulated forward pass.

### Hypernetwork anatomy (one LoRA site)

```
                state s = concat(P, Z)  :  (B, 2D=128)
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
          hyper_A : Linear           hyper_B : Linear
          (2D → C·r = 128·8)         (2D → r·C = 8·128)
          weight: 128 × 1024         weight: 128 × 1024
          66 K params + 1 K bias     66 K params + 1 K bias
                   │                   │
                   ▼                   ▼
              A : (B, 128, 8)      B : (B, 8, 128)
                   │                   │
                   └───── matmul ──────┘
                             │
                             ▼
                    ΔW(s) : (B, 128, 128)
                             │
                             ▼  (applied per-atom via graph index)
               delta = (scalar @ A) @ B · (α/r)
               delta : (N_atoms, 128)
```

Per site: ~132 K trainable params. × 14 sites = ~1.86 M / pass of the
hypernet projections; × 2 for (A, B) hypernets = ~3.7 M total.
Init: `hyper_B` starts zero → LoRA delta is zero at step 0 so the
frozen model's behaviour is preserved until training begins.

### State machinery

```
                ┌──────────────┐
                │  P_init (1,D)│  (learnable param, no grad in practice
                │  Z_init (1,D)│   because detach after no_grad block)
                └──────┬───────┘
                       │ expand(B, D)
                       ▼
                 P, Z : (B, D)     ← persistent state
                       │
    ┌──────────────────┴───────────────────┐
    │                                      │
    ▼                                      ▼
  fed into                             updated by:
  hyper_A, hyper_B                     - cross-attention (Z)
  via concat(P, Z)                     - p_update head (P)
  at every H-cycle                     after each H-cycle
```

## Parameter budget recap

```
Frozen EquiformerV3                         30.3 M
──────────────────────────────────────────────────
Hypernets (14 sites × 2 hypernets × 132K)    3.7 M   ←  dominant
Cross-attn refiner (proj + attn + FFN)       0.05 M
P_update (Linear + LN)                       0.004 M
Output head (LN + Linear)                    0.0001 M
P_init, Z_init (parameters, 2 × D)           0.0001 M
──────────────────────────────────────────────────
Total trainable                              3.75 M  ≈ 12 % of frozen
```

## Architectural summary (as implemented)

- **Frozen foundation model**: weights never touched
- **State-conditional LoRA adapters** at K sites inside the frozen model
  (14 sites for our Equiformer: attn output + ffn output per block × 7
  blocks). LoRA matrices (A, B) are generated by hypernetworks conditioned
  on concat(P, Z).
- **Two persistent states**: prediction P and reasoning Z, both (B, D)
- **H outer cycles × L inner cycles**:
  - Each H-cycle: generate LoRAs from (P, Z), run frozen model → layer
    features, then L inner cycles refine Z via cross-attention to layer
    features (LoRAs frozen during L-cycles)
  - With `prediction_via_model=True`: run a second frozen forward pass with
    LoRAs conditioned on (P, refined-Z) to update P. More expressive, 2×
    compute per H-cycle.
  - With `prediction_via_model=False`: lightweight MLP update for P. Cheaper,
    probably loses the expressiveness argument.
- **Gradient flow**: H-1 outer cycles under `no_grad` (TRM semantics);
  only the final H-cycle backprops. P_init/Z_init don't receive gradients
  (detach after no_grad block cuts the link — matches TRM original).

## Memory analysis for scaling to LLMs

For a single 48GB GPU, batch=1, bf16, without gradient-checkpointing the
frozen model:

| LLM size | Fits? |
|---|---|
| GPT-2 small (124M) | easy, seq up to 16K |
| GPT-2 XL (1.5B) | yes, seq up to 4K |
| Llama-7B | barely, seq ≤ 1K |
| 13B | needs grad checkpoint on frozen model |
| 70B | needs model parallelism |

**The real bottleneck is gradient through frozen activations, not the
wrapper itself.** Our 3-4M wrapper params are trivial to fit; the issue is
storing activations of the backbone's forward pass to backprop through.

### Critical implementation issue we hit

Equiformer's built-in per-block gradient checkpointing is **incompatible
with forward hooks**: hooks fire during backward's recompute pass, and
autograd's metadata tracking breaks (`CheckpointError: recomputed values
have different metadata`).

**Fix path for LLMs**: replace the forward-hook approach with **PEFT-style
`nn.Linear` → `LoRALinear` module swapping**. The LoRALinear class owns its
own state and integrates cleanly with torch.utils.checkpoint. This is how
HuggingFace PEFT handles it. More intrusive than hooks but widely proven.

Without this refactor, the wrapper caps out around 7B on 48GB.

## What to be careful not to claim

- **Not parameter-efficient** — adding recurrence doesn't reduce params vs
  LoRA at matched expressivity. The claim is *expressiveness per param*,
  which is subtler (and correct).
- **Not yet general across modalities** — we've demonstrated on Equiformer.
  Until we have an LLM or protein model result, "works for any foundation
  model" is aspirational. Frame as "we present a general framework and
  validate on materials; extending to other modalities is ongoing."
- **Don't downplay compute** — at training, we're 2-6× slower than one-shot
  LoRA due to H recursive forward passes. Acknowledge honestly: trade
  compute at training for expressiveness at a fixed param budget.

## Experiments needed for the paper

### Core claim: expressiveness per param

**Matched-param baseline comparison**, same task + backbone:
- (a) Static LoRA, rank chosen to match our total trainable params
- (b) Our recurrent wrapper, same total params
- (c) Full fine-tuning of final 1-2 backbone blocks, ~same param count

If ours beats all three at matched params, that's a one-figure paper.

### Ablation: recurrence depth

H ∈ {1, 2, 3, 5}, same total params (adjust rank to keep params constant).
Shows whether extra recurrence monotonically helps, saturates, or peaks.
H=1 with our wrapper ≈ sophisticated one-shot LoRA, so this is clean.

### Multi-task / swappability

Train wrappers for 2-3 different properties on the same backbone. Show:
1. Backbone memory is amortized across tasks
2. Each wrapper achieves task-competitive accuracy independently
3. (Stretch) wrappers can be swapped at inference without backbone reload

For crystals, plausible tasks: ionic conductivity + formation energy +
band gap.

**Without this experiment the swappability story is speculation.** Even
two tasks with a shared backbone would make the point.

### If scaling to LLMs

Repeat the matched-param comparison on a small LLM (GPT-2 Medium or a 1B
model that fits comfortably) for a text classification/regression task.
Validates the domain-agnostic framing.

## Current state of the prototype

- `rfw/` — generic wrapper, domain-agnostic (`lora.py`, `cross_attn.py`,
  `wrapper.py`)
- `rfw_eqv2/` — Equiformer-specific adapter (`adapter.py`)
- `train_rfw.py` + `run_train_rfw.sh` — training driver
- Trainable params: ~3.75M (at state_dim=64, rank=8, 14 LoRA sites)
- Frozen Equiformer: 30.3M
- First training job (9273608) running on Quadro RTX 8000, 35/45 GB GPU used,
  99% utilization, batch=1, H=2, L=2, no_prediction_via_model (for memory).

## Follow-up design tasks

- **PEFT-style LoRA injection** for LLM scaling (hooks → module swapping).
  Required for >7B models on single GPUs.
- **Activation offloading to CPU** as alternative to grad checkpointing.
  Cheap when PCIe bandwidth is abundant.
- **Per-token reasoning state** (Z as (N_tokens, D) instead of (B, D)) —
  richer but needs careful pooling/scattering. v2 extension.
- **Mixed precision** (bf16 autocast around the Equiformer forward pass) —
  immediate 2× memory win once verified to play with hooks.
