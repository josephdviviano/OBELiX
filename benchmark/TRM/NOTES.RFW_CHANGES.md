# RFW Algorithm Changelog

Running record of algorithmic changes to the Recurrent Foundation Wrapper.
Each entry describes a *behavior* change to the model, not a code refactor.
Pure portability/move work is collected at the bottom under "Mechanical cleanup".

The goal is to keep a clean trail of what the model does and why, so the next
design pass starts from facts, not from re-reading the diff.

## Phase 0 — 2026-05-22 — Plan locked in

**State before**: PLAN.TRM_WRAPPER_OPTIMIZE.md enumerated 10 efficiency/portability
items from three prior reviews. The wrapper was considered correctness-clean.

**What the independent review found**:
- **F1 (critical)**: In Option A, `CrossAttentionRefiner` has no gradient path to
  the loss — refined Z is not consumed in any loss-bearing computation.
- **F2 (high)**: Unconditional `.detach()` after the no_grad block severs
  P_init/Z_init at `H_cycles=1`, breaking the matched-LoRA baseline.
- 8 smaller findings (capture pinning, PASS-2 wasted compute, stale batch_vec,
  shape asserts, LoRA-input normalization gap, device-side tensor construction).

**Inspirations queued from HRM-Text**:
- MagicNorm → LayerNorm on LoRA hypernet input state.
- Input-conditional z_H⁰ → optional `init_from_input` mode using a zero-LoRA
  frozen pass.
- Warmup BPTT depth → deferred.

**Existing baseline reframed**: Job 9273608's val_mae=2.08 was trained with
a dead refiner. Not a valid number for the recurrent thesis.

(Implementation begins next.)

## Step 1 — 2026-05-22 — F1: refiner gradient path restored in Option A

**Algorithm change**: When `prediction_via_model=False`, `p_update` now consumes
`cat([per_graph_pooled, refined_Z], dim=-1)` instead of just pooled features.
The input dim of the linear is `feature_dim + D` (vs `feature_dim` in Option B).
This restores a gradient path from the loss back through `p_update` into the
`CrossAttentionRefiner`, so the L-cycle Z refinement actually learns under
Option A.

**Side effect**: `p_update` shape now depends on `prediction_via_model`, so
checkpoints are not cross-compatible across the flag. Documented in
`RFWConfig`. Job 9273608's checkpoint (trained as Option A with dead refiner)
is no longer loadable into the new wrapper — acceptable, that checkpoint was
unreliable anyway.

**Smoke**: job 9625539 PASS (rfw.test_smoke, default config = Option B → no
behavior change for default path).

**Followup**: step 14 will add `test_refiner_grad_under_option_a` that
explicitly forces `prediction_via_model=False` and asserts
`refiner.kv_proj.weight.grad.norm() > 0`.

## Step 2 — 2026-05-22 — F2: H_cycles=1 detach guard

**Algorithm change**: `P.detach()` and `Z.detach()` after the no_grad block now
only run when `H_cycles > 1`. At H=1 the no_grad block executes zero iterations,
so detaching was severing P_init/Z_init from the loss for no reason. With the
guard, at H=1 the gradient flows through `.expand().contiguous()` back into
P_init/Z_init, and the inits actually learn.

**Why it matters**: H_cycles=1 is the matched-LoRA baseline experiment promised
in NOTES.PAPER.md — RFW with H=1 should behave like a sophisticated one-shot
LoRA. Without learnable inits this comparison was rigged against RFW.

**Test update**: `test_gradient_flow` still asserts `P_init.grad is None` but
now under an explicit `cfg.H_cycles >= 2` precondition; the H=1 case will be
covered by `test_h_cycles_one_init_grad` in step 14.

**Smoke**: job 9625547 PASS.

## Step 3 — 2026-05-22 — P1.1: sibling-adapter hook cleanup

**Algorithm change**: `EqV2Adapter.install_hooks` now also clears hooks left
behind by any prior RFW adapter on the same frozen model (tracked via
`frozen_model._rfw_hook_handles`). Prevents double LoRA application and
corrupted captures when two adapters ever share a frozen model (e.g.
multi-fold CV reusing a model reference, deepcopy reuse).

**Smoke**: pending — covered by next eqv2 dispatch.

## Step 4 — 2026-05-22 — F4 + F5: defensive asserts in LoRA hook

**Algorithm change** (defensive, not behavioral): the LoRA hook now asserts
(a) frozen output's L=0 channel count matches `LoRASite.in_features`, (b)
`_current_batch_vec` is set, (c) `_current_batch_vec.shape[0]` matches the
atom-row count of the current call. `run_model` also invalidates
`_current_batch_vec = None` at its top before re-populating, so a stale value
is impossible to silently use.

**Why**: catches porting bugs (mismatched channel counts when adapting a
different foundation model) and cache-staleness bugs (P2.2 caching makes the
batch_vec value more important to keep fresh).

**Smoke**: pending — covered by next eqv2 dispatch.

## Step 6 — 2026-05-22 — P2.1: clone-free LoRA hook

**Algorithm change** (memory, not behavior): the LoRA hook no longer clones
the full (N_atoms, num_coeffs, num_channels) SO3 tensor. The L=0 slice is
combined via `torch.cat([l0_new, output[:, 1:, :]], dim=1)`, so the
(lmax+1)^2 - 1 unchanged channels pass through as views. Expected ~96%
reduction in the autograd memory wasted by the prior clone pattern. Output is
identical numerically because L>0 channels are unchanged by the LoRA.

## Step 7 — 2026-05-22 — P2.2: cache fairchem batch per outer forward

**Algorithm change** (no behavior change): `EqV2Adapter` now caches the
PyG→fairchem conversion keyed by `id(batch)`. With Option B at H=3 we call
`run_model` up to 6 times per training step on the same PyG batch; this
removes 5x of the ~5-20ms Python-side conversion overhead per step.

## Step 8 — 2026-05-22 — F3 + F6: capture release + PASS-2 skip

**Algorithm change**: `FoundationAdapter.run_model` gained a
`want_layer_features` kwarg (default True). Option B's PASS-2 call passes
False — it only consumes pooled features for the P update, so building
`layer_features`/`key_padding_mask` (a `(B, num_blocks, N_max, C)` tensor) is
wasted. Saves ~25 MB per H-cycle at typical batch sizes. Additionally,
`_captured_features` is cleared at the end of `run_model` once the derived
pooled/padded tensors exist — releases the raw per-block SO3 captures (~13 MB)
between calls.

**Test impact**: `DummyAdapter.run_model` updated to accept the new kwarg.

## Step 9 — 2026-05-22 — P2.3 + P2.4: lora_rank=4 default + CLI surface

**Algorithm change** (param budget): `RFWConfig.lora_rank` default 8→4.
14 LoRA sites × rank 4 → ~1.85M trainable LoRA params, vs ~3.7M at rank 8.
Hits the stated ~2M budget; rank 8 and 16 remain available for expressiveness
experiments. `train_rfw.py` already exposed `--no_prediction_via_model`; only
argparse default for `--lora_rank` needed updating.

## Step 10 — 2026-05-22 — F10: construct fairchem helper tensors on device

**Algorithm change** (no behavior change): `natoms`, `tags`, `pbc`, `fixed`
tensors built in `_pyg_to_fairchem_batch` now live on `device` at construction
instead of being built on CPU and transferred via `.to(device)`. Removes
~4-6 small CPU→GPU sync points per graph per call. Minor by itself; matters
more when bypassing the P2.2 cache.

## Step 11 — 2026-05-22 — 3.1: LayerNorm on LoRA state (MagicNorm spirit)

**Algorithm change**: `CycleConditionalLoRA.generate_AB` now applies a
LayerNorm to the conditioning state `concat(P, Z)` before feeding it to
`hyper_A` / `hyper_B`. Defends against the recursive accumulation: P and Z
grow unboundedly across H-cycles, and without normalization `||A(s)||` scales
linearly with `||state||`, allowing the LoRA delta to blow up under Option B's
feedback loop. Behavior at init is unchanged because `hyper_B` is still
zero-initialized → delta is zero regardless of normalized state.

**Inspiration**: HRM-Text MagicNorm — PreNorm on the *input* to a learned
module that lives inside a recurrence. Adapted to the LoRA hypernet input.

**Smoke**: pending — covered by next eqv2 dispatch (with steps 3, 4, 6, 7, 8, 9, 10).

## Step 12 — 2026-05-22 — 3.2: input-conditional P/Z init (opt-in)

**Algorithm change** (gated behind `init_from_input=False` default): new path
where the wrapper does an extra zero-LoRA frozen pass at the top of `forward`,
pools the per-graph features, and projects to D-dim via learnable
`p_init_proj` / `z_init_proj` Sequential(LayerNorm, Linear). When the flag is
on, `P_init` / `Z_init` are not allocated; the projections take their place.
Costs one extra no-grad foundation pass per `forward`.

**Gradient dynamics**: the projection weights have the *same* gradient
behavior as P_init/Z_init under the default path — they receive grad at
H_cycles=1 (via the F2 fix) and are effectively fixed under H_cycles≥2 (1-step
approximation). At H=1 the model can learn an input-conditional init; at H≥2
the projections are frozen-in-practice but still produce input-dependent
starting states.

**Inspiration**: HRM-Text initializes z_H⁰ from input embeddings rather than
a context-free random parameter. Adapted to RFW: pooled foundation features
+ learnable projection.

**Param impact**: at feature_dim=128 and D=64, projections cost ~16K params
(vs ~128 for the param-only inits). Negligible against the 1.85M LoRA budget.

## Step 13 — 2026-05-22 — Portability cleanup (no algorithm change)

Mechanical cleanup, no behavior change:

- `rfw/padding.py` created with `pad_ragged` and `mean_pool_grouped` (renamed
  from `_pad_per_graph` / `_mean_pool_per_graph`). The sortedness assert from
  P1.2 lives here. Old underscore names remain as backward-compat aliases in
  `rfw_eqv2/adapter.py` to keep `rfw/test_smoke.py` imports working.
- `rfw/__init__.py` populated with public exports.
- `RecurrentFoundationWrapper._batch_size` now tries
  `self.adapter.batch_size(batch)` first, then falls back through num_graphs /
  dict / 2D+ tensor heuristics.
- `CycleConditionalLoRA.forward` parameter `graph_index` renamed to
  `group_index`. Callers pass positionally so no API break.

## Step 14 — 2026-05-22 — Test additions for F1, F2, P1.1

**Tests added** (not algorithmic changes, but they pin down the bug fixes):

- `test_refiner_grad_under_option_a`: forces `prediction_via_model=False`,
  runs forward+backward, asserts `refiner.kv_proj.weight.grad.norm() > 1e-8`.
  Would have caught the F1 dead-refiner bug.
- `test_h_cycles_one_init_grad`: forces `H_cycles=1`, asserts P_init.grad
  and Z_init.grad are both non-None after backward. Would have caught F2.
- `test_hook_cleanup_multi_adapter`: builds two adapters on the same frozen
  model with a sibling-aware install pattern; verifies that after wrapper_b's
  backward, wrapper_a's LoRAs have no gradient (its hooks were cleared).
  Would have caught P1.1 double-hook-fire.

## Verification — 2026-05-22 — Comprehensive smoke 9625652

Both `rfw.test_smoke` (7/7 tests, including F1/F2/P1.1 regression tests) and
`rfw_eqv2.test_smoke` (real EquiformerV3 forward+backward at batch=1 and
batch=3) PASS after all 14 algorithmic + cleanup steps have landed.

Key signals:
- `test_refiner_grad_under_option_a`: `kv_proj.weight.grad.norm() = 1.797e-01`
  (well above 0 — refiner is no longer dead under Option A).
- `test_h_cycles_one_init_grad`: P_init/Z_init both receive gradient at H=1.
- `test_hook_cleanup_multi_adapter`: sibling adapter's hooks cleared on
  second install; only the active adapter's LoRAs receive gradient.
- Equiformer end-to-end: all 14 LoRA sites, refiner, p_update, head receive
  gradient. Frozen Equiformer's 292 params: 0 with gradient. No NaN/Inf.

(Step 15 — re-launch training — requires user go-ahead since it consumes
~5h of GPU time on Mila quota.)
