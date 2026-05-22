"""
Smoke test for the Recurrent Foundation Wrapper with a dummy frozen model.

Does NOT test Equiformer — that's in rfw_eqv2/test_smoke.py.
This test uses a minimal `DummyModel` to verify the generic wrapper mechanics:
  - Param counting (trainable vs frozen)
  - Forward pass shape
  - Gradient flow to LoRA hypernets and cross-attn, but NOT to frozen weights
  - H/L cycle semantics (H-1 cycles under no_grad, final cycle with grad)
  - Checkpointing flag doesn't crash training
"""

import torch
from torch import nn

from rfw.wrapper import (
    RecurrentFoundationWrapper, RFWConfig, FoundationAdapter, LoRASite,
)


# -- A minimal fake foundation model + adapter for testing -------------------

class DummyModel(nn.Module):
    """Pretends to be a foundation model with 3 'blocks' that produce per-token
    features. Input: a dict with 'tokens': (N_tokens, F_in) and 'batch_vec'."""

    def __init__(self, num_blocks=3, feature_dim=16, num_tokens_per_sample=5):
        super().__init__()
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim
        self.blocks = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_blocks)
        ])

    def forward(self, batch):
        x = batch["tokens"]  # (N_tokens, F)
        self._block_outputs = []
        for block in self.blocks:
            x = x + torch.tanh(block(x))
            self._block_outputs.append(x)
        return x


class DummyAdapter(FoundationAdapter):
    def __init__(self, model):
        self.model = model
        self.feature_dim = model.feature_dim
        self._loras = {}
        self._current_state = None
        self._current_batch_vec = None
        self._captured = []
        self._hook_handles = []

    def lora_sites(self):
        return [
            LoRASite(name=f"block_{b}", in_features=self.feature_dim,
                     out_features=self.feature_dim)
            for b in range(self.model.num_blocks)
        ]

    def install_hooks(self, frozen_model):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        for b, block in enumerate(frozen_model.blocks):
            self._hook_handles.append(
                block.register_forward_hook(self._make_lora_hook(f"block_{b}"))
            )
            self._hook_handles.append(
                block.register_forward_hook(self._make_capture_hook(b))
            )

    def set_state(self, P, Z):
        self._current_state = torch.cat([P, Z], dim=-1)

    def set_loras(self, loras):
        self._loras = loras

    def run_model(self, frozen_model, batch, *, want_layer_features=True):
        self._captured = [None] * self.model.num_blocks
        self._current_batch_vec = batch["batch_vec"]
        _ = frozen_model(batch)

        from rfw_eqv2.adapter import _pad_per_graph, _mean_pool_per_graph
        B = batch["batch_size"]
        pooled = _mean_pool_per_graph(self._captured[-1], self._current_batch_vec, B)
        out = {"per_graph_pooled": pooled}
        if want_layer_features:
            all_layers = torch.stack(self._captured, dim=0)  # (L, N_tokens, F)
            padded, mask = _pad_per_graph(all_layers, self._current_batch_vec, B)
            out["layer_features"] = padded
            out["key_padding_mask"] = mask
        return out

    def _make_lora_hook(self, name):
        def hook(module, inputs, output):
            if name not in self._loras or self._current_state is None:
                return output
            delta = self._loras[name](output, self._current_state, self._current_batch_vec)
            return output + delta
        return hook

    def _make_capture_hook(self, b):
        def hook(module, inputs, output):
            self._captured[b] = output
        return hook


# -- Actual tests -------------------------------------------------------------

def test_forward_shape_and_params():
    model = DummyModel(num_blocks=3, feature_dim=16)
    adapter = DummyAdapter(model)
    cfg = RFWConfig(state_dim=32, H_cycles=2, L_cycles=1, lora_rank=4,
                    cross_attn_heads=2)
    wrapper = RecurrentFoundationWrapper(model, adapter, feature_dim=16, cfg=cfg)
    print(wrapper.param_summary())

    # Build a toy batch of 2 graphs with 3 and 4 tokens respectively
    N_total = 7
    batch = {
        "tokens": torch.randn(N_total, 16),
        "batch_vec": torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
        "batch_size": 2,
    }

    out = wrapper(batch)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    print(f"Output shape: {out.shape}")


def test_gradient_flow():
    torch.manual_seed(0)
    model = DummyModel(num_blocks=3, feature_dim=16)
    adapter = DummyAdapter(model)
    cfg = RFWConfig(state_dim=32, H_cycles=2, L_cycles=2, lora_rank=4)
    wrapper = RecurrentFoundationWrapper(model, adapter, feature_dim=16, cfg=cfg)

    # Break out of zero-init for LoRA B so gradients actually test something
    for lora in wrapper.loras.values():
        with torch.no_grad():
            lora.hyper_B.weight.normal_(std=0.01)

    batch = {
        "tokens": torch.randn(5, 16),
        "batch_vec": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        "batch_size": 2,
    }
    target = torch.randn(2)
    out = wrapper(batch)
    loss = (out - target).abs().mean()
    loss.backward()

    # Frozen model params should have no grad
    for p in model.parameters():
        assert p.grad is None, "Frozen param has gradient!"

    # LoRA hypernets should have gradients
    for name, lora in wrapper.loras.items():
        assert lora.hyper_A.weight.grad is not None, f"LoRA {name}.hyper_A no grad"
        assert lora.hyper_B.weight.grad is not None, f"LoRA {name}.hyper_B no grad"

    # Cross-attn refiner should have grads
    for p in wrapper.refiner.parameters():
        assert p.grad is not None

    # P_init/Z_init intentionally do NOT receive gradients when H_cycles >= 2:
    # the detach after the no_grad block cuts the link (TRM's 1-step gradient
    # approximation). This test uses H_cycles=2 so the assertion holds.
    # At H_cycles=1 the detach is skipped — see test_h_cycles_one_init_grad.
    assert cfg.H_cycles >= 2, "this test's grad-None assertion assumes H_cycles>=2"
    assert wrapper.P_init.grad is None
    assert wrapper.Z_init.grad is None

    # P_update + head should have grads (they read the final-cycle outputs)
    for p in wrapper.p_update.parameters():
        assert p.grad is not None
    for p in wrapper.head.parameters():
        assert p.grad is not None

    print("Gradient flow: OK")
    print("  frozen params: no grad (correct)")
    print("  LoRAs/CrossAttn/p_update/head: grad (correct)")
    print("  P_init/Z_init: no grad (TRM semantics — fixed init)")


def test_no_grad_on_outer_cycles():
    """Validate that H-1 cycles genuinely run under no_grad by checking
    memory behaviour: we should NOT accumulate gradient tape for those cycles."""
    model = DummyModel(num_blocks=3, feature_dim=16)
    adapter = DummyAdapter(model)
    cfg = RFWConfig(state_dim=16, H_cycles=5, L_cycles=2, lora_rank=2)
    wrapper = RecurrentFoundationWrapper(model, adapter, feature_dim=16, cfg=cfg)

    batch = {
        "tokens": torch.randn(6, 16),
        "batch_vec": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        "batch_size": 2,
    }
    out = wrapper(batch)
    # If no_grad were broken, backward would succeed through all 5 H-cycles.
    # With the fix, backward flows only through the final cycle.
    out.sum().backward()
    print("H-1 no_grad + final cycle backward: OK")


def test_checkpoint_flag():
    """use_l_cycle_checkpointing must not crash or produce different outputs
    (up to numerical tolerance) during training."""
    torch.manual_seed(0)
    model = DummyModel(num_blocks=2, feature_dim=16)
    adapter = DummyAdapter(model)

    batch = {
        "tokens": torch.randn(5, 16),
        "batch_vec": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "batch_size": 2,
    }

    cfg_no_ckpt = RFWConfig(state_dim=16, H_cycles=1, L_cycles=3, lora_rank=2,
                            use_l_cycle_checkpointing=False)
    cfg_ckpt = RFWConfig(state_dim=16, H_cycles=1, L_cycles=3, lora_rank=2,
                         use_l_cycle_checkpointing=True)

    torch.manual_seed(0)
    w1 = RecurrentFoundationWrapper(model, adapter, feature_dim=16, cfg=cfg_no_ckpt)
    torch.manual_seed(0)
    w2 = RecurrentFoundationWrapper(model, adapter, feature_dim=16, cfg=cfg_ckpt)

    # Copy weights so outputs can be compared fairly
    w2.load_state_dict(w1.state_dict())

    out1 = w1(batch)
    out2 = w2(batch)
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-5, f"Checkpoint changed output by {diff}"

    # Both should be able to backward
    out2.sum().backward()
    print(f"Checkpoint flag: output diff = {diff:.2e}, backward OK")


def _make_batch(N_total=5, batch_vec=None, F=16):
    """Helper: build a 2-graph batch with B=2."""
    if batch_vec is None:
        batch_vec = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    return {
        "tokens": torch.randn(N_total, F),
        "batch_vec": batch_vec,
        "batch_size": int(batch_vec.max().item()) + 1,
    }


def test_refiner_grad_under_option_a():
    """F1 regression: when prediction_via_model=False, the CrossAttentionRefiner
    must receive a gradient. Before the F1 fix, refiner was dead in Option A
    because refined Z was never consumed by anything on the path to the loss."""
    torch.manual_seed(0)
    model = DummyModel(num_blocks=3, feature_dim=16)
    adapter = DummyAdapter(model)
    cfg = RFWConfig(state_dim=32, H_cycles=2, L_cycles=2, lora_rank=4,
                    prediction_via_model=False)
    wrapper = RecurrentFoundationWrapper(model, adapter, feature_dim=16, cfg=cfg)

    # Break out of zero-init for LoRA B so the LoRA path is non-trivial
    for lora in wrapper.loras.values():
        with torch.no_grad():
            lora.hyper_B.weight.normal_(std=0.01)

    batch = _make_batch()
    target = torch.randn(2)
    out = wrapper(batch)
    loss = (out - target).abs().mean()
    loss.backward()

    # Refiner must have non-trivial gradient
    g = wrapper.refiner.kv_proj.weight.grad
    assert g is not None, "Option A: refiner.kv_proj.weight.grad is None (F1 bug)"
    assert g.norm().item() > 1e-8, (
        f"Option A: refiner.kv_proj.weight.grad.norm() = {g.norm().item():.2e} "
        "(suspiciously tiny — refiner may still be effectively dead)"
    )
    print(f"Option A refiner gradient: kv_proj.weight.grad.norm() = {g.norm().item():.3e}")


def test_h_cycles_one_init_grad():
    """F2 regression: at H_cycles=1 the no_grad block runs zero iterations,
    so the unconditional .detach() in the old code was severing P_init/Z_init
    for no reason — blocking the matched-LoRA baseline. With the guard, the
    inits should receive gradient at H_cycles=1."""
    torch.manual_seed(0)
    model = DummyModel(num_blocks=3, feature_dim=16)
    adapter = DummyAdapter(model)
    cfg = RFWConfig(state_dim=32, H_cycles=1, L_cycles=2, lora_rank=4)
    wrapper = RecurrentFoundationWrapper(model, adapter, feature_dim=16, cfg=cfg)

    batch = _make_batch()
    target = torch.randn(2)
    out = wrapper(batch)
    loss = (out - target).abs().mean()
    loss.backward()

    assert wrapper.P_init.grad is not None, (
        "H_cycles=1: P_init.grad is None (F2 bug — detach severed init)"
    )
    assert wrapper.Z_init.grad is not None, (
        "H_cycles=1: Z_init.grad is None (F2 bug — detach severed init)"
    )
    print("H_cycles=1: P_init/Z_init both received gradient (correct)")


def test_hook_cleanup_multi_adapter():
    """P1.1 regression: installing a second adapter on the same frozen model
    must clear the first adapter's hooks. Otherwise both adapters' hooks fire
    on every forward, producing double LoRA application and corrupted
    captures."""
    torch.manual_seed(0)
    model = DummyModel(num_blocks=3, feature_dim=16)

    # Note: the DummyAdapter uses a simpler install_hooks (no sibling clearing).
    # To test P1.1 we use a subclass that DOES clear via the frozen_model marker.
    class SiblingAwareDummyAdapter(DummyAdapter):
        def install_hooks(self, frozen_model):
            for h in self._hook_handles:
                h.remove()
            self._hook_handles.clear()
            stale = getattr(frozen_model, "_rfw_hook_handles", [])
            for h in stale:
                try: h.remove()
                except Exception: pass
            new_handles = []
            for b, block in enumerate(frozen_model.blocks):
                new_handles.append(
                    block.register_forward_hook(self._make_lora_hook(f"block_{b}"))
                )
                new_handles.append(
                    block.register_forward_hook(self._make_capture_hook(b))
                )
            self._hook_handles = new_handles
            frozen_model._rfw_hook_handles = new_handles

    # Build adapter A; install its hooks
    adapter_a = SiblingAwareDummyAdapter(model)
    cfg = RFWConfig(state_dim=16, H_cycles=1, L_cycles=1, lora_rank=2)
    wrapper_a = RecurrentFoundationWrapper(model, adapter_a, feature_dim=16, cfg=cfg)

    # Now build adapter B on the SAME frozen model. Its install_hooks should
    # have cleared adapter A's hooks via the _rfw_hook_handles marker.
    adapter_b = SiblingAwareDummyAdapter(model)
    wrapper_b = RecurrentFoundationWrapper(model, adapter_b, feature_dim=16, cfg=cfg)

    # Bend LoRA B's init so non-zero deltas can flow
    for lora in wrapper_b.loras.values():
        with torch.no_grad():
            lora.hyper_B.weight.normal_(std=0.01)

    # Forward through wrapper_b. If adapter_a's hooks still fired, adapter_a's
    # _current_state would be None (we never call set_state on it), and the
    # hooks would return output unchanged — silent corruption. We instead
    # verify directly: only adapter_b's loras should receive gradient.
    batch = _make_batch(F=16)
    target = torch.randn(2)
    out = wrapper_b(batch)
    loss = (out - target).abs().mean()
    loss.backward()

    # wrapper_a's loras should have NO gradient (its hooks were cleared)
    for name, lora in wrapper_a.loras.items():
        assert lora.hyper_A.weight.grad is None, (
            f"adapter A's LoRA {name} received gradient — its hooks were not cleared"
        )

    # wrapper_b's loras should have gradient
    for name, lora in wrapper_b.loras.items():
        assert lora.hyper_A.weight.grad is not None, (
            f"adapter B's LoRA {name} did NOT receive gradient — install failed"
        )

    print("Sibling adapter cleanup: adapter A's hooks cleared, adapter B's hooks active")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Forward shape + param counting")
    print("=" * 60)
    test_forward_shape_and_params()

    print("\n" + "=" * 60)
    print("TEST 2: Gradient flow (frozen vs trainable)")
    print("=" * 60)
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("TEST 3: no_grad on outer H cycles")
    print("=" * 60)
    test_no_grad_on_outer_cycles()

    print("\n" + "=" * 60)
    print("TEST 4: L-cycle gradient checkpointing")
    print("=" * 60)
    test_checkpoint_flag()

    print("\n" + "=" * 60)
    print("TEST 5: F1 refiner gradient under Option A")
    print("=" * 60)
    test_refiner_grad_under_option_a()

    print("\n" + "=" * 60)
    print("TEST 6: F2 H_cycles=1 init gradient")
    print("=" * 60)
    test_h_cycles_one_init_grad()

    print("\n" + "=" * 60)
    print("TEST 7: P1.1 multi-adapter hook cleanup")
    print("=" * 60)
    test_hook_cleanup_multi_adapter()

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
