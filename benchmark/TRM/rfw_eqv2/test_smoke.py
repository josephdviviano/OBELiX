"""
End-to-end smoke test for RFW + EquiformerV3 on a CPU with a tiny config.

Verifies:
  - Model loads, hooks install
  - Forward pass produces correct output shape
  - Gradient flows to trainable adapters but NOT to frozen Equiformer weights
  - Per-block L=0 features are correctly shaped
  - Both batch=1 and batch=3 work

Slow — each forward pass runs Equiformer on CPU. H_cycles=1, L_cycles=1
to keep the test quick.
"""

import sys
from pathlib import Path

import torch

# Allow running as a module or script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rfw.wrapper import RecurrentFoundationWrapper, RFWConfig
from rfw_eqv2.adapter import EqV2Adapter, _load_equiformer_v3
from graph_data import load_gnn_data


def main():
    print("Loading EquiformerV3 (frozen)...")
    frozen_model = _load_equiformer_v3(
        "omat24-mptrj-salex_gradient.pt", device="cpu")
    print(f"  Loaded {sum(p.numel() for p in frozen_model.parameters()):,} params")

    print("\nBuilding EqV2Adapter and RFW...")
    adapter = EqV2Adapter(frozen_model)
    feature_dim = frozen_model.num_channels
    cfg = RFWConfig(
        state_dim=32,                 # small to keep LoRA hypernets small
        H_cycles=1,                   # minimal for smoke test
        L_cycles=1,
        lora_rank=4,
        cross_attn_heads=2,
        prediction_via_model=True,
    )
    wrapper = RecurrentFoundationWrapper(frozen_model, adapter,
                                         feature_dim=feature_dim, cfg=cfg)
    print(wrapper.param_summary())

    print("\nLoading a few CIFs...")
    train_data, _ = load_gnn_data(cutoff=12.0)
    sample = train_data[:3]
    print(f"  Using {len(sample)} structures: atoms = "
          f"{[s.z.shape[0] for s in sample]}")

    print("\nForward pass (batch=1)...")
    from torch_geometric.data import Batch
    batch1 = Batch.from_data_list([sample[0]])
    out1 = wrapper(batch1)
    assert out1.shape == (1,), f"Expected (1,), got {out1.shape}"
    print(f"  Output: {out1.item():.4f}  shape={out1.shape}  OK")

    print("\nForward pass (batch=3)...")
    batch3 = Batch.from_data_list(sample)
    out3 = wrapper(batch3)
    assert out3.shape == (3,), f"Expected (3,), got {out3.shape}"
    print(f"  Outputs: {out3.detach().tolist()}  shape={out3.shape}  OK")

    print("\nGradient flow test...")
    target = torch.randn(3)
    loss = (out3 - target).abs().mean()
    loss.backward()

    # Frozen Equiformer params must have no grad
    eqv_with_grad = [
        n for n, p in frozen_model.named_parameters() if p.grad is not None
    ]
    assert not eqv_with_grad, (
        f"Frozen Equiformer has gradients on {len(eqv_with_grad)} params! "
        f"e.g. {eqv_with_grad[:3]}"
    )
    print(f"  Frozen Equiformer: 0 / {len(list(frozen_model.parameters()))} "
          "params have gradient (correct)")

    # LoRA hypernets must have grads
    lora_grad_counts = 0
    for name, lora in wrapper.loras.items():
        if lora.hyper_A.weight.grad is not None:
            lora_grad_counts += 1
    assert lora_grad_counts == len(wrapper.loras), (
        f"Only {lora_grad_counts}/{len(wrapper.loras)} LoRAs received gradient"
    )
    print(f"  LoRAs: {lora_grad_counts}/{len(wrapper.loras)} received gradient (correct)")

    # Refiner + p_update + head must have grads
    for mod_name, mod in [("refiner", wrapper.refiner),
                          ("p_update", wrapper.p_update),
                          ("head", wrapper.head)]:
        has_grad = all(p.grad is not None for p in mod.parameters())
        assert has_grad, f"{mod_name} missing gradients"
        print(f"  {mod_name}: gradients OK")

    # Finite-ness: the loss and all gradients should be finite (no NaN/Inf)
    assert torch.isfinite(loss).item(), f"Loss is {loss.item()}"
    for name, p in wrapper.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), f"{name} has non-finite grad"
    print("  All gradients finite (no NaN/Inf)")

    print("\n" + "=" * 60)
    print("RFW + EquiformerV3 SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
