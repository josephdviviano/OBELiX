"""EquiformerV3-specific adapter for the Recurrent Foundation Wrapper.

## Setup (vendor EquiformerV3)

The frozen backbone lives in `vendor_equiformer_v3/` — an external clone of
the fairchem `equiformer_v3` package. Git-ignored; fetch separately before
running training:

    cd benchmark/TRM
    git clone https://github.com/FAIR-Chem/fairchem vendor_equiformer_v3

Then verify with:

    sbatch run_smoke.sh                # runs both smoke suites

The first eqv2 forward pass also downloads a pre-trained checkpoint from
HuggingFace Hub (`mirror-physics/equiformer_v3`) into the local cache.
"""

from .adapter import EqV2Adapter, _load_equiformer_v3

__all__ = ["EqV2Adapter", "_load_equiformer_v3"]
