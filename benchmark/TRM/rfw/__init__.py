"""Recurrent Foundation Wrapper — a generic recurrent adapter for frozen
foundation models, spiritually faithful to the Tiny Recursive Model (TRM)."""

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
