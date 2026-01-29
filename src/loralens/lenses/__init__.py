# src/loralens/lenses/__init__.py
"""
Lenses module - Neural network modules that project activations to vocabulary space.

Each lens takes hidden states from an intermediate layer and produces
logits that can be compared to the model's output distribution.

Example usage:
    
    # Create a lens
    lens = create_lens(
        "lora",
        layer_ids=list(range(12)),
        hidden_size=768,
        unembed=unembed,
        r=16,
    )
    
    # Forward pass
    output = lens(hidden_states, layer=5)
    logits = output.logits
"""

from .types import LayerId, LensOutput
from .base import BaseLens
from .logit_lens import LogitLens
from .tuned_lens import TunedLens
from .lora_lens import LoRAProjection, LoRALens
from .factory import create_lens, register_lens, list_lenses

__all__ = [
    # Types
    "LayerId",
    "LensOutput",
    # Base
    "BaseLens",
    # Implementations
    "LogitLens",
    "TunedLens",
    "LoRAProjection",
    "LoRALens",
    # Factory
    "create_lens",
    "register_lens",
    "list_lenses",
]
