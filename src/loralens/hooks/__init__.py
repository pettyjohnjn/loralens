# src/loralens/hooks/__init__.py
"""
Hooks module - Standalone activation capture system.

This module provides a clean interface for capturing intermediate
activations from transformer models. It can be used independently
for training, inference, or analysis.

Example usage:
    
    # Simple collection
    collector = ActivationCollector(model)
    with collector:
        data = collector.collect(input_ids, attention_mask)
    
    # Access activations
    for layer_id, hidden in data.iter_layers():
        print(f"Layer {layer_id}: {hidden.shape}")
"""

from .base import BaseHook, RemovableHandle
from .activation_hook import ActivationHook
from .manager import HookManager
from .collector import ActivationCollector, CollectedActivations

__all__ = [
    "BaseHook",
    "RemovableHandle",
    "ActivationHook",
    "HookManager",
    "ActivationCollector",
    "CollectedActivations",
]
