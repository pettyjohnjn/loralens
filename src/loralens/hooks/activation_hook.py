# src/loralens/hooks/activation_hook.py
"""Activation capture and transform hooks."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .base import BaseHook


# Type aliases
TransformFn = Callable[[torch.Tensor, str, nn.Module], torch.Tensor]
CallbackFn = Callable[[torch.Tensor, str, nn.Module], Any]


class ActivationHook(BaseHook):
    """
    Hook for capturing and optionally transforming activations.
    
    Parameters
    ----------
    name : str
        Unique identifier for this hook.
    on_activation : Optional[CallbackFn]
        Callback invoked with (activation, module_name, module).
        Use for capturing/logging activations.
    transform_fn : Optional[TransformFn]
        Function that transforms the activation before it continues
        through the network. Use for interventions.
    module_name : Optional[str]
        Human-readable name for the module (for logging).
        
    Examples
    --------
    >>> # Capture activations
    >>> captured = []
    >>> hook = ActivationHook(
    ...     name="capture",
    ...     on_activation=lambda x, n, m: captured.append(x.clone())
    ... )
    
    >>> # Transform activations (intervention)
    >>> hook = ActivationHook(
    ...     name="scale",
    ...     transform_fn=lambda x, n, m: x * 2.0
    ... )
    """

    def __init__(
        self,
        name: str,
        on_activation: Optional[CallbackFn] = None,
        transform_fn: Optional[TransformFn] = None,
        module_name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.on_activation = on_activation
        self.transform_fn = transform_fn
        self.module_name = module_name

    def register(self, module: nn.Module) -> None:
        """Attach hook to module's forward pass."""
        module_name = self.module_name or ""

        def hook_fn(mod: nn.Module, inputs, outputs):
            # Handle tensor outputs
            if isinstance(outputs, torch.Tensor):
                if self.on_activation is not None:
                    self.on_activation(outputs, module_name, mod)
                if self.transform_fn is not None:
                    return self.transform_fn(outputs, module_name, mod)
                return outputs
            
            # Handle tuple outputs (common in transformers)
            if isinstance(outputs, tuple) and len(outputs) > 0:
                first = outputs[0]
                if isinstance(first, torch.Tensor):
                    if self.on_activation is not None:
                        self.on_activation(first, module_name, mod)
                    if self.transform_fn is not None:
                        transformed = self.transform_fn(first, module_name, mod)
                        return (transformed,) + outputs[1:]
            
            # Non-tensor outputs: just callback, no transform
            if self.on_activation is not None:
                self.on_activation(outputs, module_name, mod)
            return outputs

        self._handle = module.register_forward_hook(hook_fn)
        self.module = module

    def __repr__(self) -> str:
        status = "registered" if self.is_registered else "unregistered"
        return (
            f"ActivationHook(name={self.name!r}, "
            f"module_name={self.module_name!r}, {status})"
        )
