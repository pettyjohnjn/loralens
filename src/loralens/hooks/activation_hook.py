# hooks/activation_hook.py

from __future__ import annotations

from typing import Callable, Optional, Any
import torch
import torch.nn as nn

from .base_hook import BaseHook

ActivationTransformFn = Callable[[torch.Tensor, str, nn.Module], torch.Tensor]
ActivationCallbackFn = Callable[[torch.Tensor, str, nn.Module], Any]


class ActivationHook(BaseHook):
    """
    Generic forward hook that captures and optionally transforms activations.

    This class is model- and lens-agnostic; you provide small callables that
    define what to do with the activations.

    Args:
        name:
            Unique identifier for this hook.
        transform_fn:
            Optional function that takes (activation, module_name, module)
            and returns a transformed activation. Its output is fed forward.
        on_activation:
            Optional side-effect function that takes (activation, module_name, module)
            and returns nothing (e.g. for logging / caching).
        module_name:
            Optional string with the name of the module in the model. If None,
            the manager can set it externally.
    """

    def __init__(
        self,
        name: str,
        transform_fn: Optional[ActivationTransformFn] = None,
        on_activation: Optional[ActivationCallbackFn] = None,
        module_name: Optional[str] = None,
    ) -> None:
        super().__init__(name)
        self.transform_fn = transform_fn
        self.on_activation = on_activation
        self.module_name = module_name

    def register(self, module: nn.Module) -> None:
        module_name = self.module_name  # may be None; informational

        def hook_fn(mod: nn.Module, inputs, outputs):
            # outputs can be a Tensor or a nested structure.
            # We only operate directly on Tensor outputs.
            x = outputs

            if isinstance(x, torch.Tensor):
                if self.on_activation is not None:
                    self.on_activation(x, module_name or "", mod)
                if self.transform_fn is not None:
                    x = self.transform_fn(x, module_name or "", mod)
                return x

            # If not a Tensor, we still allow on_activation for inspection,
            # but do not attempt to transform.
            if self.on_activation is not None:
                self.on_activation(x, module_name or "", mod)
            return x

        self._handle = module.register_forward_hook(hook_fn)
        self.module = module

    def __repr__(self) -> str:
        return (
            f"ActivationHook(name={self.name!r}, "
            f"module={self.module.__class__.__name__ if self.module is not None else None}, "
            f"module_name={self.module_name!r})"
        )