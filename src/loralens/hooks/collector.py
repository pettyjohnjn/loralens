# src/loralens/hooks/collector.py
"""
High-level activation collection interface.

This module provides the main user-facing API for collecting activations
from transformer models. It can be used as a context manager for clean
resource management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .manager import HookManager
from .activation_hook import ActivationHook


@dataclass
class CollectedActivations:
    """
    Container for collected activations from a forward pass.
    
    Attributes
    ----------
    hidden_states : Tuple[torch.Tensor, ...]
        Hidden states from each layer (including embeddings).
    logits : torch.Tensor
        Final output logits.
    custom : Dict[str, torch.Tensor]
        Custom activations captured by user-defined hooks.
    """
    hidden_states: Tuple[torch.Tensor, ...] = field(default_factory=tuple)
    logits: Optional[torch.Tensor] = None
    custom: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    @property
    def num_layers(self) -> int:
        """Number of layers (excluding embeddings)."""
        return len(self.hidden_states) - 1 if self.hidden_states else 0
    
    def get_layer(self, layer_id: Union[int, str]) -> torch.Tensor:
        """
        Get hidden states for a specific layer.
        
        Parameters
        ----------
        layer_id : int or str
            Layer index (0-based, after embeddings).
            
        Returns
        -------
        torch.Tensor
            Hidden states of shape [batch, seq, hidden].
        """
        idx = int(layer_id) + 1  # +1 to skip embeddings
        return self.hidden_states[idx]
    
    def iter_layers(self) -> Iterator[Tuple[int, torch.Tensor]]:
        """
        Iterate over layers (excluding embeddings).
        
        Yields
        ------
        Tuple[int, torch.Tensor]
            (layer_id, hidden_states) pairs.
        """
        for i, hidden in enumerate(self.hidden_states[1:]):
            yield i, hidden
            
    def __repr__(self) -> str:
        return (
            f"CollectedActivations(num_layers={self.num_layers}, "
            f"custom_keys={list(self.custom.keys())})"
        )


class ActivationCollector:
    """
    High-level interface for collecting activations from a model.
    
    Can be used as a context manager for automatic cleanup, or
    manually with attach()/detach() methods.
    
    Parameters
    ----------
    model : nn.Module
        The model to collect activations from.
    custom_hooks : Optional[Dict[str, str]]
        Mapping of custom names to module paths for additional
        activation capture beyond hidden states.
        
    Examples
    --------
    >>> # As context manager (recommended)
    >>> collector = ActivationCollector(model)
    >>> with collector:
    ...     data = collector.collect(input_ids, attention_mask)
    ...     for layer_id, hidden in data.iter_layers():
    ...         print(f"Layer {layer_id}: {hidden.shape}")
    
    >>> # Manual management
    >>> collector = ActivationCollector(model)
    >>> collector.attach()
    >>> data = collector.collect(input_ids, attention_mask)
    >>> collector.detach()
    
    >>> # With custom hooks
    >>> collector = ActivationCollector(
    ...     model,
    ...     custom_hooks={
    ...         "attn_0": "transformer.h.0.attn.c_proj",
    ...         "mlp_0": "transformer.h.0.mlp.c_proj",
    ...     }
    ... )
    """

    def __init__(
        self,
        model: nn.Module,
        custom_hooks: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.custom_hooks = custom_hooks or {}
        self._manager = HookManager(model)
        self._custom_activations: Dict[str, torch.Tensor] = {}
        self._attached = False

    def attach(self) -> "ActivationCollector":
        """
        Attach custom hooks to the model.
        
        Note: Hidden states are collected via output_hidden_states=True,
        not via hooks, for efficiency.
        
        Returns self for method chaining.
        """
        if self._attached:
            return self
            
        # Add custom hooks
        for name, module_path in self.custom_hooks.items():
            def make_callback(capture_name: str):
                def callback(x, module_name, module):
                    self._custom_activations[capture_name] = x.detach()
                return callback
            
            hook = ActivationHook(
                name=f"collector:{name}",
                on_activation=make_callback(name),
                module_name=module_path,
            )
            self._manager.add_hook(module_path, hook)
        
        self._attached = True
        return self

    def detach(self) -> None:
        """Remove all hooks from the model."""
        self._manager.remove_all()
        self._custom_activations.clear()
        self._attached = False

    def collect(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> CollectedActivations:
        """
        Run a forward pass and collect activations.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [batch, seq].
        attention_mask : Optional[torch.Tensor]
            Attention mask of shape [batch, seq].
        **model_kwargs
            Additional arguments passed to model.forward().
            
        Returns
        -------
        CollectedActivations
            Container with hidden states, logits, and custom activations.
        """
        # Clear previous custom activations
        self._custom_activations.clear()
        
        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
                **model_kwargs,
            )
        
        return CollectedActivations(
            hidden_states=outputs.hidden_states,
            logits=outputs.logits,
            custom=dict(self._custom_activations),
        )

    def collect_with_grad(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> CollectedActivations:
        """
        Run a forward pass and collect activations WITH gradients.
        
        Use this during training when you need gradients to flow
        through the activations.
        """
        self._custom_activations.clear()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **model_kwargs,
        )
        
        return CollectedActivations(
            hidden_states=outputs.hidden_states,
            logits=outputs.logits,
            custom=dict(self._custom_activations),
        )

    def __enter__(self) -> "ActivationCollector":
        """Context manager entry."""
        return self.attach()

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.detach()

    def __repr__(self) -> str:
        status = "attached" if self._attached else "detached"
        return (
            f"ActivationCollector(model={self.model.__class__.__name__}, "
            f"custom_hooks={len(self.custom_hooks)}, {status})"
        )
