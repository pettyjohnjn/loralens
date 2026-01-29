# src/loralens/hooks/manager.py
"""Hook manager for attaching/detaching multiple hooks."""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch.nn as nn

from .base import BaseHook
from .activation_hook import ActivationHook, CallbackFn, TransformFn


# Type aliases
ModulePredicate = Callable[[str, nn.Module], bool]
HookFactory = Callable[[str, nn.Module], BaseHook]


class HookManager:
    """
    Manages multiple hooks attached to a model.
    
    Provides methods to:
    - Attach hooks to specific modules by name
    - Attach hooks to modules matching a predicate
    - Remove hooks individually or all at once
    - List active hooks
    
    Parameters
    ----------
    model : nn.Module
        The model to manage hooks for.
        
    Examples
    --------
    >>> manager = HookManager(model)
    >>> 
    >>> # Add hooks to all MLP layers
    >>> manager.add_hooks_by_predicate(
    ...     predicate=lambda name, mod: "mlp" in name,
    ...     hook_factory=lambda name, mod: ActivationHook(name=name)
    ... )
    >>> 
    >>> # Run model...
    >>> 
    >>> # Clean up
    >>> manager.remove_all()
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._hooks: Dict[str, BaseHook] = {}

    def add_hook(self, module_name: str, hook: BaseHook) -> None:
        """
        Attach a hook to a named module.
        
        Parameters
        ----------
        module_name : str
            Dot-separated path to module (e.g., "transformer.h.0.mlp").
        hook : BaseHook
            Hook instance to attach.
            
        Raises
        ------
        ValueError
            If module not found or hook name already registered.
        """
        # Find module
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found in model.")

        # Check for name collision
        if hook.name in self._hooks:
            raise ValueError(f"Hook '{hook.name}' already registered.")

        # Set module_name on hook if it supports it
        if hasattr(hook, "module_name") and hook.module_name is None:
            hook.module_name = module_name

        # Register and track
        hook.register(module)
        self._hooks[hook.name] = hook

    def add_hooks_by_predicate(
        self,
        predicate: ModulePredicate,
        hook_factory: HookFactory,
    ) -> int:
        """
        Attach hooks to all modules matching a predicate.
        
        Parameters
        ----------
        predicate : Callable[[str, nn.Module], bool]
            Function that returns True for modules that should get hooks.
        hook_factory : Callable[[str, nn.Module], BaseHook]
            Function that creates a hook for a given module.
            
        Returns
        -------
        int
            Number of hooks added.
        """
        count = 0
        for name, module in self.model.named_modules():
            if predicate(name, module):
                hook = hook_factory(name, module)
                
                if hasattr(hook, "module_name") and hook.module_name is None:
                    hook.module_name = name

                hook.register(module)
                
                if hook.name in self._hooks:
                    raise ValueError(f"Hook '{hook.name}' already registered.")
                self._hooks[hook.name] = hook
                count += 1
                
        return count

    def add_activation_hooks(
        self,
        predicate: ModulePredicate,
        *,
        on_activation: Optional[CallbackFn] = None,
        transform_fn: Optional[TransformFn] = None,
        name_prefix: str = "activation",
    ) -> int:
        """
        Convenience method to add ActivationHooks to matching modules.
        
        Parameters
        ----------
        predicate : ModulePredicate
            Function that selects which modules get hooks.
        on_activation : Optional[CallbackFn]
            Callback for each activation.
        transform_fn : Optional[TransformFn]
            Transform function for interventions.
        name_prefix : str
            Prefix for generated hook names.
            
        Returns
        -------
        int
            Number of hooks added.
        """
        def factory(module_name: str, module: nn.Module) -> BaseHook:
            return ActivationHook(
                name=f"{name_prefix}:{module_name}",
                on_activation=on_activation,
                transform_fn=transform_fn,
                module_name=module_name,
            )

        return self.add_hooks_by_predicate(predicate, factory)

    def remove_hook(self, hook_name: str) -> bool:
        """
        Remove a hook by name.
        
        Returns True if hook was found and removed.
        """
        hook = self._hooks.get(hook_name)
        if hook is None:
            return False
        hook.unregister()
        del self._hooks[hook_name]
        return True

    def remove_all(self) -> int:
        """
        Remove all hooks.
        
        Returns number of hooks removed.
        """
        count = len(self._hooks)
        for hook in self._hooks.values():
            hook.unregister()
        self._hooks.clear()
        return count

    def list_hooks(self) -> List[str]:
        """Return list of active hook names."""
        return list(self._hooks.keys())

    def get_hook(self, name: str) -> Optional[BaseHook]:
        """Get a hook by name."""
        return self._hooks.get(name)

    def __len__(self) -> int:
        return len(self._hooks)

    def __iter__(self) -> Iterator[Tuple[str, BaseHook]]:
        return iter(self._hooks.items())

    def __contains__(self, name: str) -> bool:
        return name in self._hooks

    def __repr__(self) -> str:
        return f"HookManager(model={self.model.__class__.__name__}, hooks={len(self)})"
