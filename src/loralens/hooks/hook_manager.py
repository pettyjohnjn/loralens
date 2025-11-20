# hooks/hook_manager.py

from __future__ import annotations

from typing import Callable, Dict, List, Optional
import torch.nn as nn

from .base_hook import BaseHook
from .activation_hook import ActivationHook, ActivationTransformFn, ActivationCallbackFn


ModulePredicate = Callable[[str, nn.Module], bool]
HookFactory = Callable[[str, nn.Module], BaseHook]


class HookManager:
    """
    Manages hooks attached to a model.

    Responsibilities:
      - Attach hooks to specific modules or to modules matching a predicate.
      - Track active hooks by name.
      - Remove hooks individually or all at once.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.active_hooks: Dict[str, BaseHook] = {}

    # Generic hook management

    def add_hook(self, module_name: str, hook: BaseHook) -> None:
        """
        Attach an existing hook instance to the named module.
        """
        module = dict(self.model.named_modules()).get(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found in model.")

        # Allow the hook to optionally know its module_name
        if hasattr(hook, "module_name") and getattr(hook, "module_name") is None:
            setattr(hook, "module_name", module_name)

        hook.register(module)
        if hook.name in self.active_hooks:
            raise ValueError(f"Hook name '{hook.name}' already registered.")
        self.active_hooks[hook.name] = hook

    def add_hooks_by_predicate(self, predicate: ModulePredicate, hook_factory: HookFactory) -> None:
        """
        Attach hooks to every module for which predicate(name, module) is True.

        hook_factory(name, module) should return a BaseHook instance for that module.
        """
        for name, module in self.model.named_modules():
            if predicate(name, module):
                hook = hook_factory(name, module)
                # Allow hook to receive module_name if it supports it
                if hasattr(hook, "module_name") and getattr(hook, "module_name") is None:
                    setattr(hook, "module_name", name)

                hook.register(module)
                if hook.name in self.active_hooks:
                    raise ValueError(f"Hook name '{hook.name}' already registered.")
                self.active_hooks[hook.name] = hook

    def remove_hook(self, hook_name: str) -> None:
        """
        Remove a hook by its name, if present.
        """
        hook = self.active_hooks.get(hook_name)
        if hook is None:
            return
        hook.unregister()
        del self.active_hooks[hook_name]

    def remove_all(self) -> None:
        """
        Remove all active hooks.
        """
        for hook in list(self.active_hooks.values()):
            hook.unregister()
        self.active_hooks.clear()

    def list_hooks(self) -> List[str]:
        """
        Return a list of active hook names.
        """
        return list(self.active_hooks.keys())

    # Convenience: attach ActivationHooks directly

    def add_activation_hooks(
        self,
        predicate: ModulePredicate,
        *,
        transform_fn: Optional[ActivationTransformFn] = None,
        on_activation: Optional[ActivationCallbackFn] = None,
        name_prefix: str = "activation",
    ) -> None:
        """
        Attach ActivationHook instances to all modules matching `predicate`.

        Args:
            predicate:
                Function (module_name, module) -> bool; selects which modules
                receive an ActivationHook.
            transform_fn:
                Optional function that transforms the activation (see ActivationHook).
            on_activation:
                Optional callback invoked with each activation (see ActivationHook).
            name_prefix:
                Prefix used to generate hook names; final name is
                f"{name_prefix}:{module_name}".
        """

        def factory(module_name: str, module: nn.Module) -> BaseHook:
            hook_name = f"{name_prefix}:{module_name}"
            return ActivationHook(
                name=hook_name,
                transform_fn=transform_fn,
                on_activation=on_activation,
                module_name=module_name,
            )

        self.add_hooks_by_predicate(predicate, factory)