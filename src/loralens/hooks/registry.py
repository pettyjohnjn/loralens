# hooks/registry.py

from __future__ import annotations

from typing import Dict, Type

from .base_hook import BaseHook


class HookRegistry:
    """
    Simple registry mapping string keys to hook classes.

    This is optional; it can be useful if you want to construct hooks
    from configuration or CLI flags, e.g.:

        registry.register("activation", ActivationHook)
        hook_cls = registry.get("activation")
        hook = hook_cls(...)

    The registry itself is hook-type-agnostic.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseHook]] = {}

    def register(self, key: str, hook_cls: Type[BaseHook]) -> None:
        if key in self._registry:
            raise ValueError(f"Hook type '{key}' is already registered.")
        self._registry[key] = hook_cls

    def get(self, key: str) -> Type[BaseHook]:
        if key not in self._registry:
            raise ValueError(
                f"Unknown hook type '{key}'. "
                f"Registered types: {sorted(self._registry.keys())}"
            )
        return self._registry[key]

    def create(self, key: str, *args, **kwargs) -> BaseHook:
        """
        Convenience constructor: registry.create("activation", name="...", ...)
        """
        hook_cls = self.get(key)
        return hook_cls(*args, **kwargs)

    def available(self):
        """
        List all registered hook type keys.
        """
        return sorted(self._registry.keys())