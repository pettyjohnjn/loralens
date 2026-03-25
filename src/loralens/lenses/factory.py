# src/loralens/lenses/factory.py
"""Factory for creating lenses."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Type

import torch.nn as nn

from .base import BaseLens
from .types import LayerId
from .logit_lens import LogitLens
from .tuned_lens import TunedLens
from .lora_lens import LoRALens


# Global registry
_LENS_REGISTRY: Dict[str, Type[BaseLens]] = {}


def register_lens(name: str) -> Callable[[Type[BaseLens]], Type[BaseLens]]:
    """
    Decorator to register a lens class.

    Example
    -------
    >>> @register_lens("my_lens")
    >>> class MyLens(BaseLens):
    ...     pass
    """
    def decorator(cls: Type[BaseLens]) -> Type[BaseLens]:
        _LENS_REGISTRY[name] = cls
        return cls
    return decorator


def create_lens(
    name: str,
    layer_ids: Optional[Iterable[LayerId]] = None,
    hidden_size: Optional[int] = None,
    unembed: Optional[nn.Module] = None,
    **kwargs: Any,
) -> BaseLens:
    """
    Create a lens by name.

    Parameters
    ----------
    name : str
        Name of the lens. One of:
        - "logit": LogitLens (no trainable params)
        - "tuned": TunedLens (full-rank translators)
        - "lora": LoRALens (low-rank translators)
    layer_ids : Optional[Iterable[LayerId]]
        Layer identifiers (required for tuned/lora).
    hidden_size : Optional[int]
        Hidden dimension (required for tuned/lora).
    unembed : Optional[nn.Module]
        Unembedding module (required).
    **kwargs
        Additional arguments for specific lens types.

    Returns
    -------
    BaseLens
        Instantiated lens.

    Examples
    --------
    >>> lens = create_lens("logit", unembed=model.lm_head)
    >>> lens = create_lens(
    ...     "lora",
    ...     layer_ids=range(12),
    ...     hidden_size=768,
    ...     unembed=unembed,
    ...     r=16,
    ... )
    """
    if name not in _LENS_REGISTRY:
        raise ValueError(
            f"Unknown lens: {name!r}. Available: {list_lenses()}"
        )

    cls = _LENS_REGISTRY[name]

    # Handle different lens signatures
    if name == "logit":
        if unembed is None:
            raise ValueError("LogitLens requires unembed")
        return cls(unembed=unembed, **kwargs)

    elif name in ("tuned", "lora"):
        if layer_ids is None:
            raise ValueError(f"{name} lens requires layer_ids")
        if hidden_size is None:
            raise ValueError(f"{name} lens requires hidden_size")
        if unembed is None:
            raise ValueError(f"{name} lens requires unembed")
        return cls(
            layer_ids=layer_ids,
            hidden_size=hidden_size,
            unembed=unembed,
            **kwargs,
        )

    else:
        # Generic instantiation for custom lenses
        return cls(**kwargs)


def list_lenses() -> List[str]:
    """Return list of registered lens names."""
    return sorted(_LENS_REGISTRY.keys())


def get_lens_class(name: str) -> Type[BaseLens]:
    """Get the class for a lens name."""
    if name not in _LENS_REGISTRY:
        raise ValueError(f"Unknown lens: {name!r}")
    return _LENS_REGISTRY[name]


# Register built-in lenses
register_lens("logit")(LogitLens)
register_lens("tuned")(TunedLens)
register_lens("lora")(LoRALens)
