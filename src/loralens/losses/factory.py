# src/loralens/losses/factory.py
"""Factory for creating loss functions."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

from .base import BaseLoss
from .kl import KLDivergenceLoss
from .subset_kl import SubsetKLLoss
from .shared_subset_kl import SharedSubsetKLLoss
from .cross_entropy import CrossEntropyLoss


# Global registry
_LOSS_REGISTRY: Dict[str, Type[BaseLoss]] = {}


def register_loss(name: str) -> Callable[[Type[BaseLoss]], Type[BaseLoss]]:
    """
    Decorator to register a loss class.

    Example
    -------
    >>> @register_loss("my_loss")
    >>> class MyLoss(BaseLoss):
    ...     pass
    """
    def decorator(cls: Type[BaseLoss]) -> Type[BaseLoss]:
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def create_loss(name: str, **kwargs: Any) -> BaseLoss:
    """
    Create a loss function by name.

    Parameters
    ----------
    name : str
        Name of the loss function. One of:
        - "kl": Full KL divergence
        - "subset_kl": Per-position top-k subset KL
        - "shared_subset_kl": Shared candidate set KL (most memory efficient)
        - "ce" or "cross_entropy": Cross-entropy loss
    **kwargs
        Arguments passed to the loss constructor.

    Returns
    -------
    BaseLoss
        Instantiated loss function.

    Examples
    --------
    >>> loss_fn = create_loss("kl", chunk_size=128)
    >>> loss_fn = create_loss("subset_kl", k=256)
    >>> loss_fn = create_loss("shared_subset_kl", top_m=16, max_K=512)
    >>> loss_fn = create_loss("ce", label_smoothing=0.1)
    """
    if name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {name!r}. "
            f"Available: {list_losses()}"
        )
    return _LOSS_REGISTRY[name](**kwargs)


def list_losses() -> List[str]:
    """Return list of registered loss names."""
    return sorted(_LOSS_REGISTRY.keys())


def get_loss_class(name: str) -> Type[BaseLoss]:
    """Get the class for a loss name."""
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name!r}")
    return _LOSS_REGISTRY[name]


# Register built-in losses
register_loss("kl")(KLDivergenceLoss)
register_loss("kl_divergence")(KLDivergenceLoss)
register_loss("subset_kl")(SubsetKLLoss)
register_loss("shared_subset_kl")(SharedSubsetKLLoss)
register_loss("ce")(CrossEntropyLoss)
register_loss("cross_entropy")(CrossEntropyLoss)
