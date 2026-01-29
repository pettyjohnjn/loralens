# src/loralens/hooks/base.py
"""Base classes for hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn
from torch.utils.hooks import RemovableHandle


class BaseHook(ABC):
    """
    Abstract base class for all hooks.
    
    Subclasses must implement `register()` to attach the hook to a module.
    
    Attributes
    ----------
    name : str
        Unique identifier for this hook.
    module : Optional[nn.Module]
        The module this hook is attached to (None if not registered).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._handle: Optional[RemovableHandle] = None
        self.module: Optional[nn.Module] = None

    @abstractmethod
    def register(self, module: nn.Module) -> None:
        """
        Attach the hook to a module.
        
        Must set self._handle and self.module.
        """
        raise NotImplementedError

    def unregister(self) -> None:
        """Remove the hook from its module."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.module = None

    @property
    def is_registered(self) -> bool:
        """Whether this hook is currently attached to a module."""
        return self._handle is not None

    def __repr__(self) -> str:
        status = "registered" if self.is_registered else "unregistered"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"
