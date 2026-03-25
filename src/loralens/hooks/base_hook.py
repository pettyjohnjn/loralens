# hooks/base_hook.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import torch.nn as nn


class BaseHook(ABC):
    """
    Base class for all hooks.

    Subclasses should:
      - implement `register` to attach a PyTorch hook to a module
      - set `self._handle` to the returned handle
      - set `self.module` to the attached nn.Module
    """

    def __init__(self, name: str):
        self.name: str = name
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None  # type: ignore[attr-defined]
        self.module: Optional[nn.Module] = None

    @abstractmethod
    def register(self, module: nn.Module) -> None:
        """
        Attach the hook to the given module.
        Must set self._handle and self.module.
        """
        raise NotImplementedError

    def unregister(self) -> None:
        """
        Remove the hook from its module, if attached.
        """
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.module = None

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError