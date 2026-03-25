<<<<<<< HEAD
# src/loralens/hooks/__init__.py
"""
Hooks module - Activation capture system.

This module re-exports from the ``hookbox`` package, which provides
a standalone activation capture library for PyTorch models.

All imports that previously came from ``loralens.hooks`` continue to work:

    >>> from loralens.hooks import ActivationCollector, HookManager
"""

from hookbox import (
    BaseHook,
    ActivationHook,
    HookManager,
    ActivationCollector,
    CollectedActivations,
)
=======
# hooks/__init__.py

from .base_hook import BaseHook
from .activation_hook import ActivationHook
from .hook_manager import HookManager
from .registry import HookRegistry
>>>>>>> origin/main

__all__ = [
    "BaseHook",
    "ActivationHook",
    "HookManager",
<<<<<<< HEAD
    "ActivationCollector",
    "CollectedActivations",
]
=======
    "HookRegistry",
]
>>>>>>> origin/main
