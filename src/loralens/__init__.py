# src/loralens/__init__.py
"""LoRA Lens - Scalable lens-based interpretability for large language models."""

__version__ = "0.3.0"

from loralens.hooks import ActivationCollector, HookManager
from loralens.losses import create_loss, BaseLoss
from loralens.lenses import create_lens, BaseLens, LogitLens, TunedLens, LoRALens
from loralens.training import LensTrainer, TrainConfig, HFUnembed, get_model_config

try:
    from loralens.ops import indexed_logits, indexed_logits_available
except ImportError:
    indexed_logits = None
    indexed_logits_available = lambda: False

__all__ = [
    "__version__",
    "ActivationCollector",
    "HookManager",
    "create_loss",
    "BaseLoss",
    "create_lens",
    "BaseLens",
    "LogitLens",
    "TunedLens",
    "LoRALens",
    "indexed_logits",
    "indexed_logits_available",
    "LensTrainer",
    "TrainConfig",
    "HFUnembed",
    "get_model_config",
]
