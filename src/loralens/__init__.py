# src/loralens/__init__.py
"""LoRA Lens - Scalable lens-based interpretability for large language models."""

__version__ = "0.3.0"

from loralens.hooks import ActivationCollector
from loralens.losses import create_loss, BaseLoss
from loralens.lenses import create_lens, BaseLens, LogitLens, TunedLens, LoRALens
from loralens.training import LensTrainer, TrainConfig, HFUnembed, get_model_config

try:
    from loralens.ops import indexed_logits, indexed_logits_available
except ImportError:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "loralens.ops (CUDA extension for indexed_logits) is not available. "
        "Falling back to torch.gather. "
        "To build the extension: cd src/loralens/ops && python setup.py build_ext --inplace"
    )
    indexed_logits = None
    indexed_logits_available = lambda: False

__all__ = [
    "__version__",
    "ActivationCollector",
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
