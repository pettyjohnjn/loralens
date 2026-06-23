# src/loralens/losses/__init__.py
"""
Losses module - Pluggable loss functions for lens training.

Loss Types
----------
- KLDivergenceLoss: Full-vocabulary KL divergence (via subset-kl)
- SubsetKLLoss: Memory-efficient top-k / MC subset KL (via subset-kl)
- CrossEntropyLoss: Next-token prediction CE loss

Example usage::

    # Simple top-k (recommended)
    loss_fn = SubsetKLLoss(k=256)

    # Compute loss
    loss = loss_fn(
        student_logits=lens_output,
        teacher_logits=model_output,
        attention_mask=mask,
    )
"""

from .base import BaseLoss
from .kl import KLDivergenceLoss
from .subset_kl import SubsetKLLoss
from .cross_entropy import CrossEntropyLoss
from .factory import create_loss, register_loss, list_losses

__all__ = [
    # Core losses
    "BaseLoss",
    "KLDivergenceLoss",
    "SubsetKLLoss",
    "CrossEntropyLoss",
    # Factory functions
    "create_loss",
    "register_loss",
    "list_losses",
]
