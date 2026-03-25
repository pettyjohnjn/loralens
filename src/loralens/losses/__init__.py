# src/loralens/losses/__init__.py
"""
Losses module - Pluggable loss functions for lens training.

Loss Types
----------
- KLDivergenceLoss: Full-vocabulary KL divergence (via subset-kl)
- SubsetKLLoss: Memory-efficient top-k subset KL (via subset-kl)
- SharedSubsetKLLoss: Memory-efficient shared candidate set KL
- CrossEntropyLoss: Next-token prediction CE loss

Example usage::

    # Simple top-k (recommended)
    loss_fn = SubsetKLLoss(k=256)

    # Shared subset (most memory efficient)
    loss_fn = SharedSubsetKLLoss(top_m=16, max_K=512)

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
from .shared_subset_kl import SharedSubsetKLLoss
from .cross_entropy import CrossEntropyLoss
from .factory import create_loss, register_loss, list_losses

# Advanced sampling utilities (re-exported from subset_kl)
from .sampling import (
    pps_sample_indices_batched,
    hajek_kl_estimate,
    head_tail_kl,
    SamplingDiagnostics,
)

__all__ = [
    # Core losses
    "BaseLoss",
    "KLDivergenceLoss",
    "SubsetKLLoss",
    "SharedSubsetKLLoss",
    "CrossEntropyLoss",
    # Factory functions
    "create_loss",
    "register_loss",
    "list_losses",
    # Sampling utilities (advanced)
    "pps_sample_indices_batched",
    "hajek_kl_estimate",
    "head_tail_kl",
    "SamplingDiagnostics",
]
