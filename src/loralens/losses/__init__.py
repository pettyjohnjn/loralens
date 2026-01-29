# src/loralens/losses/__init__.py
"""
Losses module - Standalone loss computation for lens training.

This module provides pluggable loss functions that can be used
independently of the training loop. Each loss implements a common
interface for easy swapping.

Loss Types
----------
- KLDivergenceLoss: Full-vocabulary KL divergence
- SubsetKLLoss: Memory-efficient top-k (or Hajek) subset KL
- SharedSubsetKLLoss: Memory-efficient shared candidate set KL
- CrossEntropyLoss: Next-token prediction CE loss

Sampling Utilities (for advanced users)
---------------------------------------
- pps_sample_indices_batched: PPS sampling for Hajek estimator
- hajek_kl_estimate: Self-normalized importance sampling
- head_tail_kl: Simplified head-tail KL interface

Example usage:
    
    # Simple top-k (recommended)
    loss_fn = SubsetKLLoss(k=256)
    
    # Shared subset (most memory efficient)
    loss_fn = SharedSubsetKLLoss(top_m=16, max_K=512)
    
    # Hajek estimator (advanced)
    loss_fn = SubsetKLLoss(k=128, mode="hajek", k_tail=64)
    
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

# Advanced sampling utilities
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
