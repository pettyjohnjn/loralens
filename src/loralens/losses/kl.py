# src/loralens/losses/kl.py
"""
KL divergence loss for lens training.

Thin adapter around ``subset_kl.KLDivergenceLoss`` that conforms
to the loralens ``BaseLoss`` interface (with ``labels`` parameter).
"""

from __future__ import annotations

from typing import Optional

import torch

from subset_kl import KLDivergenceLoss as _ExternalKL
from subset_kl import ReductionType

from .base import BaseLoss


class KLDivergenceLoss(BaseLoss):
    """
    KL divergence loss: KL(teacher || student).

    Delegates computation to ``subset_kl.KLDivergenceLoss`` while
    providing the loralens ``BaseLoss`` interface (``labels`` parameter).

    Parameters
    ----------
    reduction : str
        How to reduce: "none", "mean", or "sum".
    temperature : float
        Temperature for softmax (1.0 = no scaling).
    chunk_size : Optional[int]
        If set, compute KL in chunks along sequence dimension
        to reduce peak memory usage.
    """

    def __init__(
        self,
        reduction: ReductionType = "mean",
        temperature: float = 1.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.chunk_size = chunk_size
        self._inner = _ExternalKL(
            reduction=reduction,
            temperature=temperature,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL(teacher || student).

        ``labels`` is accepted for interface compatibility but unused.
        """
        return self._inner.forward(student_logits, teacher_logits, attention_mask)

    def __repr__(self) -> str:
        return (
            f"KLDivergenceLoss(reduction={self.reduction!r}, "
            f"temperature={self.temperature}, chunk_size={self.chunk_size})"
        )
