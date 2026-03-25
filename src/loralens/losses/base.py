# src/loralens/losses/base.py
"""
Base class for all loralens loss functions.

This extends the ``subset_kl.BaseLoss`` interface with an additional
``labels`` parameter needed by cross-entropy and other label-based losses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch

from subset_kl import ReductionType


class BaseLoss(ABC):
    """
    Abstract base class for all lens losses.

    All loss functions take student logits, teacher logits/labels,
    and an attention mask, returning a scalar (or per-token) loss.

    Parameters
    ----------
    reduction : str
        How to reduce the loss: "none", "mean", or "sum".
    """

    def __init__(self, reduction: ReductionType = "mean") -> None:
        self.reduction = reduction

    @abstractmethod
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        student_logits : torch.Tensor
            Logits from the lens, shape [batch, seq, vocab].
        teacher_logits : torch.Tensor
            Logits from the teacher model, shape [batch, seq, vocab].
        attention_mask : Optional[torch.Tensor]
            Mask indicating valid tokens, shape [batch, seq].
        labels : Optional[torch.Tensor]
            Target token IDs (for CE loss), shape [batch, seq].

        Returns
        -------
        torch.Tensor
            Loss value (scalar if reduction != "none").
        """
        raise NotImplementedError

    def __call__(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss (calls forward)."""
        return self.forward(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _apply_reduction(
        self,
        loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply reduction to per-token loss."""
        if self.reduction == "none":
            return loss

        if mask is not None:
            loss = loss * mask.to(loss.dtype)

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "mean":
            if mask is not None:
                denom = mask.sum().clamp_min(1.0)
            else:
                denom = loss.numel()
            return loss.sum() / denom

        raise ValueError(f"Unknown reduction: {self.reduction}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction={self.reduction!r})"
