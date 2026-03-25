# src/loralens/losses/cross_entropy.py
"""Cross-entropy loss for lens training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss, ReductionType


class CrossEntropyLoss(BaseLoss):
    """
    Cross-entropy loss against target labels.

    Unlike KL divergence losses, this computes loss against the
    ground-truth token labels rather than the teacher's distribution.

    Parameters
    ----------
    reduction : str
        How to reduce: "none", "mean", or "sum".
    ignore_index : int
        Token ID to ignore in loss computation (typically pad token).
    label_smoothing : float
        Label smoothing factor (0.0 = no smoothing).
    """

    def __init__(
        self,
        reduction: ReductionType = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Parameters
        ----------
        student_logits : torch.Tensor
            Shape [batch, seq, vocab].
        teacher_logits : torch.Tensor
            Unused (provided for interface consistency).
        attention_mask : Optional[torch.Tensor]
            Shape [batch, seq]. Used to mask padding.
        labels : Optional[torch.Tensor]
            Shape [batch, seq]. Target token IDs.
            If None, uses argmax of teacher_logits.

        Returns
        -------
        torch.Tensor
            Cross-entropy loss.
        """
        batch, seq, vocab = student_logits.shape

        # Get labels from teacher if not provided
        if labels is None:
            with torch.no_grad():
                labels = teacher_logits.argmax(dim=-1)

        # Flatten for F.cross_entropy
        logits_flat = student_logits.view(-1, vocab)
        labels_flat = labels.view(-1)

        # Compute loss
        loss_flat = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Reshape and apply mask
        loss = loss_flat.view(batch, seq)

        # Apply attention mask (in addition to ignore_index)
        if attention_mask is not None:
            loss = loss * attention_mask.to(loss.dtype)

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            if attention_mask is not None:
                valid_count = attention_mask.sum()
                # Also account for ignore_index tokens
                ignored = (labels == self.ignore_index).to(loss.dtype)
                valid_count = valid_count - (ignored * attention_mask.to(ignored.dtype)).sum()
                return loss.sum() / valid_count.clamp_min(1.0)
            else:
                # Let ignore_index handle masking
                valid = (labels != self.ignore_index).sum()
                return loss.sum() / valid.clamp_min(1.0)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def __repr__(self) -> str:
        return (
            f"CrossEntropyLoss(reduction={self.reduction!r}, "
            f"ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing})"
        )
