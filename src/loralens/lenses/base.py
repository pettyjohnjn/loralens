from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import LayerId, LensOutput


class BaseLens(nn.Module, ABC):
    """
    Abstract base class for all lenses: activations -> logits.

    Subclasses implement `compute_logits` (activations -> logits).
    This base class optionally computes cross-entropy loss vs. labels.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        ignore_index: int = -100,
        loss_reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.loss_reduction = loss_reduction

    @abstractmethod
    def compute_logits(
        self,
        activations: torch.Tensor,
        *,
        layer: Optional[LayerId] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Map activations to logits.

        Parameters
        ----------
        activations:
            Hidden states of shape [batch, seq, hidden].
        layer:
            Optional layer identifier used to select the appropriate
            per-layer parameters (if applicable).

        Returns
        -------
        logits:
            Tensor of shape [batch, seq, vocab_size].
        """
        raise NotImplementedError

    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional attention mask.
        Expects labels of shape [batch, seq] and logits [batch, seq, vocab].
        """
        reduction = reduction or self.loss_reduction

        vocab = logits.size(-1)
        logits_flat = logits.reshape(-1, vocab)
        labels_flat = labels.reshape(-1)

        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        if attention_mask is not None:
            # attention_mask: [batch, seq] -> [batch*seq]
            mask_flat = attention_mask.reshape(-1).to(loss.dtype)
            loss = loss * mask_flat

        if reduction == "none":
            return loss.reshape_as(labels)
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            if attention_mask is None:
                denom = (labels_flat != self.ignore_index).float()
            else:
                denom = attention_mask.reshape(-1) * (
                    labels_flat != self.ignore_index
                ).float()
            denom = denom.sum().clamp_min(1.0)
            return loss.sum() / denom
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(
        self,
        activations: torch.Tensor,
        *,
        layer: Optional[LayerId] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: Optional[str] = None,
        return_logits: bool = True,
        return_loss: bool = False,
        **kwargs: Any,
    ) -> LensOutput:
        """
        High-level interface used by hooks/runners.

        Parameters
        ----------
        activations:
            Hidden states [batch, seq, hidden].
        layer:
            Optional layer identifier.
        labels:
            Optional target ids [batch, seq] for CE loss.
        attention_mask:
            Optional [batch, seq] bool or 0/1 mask.
        reduction:
            Optional loss reduction override.
        return_logits:
            If False, logits will be omitted from the output.
        return_loss:
            If True and labels are provided, compute CE loss here.
        """
        logits = self.compute_logits(activations, layer=layer, **kwargs)

        loss = None
        if return_loss and labels is not None:
            loss = self._compute_ce_loss(
                logits,
                labels,
                attention_mask=attention_mask,
                reduction=reduction,
            )

        return LensOutput(
            logits=logits if return_logits else None,
            loss=loss,
            extra={},
        )