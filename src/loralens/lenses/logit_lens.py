from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseLens
from .types import LayerId


class LogitLens(BaseLens):
    """
    Simple logit lens: directly apply the model's readout/unembedding to
    hidden states without any learned transformation.

    This corresponds to the classic "logit lens" where you reuse the
    existing LM head at intermediate layers.
    """

    def __init__(
        self,
        readout: nn.Module,
        *,
        vocab_size: Optional[int] = None,
        ignore_index: int = -100,
        loss_reduction: str = "mean",
    ) -> None:
        if not hasattr(readout, "forward"):
            raise TypeError("`readout` must be an nn.Module with a `forward` method.")

        if vocab_size is None:
            if isinstance(readout, nn.Linear):
                vocab_size = readout.out_features
            else:
                raise ValueError(
                    "Could not infer vocab_size automatically. "
                    "Pass `vocab_size` explicitly."
                )

        super().__init__(
            vocab_size=vocab_size,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
        )
        self.readout = readout

    def compute_logits(
        self,
        activations: torch.Tensor,
        *,
        layer: Optional[LayerId] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply the provided readout head directly to activations.
        """
        batch, seq, hidden = activations.shape

        flat = activations.reshape(batch * seq, hidden)
        flat_logits = self.readout(flat)  # [batch*seq, vocab]
        logits = flat_logits.reshape(batch, seq, -1)
        return logits