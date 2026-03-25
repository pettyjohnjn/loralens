<<<<<<< HEAD
# src/loralens/lenses/logit_lens.py
"""
Logit Lens - Direct unembedding without learned transformation.

The simplest lens: just apply the model's unembedding to intermediate
hidden states without any additional parameters.
"""

=======
>>>>>>> origin/main
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseLens
from .types import LayerId


class LogitLens(BaseLens):
    """
<<<<<<< HEAD
    Logit lens: direct application of unembedding to hidden states.

    This is the original "logit lens" approach where intermediate
    hidden states are directly projected to vocabulary space using
    the model's final layer norm and language model head.

    Parameters
    ----------
    unembed : nn.Module
        Unembedding module (typically includes layer norm + LM head).
    vocab_size : Optional[int]
        Vocabulary size. Inferred from unembed if not provided.
    freeze_unembed : bool
        Whether to freeze unembed parameters (default True).

    Notes
    -----
    The logit lens has zero trainable parameters - it purely reuses
    the model's existing unembedding weights.
=======
    Simple logit lens: directly apply the model's readout/unembedding to
    hidden states without any learned transformation.

    This corresponds to the classic "logit lens" where you reuse the
    existing LM head at intermediate layers.
>>>>>>> origin/main
    """

    def __init__(
        self,
<<<<<<< HEAD
        unembed: nn.Module,
        vocab_size: Optional[int] = None,
        freeze_unembed: bool = True,
    ) -> None:
        # Infer vocab size
        if vocab_size is None:
            if hasattr(unembed, "vocab_size"):
                vocab_size = unembed.vocab_size
            elif isinstance(unembed, nn.Linear):
                vocab_size = unembed.out_features
            else:
                raise ValueError("Cannot infer vocab_size; provide it explicitly.")

        super().__init__(vocab_size=vocab_size)

        self.unembed = unembed

        if freeze_unembed:
            for p in self.unembed.parameters():
                p.requires_grad = False
=======
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
>>>>>>> origin/main

    def compute_logits(
        self,
        activations: torch.Tensor,
<<<<<<< HEAD
        layer: Optional[LayerId] = None,
    ) -> torch.Tensor:
        """
        Apply unembedding to activations.

        The layer parameter is accepted but ignored since logit lens
        uses the same (frozen) unembedding for all layers.
        """
        batch, seq, hidden = activations.shape

        # Flatten for unembedding
        flat = activations.view(batch * seq, hidden)
        flat_logits = self.unembed(flat)

        return flat_logits.view(batch, seq, -1)
=======
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
>>>>>>> origin/main
