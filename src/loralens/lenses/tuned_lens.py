# src/loralens/lenses/tuned_lens.py
"""
Tuned Lens - Per-layer learned translators.

Implements the tuned lens from Belrose et al. (2023): each layer
gets a learned affine translator that maps hidden states to a
"more final" representation before unembedding.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .base import BaseLens
from .types import LayerId, canonical_layer_id, module_safe_layer_key
from ._unembed import (
    subset_logits_and_logsumexp_from_flat,
    subset_logits_from_flat,
)


class TunedLens(BaseLens):
    """
    Tuned lens with per-layer residual translators.

    For each layer, applies:
        h' = h + translator(h)
        logits = unembed(h')

    The translators are initialized to zero so the lens starts
    as an identity mapping (equivalent to logit lens).

    Parameters
    ----------
    layer_ids : Iterable[LayerId]
        Layer identifiers to create translators for.
    hidden_size : int
        Dimension of hidden states.
    unembed : nn.Module
        Unembedding module.
    vocab_size : Optional[int]
        Vocabulary size (inferred from unembed if not provided).
    bias : bool
        Whether translators have bias terms.
    init_identity : bool
        If True, initialize translators to zero (identity mapping).

    Notes
    -----
    Parameter count scales as O(num_layers * hidden_size^2), which
    can be prohibitive for large models. See LoRALens for a
    parameter-efficient alternative.
    """

    def __init__(
        self,
        layer_ids: Iterable[LayerId],
        hidden_size: int,
        unembed: nn.Module,
        vocab_size: Optional[int] = None,
        bias: bool = True,
        init_identity: bool = True,
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

        self.hidden_size = hidden_size
        self.unembed = unembed

        # Freeze unembed
        for p in self.unembed.parameters():
            p.requires_grad = False

        # Create per-layer translators
        self._layer_ids = [canonical_layer_id(lid) for lid in layer_ids]
        self._module_keys = {
            lid: module_safe_layer_key(lid) for lid in self._layer_ids
        }

        self.translators = nn.ModuleDict({
            self._module_keys[lid]: nn.Linear(hidden_size, hidden_size, bias=bias)
            for lid in self._layer_ids
        })

        if init_identity:
            self._init_identity()

    def _init_identity(self) -> None:
        """Initialize translators to zero (identity mapping)."""
        for translator in self.translators.values():
            nn.init.zeros_(translator.weight)
            if translator.bias is not None:
                nn.init.zeros_(translator.bias)

    @property
    def layer_ids(self) -> List[str]:
        """List of layer IDs."""
        return list(self._layer_ids)

    def compute_logits(
        self,
        activations: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> torch.Tensor:
        """
        Apply translator and unembed.

        Parameters
        ----------
        activations : torch.Tensor
            Hidden states of shape [batch, seq, hidden].
        layer : LayerId
            Layer identifier (required).
        """
        batch, seq, hidden = activations.shape
        flat = self._apply_layer(activations.view(batch * seq, hidden), layer)
        return self.unembed(flat).view(batch, seq, -1)

    def _apply_layer(self, flat: torch.Tensor, layer: Optional[LayerId]) -> torch.Tensor:
        """Apply the layer's translator with a residual connection to flat [N, d]."""
        translator = self.translators[self._resolve_module_key(layer)]
        return flat + translator(flat)

    def compute_logits_subset(
        self,
        activations: torch.Tensor,
        vocab_indices: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> torch.Tensor:
        """Compute logits for a vocabulary subset (see ``subset_logits_from_flat``)."""
        batch, seq, hidden = activations.shape
        flat = self._apply_layer(activations.view(batch * seq, hidden), layer)
        return subset_logits_from_flat(self.unembed, flat, vocab_indices, batch, seq)

    def compute_logits_subset_with_logsumexp(
        self,
        activations: torch.Tensor,
        vocab_indices: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute subset logits plus the full-vocab logsumexp (MC subset KL path)."""
        batch, seq, hidden = activations.shape
        flat = self._apply_layer(activations.view(batch * seq, hidden), layer)
        return subset_logits_and_logsumexp_from_flat(
            self.unembed, flat, vocab_indices, batch, seq
        )
