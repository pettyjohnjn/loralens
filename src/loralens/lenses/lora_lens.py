# src/loralens/lenses/lora_lens.py
"""
LoRA Lens - Parameter-efficient lens using low-rank adaptation.

Replaces full-rank per-layer translators with low-rank decompositions,
dramatically reducing parameter count while maintaining expressivity.

CRITICAL DIFFERENCE from TunedLens:
- TunedLens: stores d×d translator per layer = O(L * d²) parameters
- LoRALens: stores d×r + r×d per layer = O(L * 2dr) parameters

For d=4096, r=16: 16M vs 131K per layer = 128x reduction
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
    unembed_weight_dtype,
)


class LoRAProjection(nn.Module):
    """
    Low-rank projection: h' = h + B @ A @ h * (alpha / r)

    NO base linear stored - just the low-rank matrices.
    This is the key memory savings vs TunedLens.

    Parameters
    ----------
    hidden_size : int
        Dimension of hidden states.
    r : int
        Rank of the low-rank matrices.
    alpha : float
        Scaling factor.
    dropout : float
        Dropout rate applied before projection.
    """

    def __init__(
        self,
        hidden_size: int,
        r: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0

        # Only low-rank matrices - NO d×d base!
        self.lora_A = nn.Linear(hidden_size, r, bias=False)
        self.lora_B = nn.Linear(r, hidden_size, bias=False)

        # Optional bias (small: just d parameters)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize to approximate identity (LoRA delta starts at zero).

        Following the original LoRA paper and tuned-lens implementation:
        - down (A) initialized with xavier_uniform for good gradient flow
        - up (B) initialized to zero so residual starts as pure identity
        """
        nn.init.xavier_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)  # B=0 means LoRA contribution starts at 0
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply low-rank projection with residual.

        h' = h + B @ A @ dropout(h) * scaling + bias
        """
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return x + lora_out + self.bias

    def __repr__(self) -> str:
        return f"LoRAProjection(hidden={self.hidden_size}, r={self.r}, alpha={self.alpha})"


class LoRALens(BaseLens):
    """
    LoRA Lens: parameter-efficient per-layer projections.

    Like TunedLens, but uses low-rank adaptation instead of full
    linear layers, reducing parameters from O(d²) to O(d*r).

    Parameters
    ----------
    layer_ids : Iterable[LayerId]
        Layer identifiers.
    hidden_size : int
        Dimension of hidden states.
    unembed : nn.Module
        Unembedding module.
    vocab_size : Optional[int]
        Vocabulary size.
    r : int
        LoRA rank.
    alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout rate.

    Notes
    -----
    Memory comparison for GPT-2 (d=768, L=12, r=16):
    - TunedLens: 12 * 768² = 7.1M trainable params
    - LoRALens: 12 * 2 * 768 * 16 = 295K trainable params (24x reduction)

    For LLaMA-70B (d=8192, L=80, r=16):
    - TunedLens: 80 * 8192² = 5.4B trainable params (!)
    - LoRALens: 80 * 2 * 8192 * 16 = 21M trainable params (256x reduction)
    """

    def __init__(
        self,
        layer_ids: Iterable[LayerId],
        hidden_size: int,
        unembed: nn.Module,
        vocab_size: Optional[int] = None,
        r: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        # Infer vocab size
        if vocab_size is None:
            if hasattr(unembed, "vocab_size"):
                vocab_size = unembed.vocab_size
            elif hasattr(unembed, "out_features"):
                vocab_size = unembed.out_features
            elif isinstance(unembed, nn.Linear):
                vocab_size = unembed.out_features
            else:
                raise ValueError("Cannot infer vocab_size; provide it explicitly.")

        super().__init__(vocab_size=vocab_size)

        self.hidden_size = hidden_size
        self.r = r
        self.alpha = alpha
        self.unembed = unembed

        # Freeze unembed
        for p in self.unembed.parameters():
            p.requires_grad = False

        # Create per-layer LoRA projections (NO d×d matrices!)
        self._layer_ids = [canonical_layer_id(lid) for lid in layer_ids]
        self._module_keys = {
            lid: module_safe_layer_key(lid) for lid in self._layer_ids
        }

        self.projections = nn.ModuleDict({
            self._module_keys[lid]: LoRAProjection(
                hidden_size=hidden_size,
                r=r,
                alpha=alpha,
                dropout=dropout,
            )
            for lid in self._layer_ids
        })

    @property
    def layer_ids(self) -> List[str]:
        """List of layer IDs."""
        return list(self._layer_ids)

    def _apply_layer(self, flat: torch.Tensor, layer: Optional[LayerId]) -> torch.Tensor:
        """Apply the layer's LoRA projection (residual + bias) to flat [N, d]."""
        return self.projections[self._resolve_module_key(layer)](flat)

    def compute_logits(
        self,
        activations: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> torch.Tensor:
        """Apply LoRA projection and unembed."""
        batch, seq, hidden = activations.shape
        flat = self._apply_layer(activations.view(batch * seq, hidden), layer)

        unembed_dtype = unembed_weight_dtype(self.unembed)
        if unembed_dtype is not None and flat.dtype != unembed_dtype:
            flat = flat.to(unembed_dtype)

        return self.unembed(flat).view(batch, seq, -1)

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

    def trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Count total parameters (including frozen unembed)."""
        return sum(p.numel() for p in self.parameters())

    def parameter_savings_ratio(self) -> float:
        """Ratio of full-rank to LoRA parameters."""
        full_rank = self.hidden_size ** 2 * len(self._layer_ids)
        lora_params = (2 * self.hidden_size * self.r + self.hidden_size) * len(self._layer_ids)
        return full_rank / max(lora_params, 1)

    def __repr__(self) -> str:
        trainable = self.trainable_parameters()
        total = self.total_parameters()
        return (
            f"LoRALens(vocab_size={self.vocab_size}, r={self.r}, "
            f"layers={len(self._layer_ids)}, "
            f"params={trainable:,}/{total:,})"
        )
