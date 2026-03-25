<<<<<<< HEAD
# src/loralens/lenses/types.py
"""Type definitions for lenses."""

=======
>>>>>>> origin/main
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch


<<<<<<< HEAD
# Layer identifier can be int or str
LayerId = Union[int, str]


def canonical_layer_id(layer: LayerId) -> str:
    """Convert layer ID to canonical string form."""
    return str(layer)


@dataclass
class LensOutput:
    """
    Output container for lens forward pass.

    Attributes
    ----------
    logits : Optional[torch.Tensor]
        Predicted logits of shape [batch, seq, vocab].
    loss : Optional[torch.Tensor]
        Loss value if computed during forward pass.
    extra : Dict[str, Any]
        Additional outputs (e.g., per-layer info, attention weights).
    """
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = []
        if self.logits is not None:
            parts.append(f"logits={tuple(self.logits.shape)}")
        if self.loss is not None:
            parts.append(f"loss={self.loss.item():.4f}")
        if self.extra:
            parts.append(f"extra_keys={list(self.extra.keys())}")
        return f"LensOutput({', '.join(parts)})"
=======
LayerId = Union[int, str]


@dataclass
class LensOutput:
    """
    Standard output container for all lens types.

    Attributes
    ----------
    logits:
        Predicted logits of shape [batch, seq, vocab].
    loss:
        Optional scalar or per-position loss (usually set by BaseLens when
        you pass labels, or by external objectives like subset KL).
    extra:
        Extra diagnostics or metadata (e.g. per-layer info, masks).
    """
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)
>>>>>>> origin/main
