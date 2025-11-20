from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch


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