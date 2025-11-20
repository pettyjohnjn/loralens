from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .base import BaseLens
from .types import LayerId


def _canonical_layer_id(layer: LayerId) -> str:
    return str(layer)


class TunedLens(BaseLens):
    """
    Tuned lens: one affine map per layer, followed by the model's readout head.

    For layer ids L, we maintain:
        projection[L]: Linear(hidden -> hidden)

    Pipeline:
        h_L -> projection_L(h_L) -> readout -> logits
    """

    def __init__(
        self,
        layer_ids: Iterable[LayerId],
        hidden_size: int,
        readout: nn.Module,
        *,
        vocab_size: Optional[int] = None,
        ignore_index: int = -100,
        loss_reduction: str = "mean",
        bias: bool = True,
        init_identity: bool = True,
    ) -> None:
        if vocab_size is None:
            if isinstance(readout, nn.Linear):
                vocab_size = readout.out_features
            else:
                raise ValueError(
                    "Could not infer vocab_size automatically from readout. "
                    "Pass `vocab_size` explicitly."
                )

        super().__init__(
            vocab_size=vocab_size,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
        )

        self.readout = readout
        self.hidden_size = hidden_size

        layer_ids = list(layer_ids)
        self._layer_ids: List[str] = [_canonical_layer_id(l) for l in layer_ids]

        self.projections = nn.ModuleDict(
            {
                lid: nn.Linear(hidden_size, hidden_size, bias=bias)
                for lid in self._layer_ids
            }
        )

        if init_identity:
            self._init_identity()

    def _init_identity(self) -> None:
        """
        Initialize projections close to identity: W ~ I, b ~ 0.
        """
        for proj in self.projections.values():
            nn.init.eye_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def compute_logits(
        self,
        activations: torch.Tensor,
        *,
        layer: Optional[LayerId] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Select the per-layer projection and apply the readout head.
        """
        if layer is None:
            raise ValueError("TunedLens requires a `layer` id to be provided.")

        lid = _canonical_layer_id(layer)
        if lid not in self.projections:
            raise KeyError(
                f"Layer id {layer!r} (canonical {lid!r}) is not registered in this TunedLens."
            )

        proj = self.projections[lid]

        batch, seq, hidden = activations.shape
        if hidden != self.hidden_size:
            raise ValueError(
                f"Expected activations last dim {self.hidden_size}, got {hidden}."
            )

        flat = activations.reshape(batch * seq, hidden)
        projected = proj(flat)
        flat_logits = self.readout(projected)
        logits = flat_logits.reshape(batch, seq, -1)
        return logits