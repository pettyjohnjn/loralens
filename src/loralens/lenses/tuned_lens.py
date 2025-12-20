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
    Tuned lens (tuned-lens style):
      h_L -> (h_L + translator_L(h_L)) -> unembed -> logits

    translator_L is initialized to zero so the whole transform starts as identity.
    """

    def __init__(
        self,
        layer_ids: Iterable[LayerId],
        hidden_size: int,
        unembed: nn.Module,
        *,
        vocab_size: Optional[int] = None,
        ignore_index: int = -100,
        loss_reduction: str = "mean",
        bias: bool = True,
        init_zero: bool = True,
    ) -> None:
        if vocab_size is None:
            if isinstance(unembed, nn.Module) and hasattr(unembed, "vocab_size"):
                vocab_size = int(getattr(unembed, "vocab_size"))
            elif isinstance(unembed, nn.Linear):
                vocab_size = unembed.out_features
            else:
                raise ValueError(
                    "Could not infer vocab_size automatically. Pass vocab_size explicitly."
                )

        super().__init__(
            vocab_size=vocab_size,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
        )

        self.unembed = unembed
        self.hidden_size = hidden_size

        layer_ids = list(layer_ids)
        self._layer_ids: List[str] = [_canonical_layer_id(l) for l in layer_ids]

        # In tuned-lens repo: they do not include final layer translator.
        # We'll still register all ids you pass, but typical usage should pass all layers
        # and allow the training loop to decide which ids to include.
        self.translators = nn.ModuleDict(
            {lid: nn.Linear(hidden_size, hidden_size, bias=bias) for lid in self._layer_ids}
        )

        if init_zero:
            self._init_zero()

    def _init_zero(self) -> None:
        for tr in self.translators.values():
            tr.weight.data.zero_()
            if tr.bias is not None:
                tr.bias.data.zero_()

    def compute_logits(
        self,
        activations: torch.Tensor,
        *,
        layer: Optional[LayerId] = None,
        **kwargs,
    ) -> torch.Tensor:
        if layer is None:
            raise ValueError("TunedLens requires a `layer` id to be provided.")
        lid = _canonical_layer_id(layer)
        if lid not in self.translators:
            raise KeyError(f"Layer id {layer!r} (canonical {lid!r}) is not registered in this TunedLens.")

        batch, seq, hidden = activations.shape
        if hidden != self.hidden_size:
            raise ValueError(f"Expected activations last dim {self.hidden_size}, got {hidden}.")

        tr = self.translators[lid]
        flat = activations.reshape(batch * seq, hidden)
        flat = flat + tr(flat)  # residual translator
        flat_logits = self.unembed(flat)
        return flat_logits.reshape(batch, seq, -1)