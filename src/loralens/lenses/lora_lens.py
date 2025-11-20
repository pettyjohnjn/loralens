from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .base import BaseLens
from .types import LayerId
from .tuned_lens import _canonical_layer_id


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation:

        y = x W^T + x (B A)^T * (alpha / r) + b

    - Base Linear (weight + optional bias) may be frozen.
    - Two low-rank matrices A: [in, r], B: [r, out] give the LoRA delta.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        r: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0

        self.base = nn.Linear(in_features, out_features, bias=bias)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
        else:
            self.lora_A = None
            self.lora_B = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard init for base
        nn.init.kaiming_uniform_(self.base.weight, a=5**0.5)
        if self.base.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base.weight)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.base.bias, -bound, bound)

        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            return base_out + lora_out
        else:
            return base_out


class LoRALens(BaseLens):
    """
    Lens where each per-layer projection is a LoRA-adapted linear map instead
    of a plain Linear, giving a low-rank tuned lens.

    Pipeline:
        h_L -> LoRALinear_L(h_L) -> readout -> logits
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
        r: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
        init_identity_base: bool = True,
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
                lid: LoRALinear(
                    hidden_size,
                    hidden_size,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    bias=bias,
                    freeze_base=freeze_base,
                )
                for lid in self._layer_ids
            }
        )

        if init_identity_base:
            self._init_identity_base()

    def _init_identity_base(self) -> None:
        """
        Initialize base part of LoRALinear projections close to identity.
        LoRA weights remain as initialized.
        """
        for proj in self.projections.values():
            nn.init.eye_(proj.base.weight)
            if proj.base.bias is not None:
                nn.init.zeros_(proj.base.bias)

    def compute_logits(
        self,
        activations: torch.Tensor,
        *,
        layer: Optional[LayerId] = None,
        **kwargs,
    ) -> torch.Tensor:
        if layer is None:
            raise ValueError("LoRALens requires a `layer` id to be provided.")

        lid = _canonical_layer_id(layer)
        if lid not in self.projections:
            raise KeyError(
                f"Layer id {layer!r} (canonical {lid!r}) is not registered in this LoRALens."
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