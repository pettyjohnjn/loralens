<<<<<<< HEAD
# src/loralens/training/amp.py
"""Automatic mixed precision utilities."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal, Optional

import torch
import torch.nn as nn


class AMPContext:
    """
    Unified context manager for automatic mixed precision.

    Handles both autocast and gradient scaling for mixed precision training.

    Parameters
    ----------
    enabled : bool
        Whether AMP is enabled.
    dtype : str
        Data type: "bf16" or "fp16".
    device : torch.device
        Device for training.
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: Literal["bf16", "fp16"] = "bf16",
        device: Optional[torch.device] = None,
    ) -> None:
        self.enabled = enabled
        self.dtype_str = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine torch dtype
        if dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            self.torch_dtype = torch.float16
        else:
            raise ValueError(f"Unknown dtype: {dtype}")

        # Only use scaler for fp16 (bf16 doesn't need it)
        use_scaler = enabled and dtype == "fp16" and self.device.type == "cuda"
        try:
            # PyTorch >= 2.1
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        except (AttributeError, TypeError):
            # PyTorch < 2.1
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def autocast(self):
        """Return autocast context manager."""
        if not self.enabled or self.device.type != "cuda":
            return nullcontext()
        try:
            # PyTorch >= 2.1
            return torch.amp.autocast("cuda", dtype=self.torch_dtype)
        except (AttributeError, TypeError):
            # PyTorch < 2.1
            return torch.cuda.amp.autocast(dtype=self.torch_dtype)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient computation."""
        return self.scaler.scale(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before clipping."""
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimizer step with scaler."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def get_lens_dtype(self) -> torch.dtype:
        """Get dtype for lens parameters."""
        if not self.enabled:
            return torch.float32
        return self.torch_dtype

    def __repr__(self) -> str:
        return f"AMPContext(enabled={self.enabled}, dtype={self.dtype_str})"
=======
from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch


def autocast_ctx(device: torch.device, amp: bool, amp_dtype: Literal["bf16", "fp16"]):
    if (not amp) or device.type != "cuda":
        return nullcontext()
    if amp_dtype == "bf16":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if amp_dtype == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    raise ValueError(f"Unknown amp_dtype={amp_dtype!r}")


def make_scaler(amp: bool, amp_dtype: Literal["bf16", "fp16"]) -> torch.cuda.amp.GradScaler:
    # bf16: no scaler
    if (not amp) or (amp_dtype != "fp16"):
        return torch.cuda.amp.GradScaler(enabled=False)
    return torch.cuda.amp.GradScaler(enabled=True)
>>>>>>> origin/main
