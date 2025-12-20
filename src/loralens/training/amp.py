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