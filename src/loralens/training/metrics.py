from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from .ddp import DDPState, all_gather_vec, all_reduce_sum


def estimate_batch_bytes(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
    """
    Estimate UTF-8 bytes represented by the non-padding tokens in a batch.

    IMPORTANT: This should be called on CPU tensors to avoid GPU->CPU sync overhead.
    """
    bsz = input_ids.size(0)
    total = 0
    for i in range(bsz):
        valid = attention_mask[i].to(torch.bool)
        ids = input_ids[i][valid].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        total += len(text.encode("utf-8"))
    return total


@dataclass
class BitsPerByteState:
    total_tokens: float = 0.0
    total_bytes: float = 0.0

    def update(self, tokens: float, nbytes: float) -> None:
        self.total_tokens += float(tokens)
        self.total_bytes += float(nbytes)

    def nats_to_bpb_factor(self) -> Optional[float]:
        if self.total_bytes <= 0:
            return None
        return (self.total_tokens / self.total_bytes) / math.log(2.0)


def update_bpb_state(
    *,
    bpb: BitsPerByteState,
    tokenizer,
    input_ids_cpu: torch.Tensor,
    attention_mask_cpu: torch.Tensor,
    ddp_state: DDPState,
) -> None:
    """
    Update running totals for bits/byte conversion.

    This function expects CPU tensors for input_ids and attention_mask to avoid
    implicit GPU synchronization (e.g., .tolist()).
    """
    with torch.no_grad():
        local_tokens = float(attention_mask_cpu.sum().item())
        local_bytes = float(max(estimate_batch_bytes(tokenizer, input_ids_cpu, attention_mask_cpu), 1))

    bt = torch.tensor([local_tokens, local_bytes], device=ddp_state.device, dtype=torch.float32)
    bt = all_reduce_sum(bt, ddp_state)
    bpb.update(tokens=float(bt[0].item()), nbytes=float(bt[1].item()))


def gather_peak_mem_lines(ddp_state: DDPState) -> str:
    if ddp_state.device.type != "cuda":
        return ""

    peak_alloc = float(torch.cuda.max_memory_allocated(ddp_state.device))
    peak_reserved = float(torch.cuda.max_memory_reserved(ddp_state.device))
    mem_vec = torch.tensor([peak_alloc, peak_reserved], device=ddp_state.device, dtype=torch.float64)
    gathered = all_gather_vec(mem_vec, ddp_state)

    if not ddp_state.is_main:
        return ""

    parts: List[str] = []
    for r, m in enumerate(gathered):
        a = float(m[0].item()) / (1024**2)
        rs = float(m[1].item()) / (1024**2)
        parts.append(f"gpu{r}:alloc={a:.0f}MiB,resv={rs:.0f}MiB")
    return " | " + " ".join(parts)