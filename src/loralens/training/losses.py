# src/loralens/training/losses.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_kl_logtarget(
    student_logits: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Exact KL(teacher || student), where teacher_logprobs are log p_teacher (no grad)
    and student_logits require grad.

    Memory policy:
      - Keep all large [B,T,V] tensors in bf16/fp16 (autocast dtype).
      - Only allow fp32 for small reductions (scalar sums), using sum(dtype=torch.float32)
        to avoid creating large fp32 intermediates.
    """
    # [B,T,V] in autocast dtype (bf16/fp16)
    s_logprobs = F.log_softmax(student_logits, dim=-1)
    t_logprobs = teacher_logprobs

    # [B,T,V] in autocast dtype
    kl_token = (t_logprobs.exp() * (t_logprobs - s_logprobs)).sum(dim=-1)

    # mask and reduce; keep mask in same dtype to avoid fp32 tensors of size [B,T]
    mask = attention_mask.to(dtype=kl_token.dtype)
    kl_token = kl_token * mask  # [B,T] bf16/fp16

    if reduction == "none":
        return kl_token

    if reduction == "sum":
        # scalar fp32 reduction (no large fp32 allocations)
        return kl_token.sum(dtype=torch.float32)

    if reduction == "mean":
        num = kl_token.sum(dtype=torch.float32)
        denom = mask.sum(dtype=torch.float32).clamp_min(1.0)
        return num / denom

    raise ValueError(f"Unknown reduction: {reduction}")