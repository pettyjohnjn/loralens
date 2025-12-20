from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def masked_kl_logtarget(
    *,
    student_logits: torch.Tensor,      # [B,T,V]
    teacher_logprobs: torch.Tensor,    # [B,T,V] (no grad)
    attention_mask: Optional[torch.Tensor],  # [B,T]
) -> torch.Tensor:
    """
    KL(p_teacher || p_student) averaged over non-masked tokens.

    Uses log_target=True to avoid materializing teacher probabilities tensor.
    """
    log_q = F.log_softmax(student_logits, dim=-1)
    kl_elem = F.kl_div(log_q, teacher_logprobs, log_target=True, reduction="none")  # [B,T,V]
    kl_per_tok = kl_elem.sum(dim=-1)  # [B,T]
    return masked_mean(kl_per_tok, attention_mask)