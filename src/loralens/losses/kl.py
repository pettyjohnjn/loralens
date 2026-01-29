# src/loralens/losses/kl.py
"""KL divergence loss for lens training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss, ReductionType


class KLDivergenceLoss(BaseLoss):
    """
    KL divergence loss: KL(teacher || student).
    
    Computes the full-vocabulary KL divergence between the teacher
    and student distributions. This is the standard loss for lens
    training but requires materializing full logit tensors.
    
    Parameters
    ----------
    reduction : str
        How to reduce: "none", "mean", or "sum".
    temperature : float
        Temperature for softmax (1.0 = no scaling).
    chunk_size : Optional[int]
        If set, compute KL in chunks along sequence dimension
        to reduce peak memory usage.
        
    Notes
    -----
    Memory usage scales as O(batch * seq * vocab), which can be
    problematic for large vocabularies. Consider SubsetKLLoss for
    memory-constrained settings.
    """

    def __init__(
        self,
        reduction: ReductionType = "mean",
        temperature: float = 1.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.chunk_size = chunk_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL(teacher || student).
        
        Uses log-space computation for numerical stability.
        """
        if self.chunk_size is not None:
            return self._forward_chunked(
                student_logits, teacher_logits, attention_mask
            )
        
        return self._forward_full(
            student_logits, teacher_logits, attention_mask
        )

    def _forward_full(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Full KL computation (no chunking)."""
        # Apply temperature
        if self.temperature != 1.0:
            student_logits = student_logits / self.temperature
            teacher_logits = teacher_logits / self.temperature
        
        # Compute log probabilities
        student_logprobs = F.log_softmax(student_logits, dim=-1)
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
        
        # KL(p || q) = sum(p * (log p - log q))
        # Using teacher probs, not student probs
        teacher_probs = teacher_logprobs.exp()
        kl_per_token = (teacher_probs * (teacher_logprobs - student_logprobs)).sum(dim=-1)
        
        return self._apply_reduction(kl_per_token, attention_mask)

    def _forward_chunked(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Chunked KL computation for memory efficiency."""
        batch, seq, vocab = student_logits.shape
        chunk_size = self.chunk_size
        
        total_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        total_count = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        
        for t0 in range(0, seq, chunk_size):
            t1 = min(t0 + chunk_size, seq)
            
            s_chunk = student_logits[:, t0:t1, :]
            t_chunk = teacher_logits[:, t0:t1, :]
            
            if attention_mask is not None:
                m_chunk = attention_mask[:, t0:t1]
            else:
                m_chunk = None
            
            # Compute KL for this chunk
            if self.temperature != 1.0:
                s_chunk = s_chunk / self.temperature
                t_chunk = t_chunk / self.temperature
            
            s_logprobs = F.log_softmax(s_chunk, dim=-1)
            t_logprobs = F.log_softmax(t_chunk, dim=-1)
            t_probs = t_logprobs.exp()
            
            kl_chunk = (t_probs * (t_logprobs - s_logprobs)).sum(dim=-1)
            
            if m_chunk is not None:
                kl_chunk = kl_chunk * m_chunk.to(kl_chunk.dtype)
                total_count += m_chunk.sum()
            else:
                total_count += kl_chunk.numel()
            
            total_loss += kl_chunk.sum()
        
        if self.reduction == "sum":
            return total_loss
        elif self.reduction == "mean":
            return total_loss / total_count.clamp_min(1.0)
        else:
            raise ValueError("Chunked KL only supports 'mean' or 'sum' reduction")

    def __repr__(self) -> str:
        return (
            f"KLDivergenceLoss(reduction={self.reduction!r}, "
            f"temperature={self.temperature}, chunk_size={self.chunk_size})"
        )
