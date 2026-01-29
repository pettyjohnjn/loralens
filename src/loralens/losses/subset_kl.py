# src/loralens/losses/subset_kl.py
"""
Subset KL divergence loss for memory-efficient lens training.

Two modes are supported:

**Simple Top-k (default, recommended):**
1. Extract top-k tokens from teacher distribution
2. Renormalize both distributions over just those k tokens  
3. Compute KL between the two k-way distributions

This is simpler and more stable than importance sampling.
For most use cases with k=256-512, this captures >99% of probability mass.

**Hajek Estimator (advanced):**
1. Deterministic head: top-k_head tokens
2. Stochastic tail: PPS sample k_tail additional tokens
3. Use importance-weighted Hajek estimator for unbiased KL estimate

The Hajek mode provides theoretically unbiased estimation but has:
- Higher variance (stochastic)
- More hyperparameters to tune
- Risk of instability with heavy-tailed distributions

Memory savings: O(B*T*k) instead of O(B*T*V) where k << V
"""

from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from .base import BaseLoss, ReductionType

if TYPE_CHECKING:
    from loralens.lenses import BaseLens


SubsetMode = Literal["topk", "hajek"]


class SubsetKLLoss(BaseLoss):
    """
    Memory-efficient subset KL loss.
    
    Computes KL divergence using only a subset of vocabulary tokens.
    
    Parameters
    ----------
    k : int
        Number of top tokens for simple mode, or k_head for Hajek mode.
    mode : str
        "topk" (default): Simple top-k with renormalization.
        "hajek": Head-tail with importance-weighted estimation.
    reduction : str
        How to reduce: "mean" or "sum".
    k_tail : int
        (Hajek mode only) Number of tail samples. Default 0 = no tail.
    tail_clip : float
        (Hajek mode only) Maximum importance weight.
    tail_oversample : int
        (Hajek mode only) Oversample factor for PPS sampling.
        
    Examples
    --------
    Simple top-k (recommended for most cases):
    >>> loss_fn = SubsetKLLoss(k=256)  # Top-256 tokens
    
    Hajek estimator (for research/large vocab):
    >>> loss_fn = SubsetKLLoss(k=128, mode="hajek", k_tail=64, tail_clip=50.0)
    """

    def __init__(
        self,
        k: int = 128,
        mode: SubsetMode = "topk",
        reduction: ReductionType = "mean",
        # Hajek mode parameters
        k_tail: int = 0,
        tail_clip: float = 50.0,
        tail_oversample: int = 4,
        # Legacy parameters for backwards compatibility
        k_head: Optional[int] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        # Support old k_head parameter name
        self.k = k_head if k_head is not None else k
        self.mode = mode
        self.k_tail = k_tail
        self.tail_clip = tail_clip
        self.tail_oversample = tail_oversample
        
        # Track variance for Hajek mode monitoring
        self._last_variance_proxy: Optional[float] = None

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute subset KL given pre-computed student logits.
        
        Note: This path still requires full student logits.
        For full memory efficiency, use forward_with_lens().
        """
        if self.mode == "hajek" and self.k_tail > 0:
            return self._forward_hajek(student_logits, teacher_logits, attention_mask)
        return self._forward_topk(student_logits, teacher_logits, attention_mask)
    
    def _forward_topk(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simple top-k subset KL with renormalization."""
        # Get top-k from teacher (NO float conversion needed!)
        with torch.no_grad():
            top_vals, top_idx = teacher_logits.topk(k=self.k, dim=-1)
            # Renormalize teacher over top-k
            teacher_logprobs_k = F.log_softmax(top_vals, dim=-1)
        
        # Gather student logits for top-k indices
        student_logits_k = torch.gather(student_logits, -1, top_idx)
        # Renormalize student over top-k  
        student_logprobs_k = F.log_softmax(student_logits_k, dim=-1)
        
        # KL(teacher || student) over k tokens
        # = sum_i p_i * (log p_i - log q_i)
        teacher_probs_k = teacher_logprobs_k.exp()
        kl = (teacher_probs_k * (teacher_logprobs_k - student_logprobs_k)).sum(dim=-1)
        
        return self._apply_reduction(kl, attention_mask)
    
    def _forward_hajek(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Head-tail KL with Hajek importance-weighted estimation.
        
        NOTE: This path still needs full-vocab softmax for proper importance
        sampling. Use _forward_topk for maximum memory efficiency.
        """
        from .sampling import pps_sample_indices_batched, hajek_kl_estimate
        
        B, T, V = student_logits.shape
        
        # Flatten for sampling
        teacher_flat = teacher_logits.view(B * T, V)
        student_flat = student_logits.view(B * T, V)
        
        with torch.no_grad():
            # This path does need full softmax for proper importance weights
            # but we keep it in autocast dtype, not fp32
            teacher_log_probs = F.log_softmax(teacher_flat, dim=-1)
            
            # PPS sampling: deterministic head + stochastic tail
            indices, inc_probs, mask, diagnostics = pps_sample_indices_batched(
                teacher_log_probs,
                k_head=self.k,
                k_tail=self.k_tail,
                oversample=self.tail_oversample,
            )
        
        # Gather values for selected indices
        teacher_sel = torch.gather(teacher_log_probs, -1, indices)
        student_sel = torch.gather(student_flat, -1, indices)
        
        # Hajek estimator
        kl_flat, variance_proxy = hajek_kl_estimate(
            teacher_sel, student_sel, indices, inc_probs, mask, self.tail_clip
        )
        self._last_variance_proxy = variance_proxy
        
        kl = kl_flat.view(B, T)
        return self._apply_reduction(kl, attention_mask)

    def forward_with_lens(
        self,
        hidden_states: torch.Tensor,
        teacher_logits: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Maximum memory efficiency: compute only k student logits.
        
        This path NEVER materializes full [B, T, V] student logits.
        
        Parameters
        ----------
        hidden_states : torch.Tensor
            Activations [batch, seq, hidden].
        teacher_logits : torch.Tensor
            Teacher logits [batch, seq, vocab].
        lens : BaseLens
            Lens module.
        layer : LayerId
            Layer for the lens.
        attention_mask : Optional[torch.Tensor]
            Mask [batch, seq].
        """
        if self.mode == "hajek" and self.k_tail > 0:
            return self._forward_with_lens_hajek(
                hidden_states, teacher_logits, lens, layer, attention_mask
            )
        return self._forward_with_lens_topk(
            hidden_states, teacher_logits, lens, layer, attention_mask
        )
    
    def _forward_with_lens_topk(
        self,
        hidden_states: torch.Tensor,
        teacher_logits: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simple top-k with memory-efficient lens computation.
        
        MEMORY OPTIMIZATIONS:
        1. NO full-vocab softmax - only softmax over k tokens
        2. NO dtype conversion - topk works on bf16
        3. Student logits computed only for k indices
        """
        # Get top-k indices directly (NO float conversion - topk works on bf16!)
        with torch.no_grad():
            top_vals, top_idx = teacher_logits.topk(k=self.k, dim=-1)  # [B, T, k]
            # Renormalize teacher over top-k only (small tensor!)
            teacher_logprobs_k = F.log_softmax(top_vals, dim=-1)
            teacher_probs_k = teacher_logprobs_k.exp()
        
        # Compute student logits ONLY for top-k indices
        # This uses lens.compute_logits_subset() - never materializes [B,T,V]
        student_logits_k = lens.forward(
            hidden_states, 
            layer=layer, 
            vocab_indices=top_idx
        ).logits
        
        # Renormalize student over top-k (small tensor!)
        student_logprobs_k = F.log_softmax(student_logits_k, dim=-1)
        
        # KL over k tokens only
        kl = (teacher_probs_k * (teacher_logprobs_k - student_logprobs_k)).sum(dim=-1)
        
        return self._apply_reduction(kl, attention_mask)
    
    def _forward_with_lens_hajek(
        self,
        hidden_states: torch.Tensor,
        teacher_logits: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hajek estimator with memory-efficient lens computation.
        
        NOTE: This still requires full-vocab teacher softmax for proper
        importance weights. For maximum memory efficiency, use topk mode.
        """
        from .sampling import pps_sample_indices_batched, hajek_kl_estimate
        
        B, T, V = teacher_logits.shape
        
        # Flatten teacher for sampling
        teacher_flat = teacher_logits.view(B * T, V)
        
        with torch.no_grad():
            # Keep in autocast dtype
            teacher_log_probs = F.log_softmax(teacher_flat, dim=-1)
            
            # PPS sampling
            indices, inc_probs, mask, diagnostics = pps_sample_indices_batched(
                teacher_log_probs,
                k_head=self.k,
                k_tail=self.k_tail,
                oversample=self.tail_oversample,
            )
            
            # Gather teacher values for selected indices
            teacher_sel = torch.gather(teacher_log_probs, -1, indices)
            
            # Reshape indices for lens (need [B, T, S])
            S = indices.shape[-1]
            indices_3d = indices.view(B, T, S)
        
        # Compute student logits ONLY for selected indices
        student_logits_subset = lens.forward(
            hidden_states,
            layer=layer,
            vocab_indices=indices_3d
        ).logits  # [B, T, S]
        
        # Flatten student for Hajek computation
        student_sel = student_logits_subset.view(B * T, S)
        
        # Hajek estimator
        kl_flat, variance_proxy = hajek_kl_estimate(
            teacher_sel, student_sel, indices, inc_probs, mask, self.tail_clip
        )
        self._last_variance_proxy = variance_proxy
        
        kl = kl_flat.view(B, T)
        return self._apply_reduction(kl, attention_mask)
    
    @property
    def variance_proxy(self) -> Optional[float]:
        """Variance proxy from last Hajek computation (for monitoring)."""
        return self._last_variance_proxy

    def __repr__(self) -> str:
        if self.mode == "hajek" and self.k_tail > 0:
            return (
                f"SubsetKLLoss(k={self.k}, mode='hajek', "
                f"k_tail={self.k_tail}, tail_clip={self.tail_clip}, "
                f"reduction={self.reduction!r})"
            )
        return f"SubsetKLLoss(k={self.k}, reduction={self.reduction!r})"
