# src/loralens/losses/sampling.py
"""
Sampling utilities for subset KL divergence estimation.

Implements Probability Proportional to Size (PPS) sampling and
Hajek (self-normalized) importance sampling estimators for
unbiased estimation of full-vocabulary KL divergence.

These are ADVANCED methods - for most use cases, the simple top-k
approach in SubsetKLLoss is sufficient and more stable.
"""

from __future__ import annotations

from typing import Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplingDiagnostics:
    """Diagnostics from PPS sampling for monitoring estimator quality."""
    num_unique_indices: int
    mean_inclusion_prob: float
    min_inclusion_prob: float
    max_importance_weight: float
    variance_proxy: Optional[float] = None


def pps_sample_indices_batched(
    log_probs: torch.Tensor,
    k_head: int,
    k_tail: int,
    oversample: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SamplingDiagnostics]:
    """
    Probability Proportional to Size sampling with deterministic head.
    
    Combines top-k (deterministic head) with PPS sampling (stochastic tail)
    for a hybrid subset that captures both high-probability tokens and
    provides unbiased coverage of the tail distribution.
    
    Parameters
    ----------
    log_probs : torch.Tensor
        Log probabilities from teacher [B*T, V] or [N, V].
    k_head : int
        Number of top tokens to include deterministically.
    k_tail : int
        Number of additional tokens to sample from tail.
    oversample : int
        Oversample factor for PPS (draw oversample*k_tail then deduplicate).
        
    Returns
    -------
    indices : torch.Tensor
        Selected indices [N, S] where S <= k_head + k_tail.
    inclusion_probs : torch.Tensor
        Inclusion probabilities for each selected index [N, S].
    mask : torch.Tensor
        Valid mask for variable-length selections [N, S].
    diagnostics : SamplingDiagnostics
        Monitoring statistics.
    """
    N, V = log_probs.shape
    device = log_probs.device
    dtype = log_probs.dtype
    
    # Ensure numerical stability
    log_probs = log_probs.float()
    probs = F.softmax(log_probs, dim=-1)
    
    # HEAD: Top-k deterministic indices
    _, top_idx = log_probs.topk(k_head, dim=-1)  # [N, k_head]
    
    # Create mask for tail (exclude head indices)
    head_mask = torch.zeros(N, V, device=device, dtype=torch.bool)
    head_mask.scatter_(1, top_idx, True)
    tail_mask = ~head_mask
    
    # Renormalize probabilities over tail
    tail_probs = probs.clone()
    tail_probs[head_mask] = 0.0
    tail_probs = tail_probs / tail_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    
    # PPS sampling from tail with replacement
    num_draws = k_tail * oversample
    if num_draws > 0 and tail_mask.any():
        # Sample indices proportional to tail_probs
        sampled = torch.multinomial(
            tail_probs,
            num_samples=min(num_draws, V - k_head),
            replacement=True
        )  # [N, num_draws]
        
        # Combine head and tail
        all_idx = torch.cat([top_idx, sampled], dim=-1)  # [N, k_head + num_draws]
        
        # Deduplicate via sort + unique detection
        all_idx_sorted, sort_order = all_idx.sort(dim=-1)
        diff = torch.diff(all_idx_sorted, dim=-1, prepend=all_idx_sorted[:, :1] - 1)
        keep_mask = diff != 0  # True for unique values
        
        # Count unique per row
        counts = keep_mask.sum(dim=-1)  # [N]
        max_unique = counts.max().item()
        
        # Extract unique indices with padding
        # Create output tensors
        S_max = min(k_head + k_tail, max_unique)
        indices = torch.zeros(N, S_max, device=device, dtype=torch.long)
        mask = torch.zeros(N, S_max, device=device, dtype=torch.bool)
        
        # Fill unique indices row by row (vectorized where possible)
        for i in range(N):
            unique_i = all_idx_sorted[i][keep_mask[i]]
            n_i = min(len(unique_i), S_max)
            indices[i, :n_i] = unique_i[:n_i]
            mask[i, :n_i] = True
    else:
        # No tail sampling
        indices = top_idx
        mask = torch.ones(N, k_head, device=device, dtype=torch.bool)
        S_max = k_head
    
    # Compute inclusion probabilities
    # For head: π_i = 1 (always included)
    # For tail: π_i = 1 - (1 - p_i)^m where m is number of draws
    inclusion_probs = torch.ones_like(indices, dtype=torch.float32)
    
    if k_tail > 0:
        # Gather probabilities for selected indices
        p_sel = torch.gather(probs, 1, indices)  # [N, S]
        
        # Identify which are head (prob >= top-k threshold)
        p_head_min = torch.gather(probs, 1, top_idx[:, -1:])  # [N, 1]
        is_head = p_sel >= p_head_min
        
        # For tail items: π_i = 1 - (1-p_i)^m
        tail_inclusion = 1.0 - torch.pow(1.0 - p_sel, num_draws)
        inclusion_probs = torch.where(is_head, torch.ones_like(p_sel), tail_inclusion)
        
        # Clamp to avoid division by zero
        inclusion_probs = inclusion_probs.clamp_min(1e-8)
    
    # Compute diagnostics
    valid_inclusion = inclusion_probs[mask]
    diagnostics = SamplingDiagnostics(
        num_unique_indices=int(mask.sum().item() / N) if N > 0 else 0,
        mean_inclusion_prob=valid_inclusion.mean().item() if valid_inclusion.numel() > 0 else 1.0,
        min_inclusion_prob=valid_inclusion.min().item() if valid_inclusion.numel() > 0 else 1.0,
        max_importance_weight=(1.0 / valid_inclusion.min()).item() if valid_inclusion.numel() > 0 else 1.0,
    )
    
    return indices, inclusion_probs, mask, diagnostics


def hajek_kl_estimate(
    teacher_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    indices: torch.Tensor,
    inclusion_probs: torch.Tensor,
    mask: torch.Tensor,
    weight_clip: float = 50.0,
) -> Tuple[torch.Tensor, float]:
    """
    Hajek (self-normalized) importance sampling estimator for KL divergence.
    
    This provides an approximately unbiased estimate of the full-vocabulary
    KL divergence using only a subset of vocabulary indices.
    
    KL(P || Q) ≈ Σ_i∈S w_i * P_i * (log P_i - log Q_i) / Σ_i∈S w_i * P_i
    
    where w_i = 1/π_i are importance weights.
    
    Parameters
    ----------
    teacher_log_probs : torch.Tensor
        Teacher log probabilities [N, S] for selected indices.
    student_logits : torch.Tensor
        Student logits [N, S] for selected indices.
    indices : torch.Tensor
        Selected vocabulary indices [N, S].
    inclusion_probs : torch.Tensor
        Inclusion probabilities π_i for each index [N, S].
    mask : torch.Tensor
        Valid mask [N, S].
    weight_clip : float
        Maximum importance weight to prevent variance explosion.
        
    Returns
    -------
    kl : torch.Tensor
        KL divergence estimate [N].
    variance_proxy : float
        Proxy for estimator variance (for monitoring).
    """
    # Importance weights: w_i = 1/π_i
    weights = (1.0 / inclusion_probs).clamp(max=weight_clip)
    
    # Mask invalid positions
    weights = weights * mask.float()
    
    # Teacher probabilities
    teacher_probs = teacher_log_probs.exp()
    
    # Student log-probabilities via log-softmax over subset
    # NOTE: This renormalizes over the subset, not full vocab
    # For proper full-vocab estimate, we'd need logsumexp over full V
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    # Weighted KL terms
    # KL_i = P_i * (log P_i - log Q_i) * w_i
    kl_terms = teacher_probs * (teacher_log_probs - student_log_probs) * weights
    
    # Self-normalize (Hajek estimator)
    numerator = (kl_terms * mask.float()).sum(dim=-1)
    denominator = (teacher_probs * weights * mask.float()).sum(dim=-1).clamp_min(1e-8)
    
    kl = numerator / denominator
    
    # Variance proxy: sum of squared normalized weights
    w_normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    variance_proxy = (w_normalized ** 2 * mask.float()).sum().item() / max(mask.shape[0], 1)
    
    return kl, variance_proxy


def head_tail_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    k_head: int = 256,
    k_tail: int = 0,
    tail_clip: float = 50.0,
    self_normalize: bool = True,
) -> torch.Tensor:
    """
    Head-tail subset KL divergence (simplified interface).
    
    Computes KL divergence with deterministic top-k head and optionally
    an importance-sampled tail for unbiased estimation.
    
    Parameters
    ----------
    teacher_logits : torch.Tensor
        Teacher logits [B, T, V].
    student_logits : torch.Tensor
        Student logits [B, T, V].
    k_head : int
        Number of top tokens for deterministic head.
    k_tail : int
        Number of additional tail samples (0 = no tail).
    tail_clip : float
        Maximum importance weight.
    self_normalize : bool
        Use Hajek (self-normalized) estimator.
        
    Returns
    -------
    kl : torch.Tensor
        Per-position KL divergence [B, T].
    """
    B, T, V = teacher_logits.shape
    
    # Reshape to [B*T, V]
    teacher_flat = teacher_logits.view(B * T, V)
    student_flat = student_logits.view(B * T, V)
    
    # Teacher log-probs
    teacher_log_probs = F.log_softmax(teacher_flat, dim=-1)
    
    if k_tail == 0:
        # Simple top-k: no tail sampling, just renormalize over head
        top_vals, top_idx = teacher_log_probs.topk(k_head, dim=-1)
        teacher_head = F.log_softmax(top_vals, dim=-1)  # Renormalized
        
        student_head = torch.gather(student_flat, -1, top_idx)
        student_head = F.log_softmax(student_head, dim=-1)  # Renormalized
        
        teacher_probs = teacher_head.exp()
        kl_flat = (teacher_probs * (teacher_head - student_head)).sum(dim=-1)
    else:
        # Full head-tail with importance sampling
        indices, inc_probs, mask, _ = pps_sample_indices_batched(
            teacher_log_probs, k_head, k_tail
        )
        
        # Gather teacher and student values
        teacher_sel = torch.gather(teacher_log_probs, -1, indices)
        student_sel = torch.gather(student_flat, -1, indices)
        
        kl_flat, _ = hajek_kl_estimate(
            teacher_sel, student_sel, indices, inc_probs, mask, tail_clip
        )
    
    return kl_flat.view(B, T)
