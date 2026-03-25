# src/loralens/losses/shared_subset_kl.py
"""
Shared Subset KL Loss - Memory-efficient KL with shared candidate vocabulary.

Key insight: Instead of per-position top-k (which forces [N,V] or [N,k,d]),
we build a SHARED candidate set S per sequence+chunk, enabling standard GEMM:

    logits_S = H @ W[S].T   # [chunk_T, K] via cuBLAS

This avoids:
- Full [B, T, V] student logits
- Per-position weight gathering [B, T, k, d]

The candidate set S is built by taking union of per-position importance
samples from the teacher, pruned to K by aggregated probability mass.

Memory: O(B * chunk_T * K) where K << V (typically K < 1% of V)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, ReductionType

if TYPE_CHECKING:
    from loralens.lenses import BaseLens


@dataclass
class SharedSubsetConfig:
    """Configuration for shared subset construction."""
    # Per-position candidates
    top_m: int = 16  # Importance-sampled tokens per position
    sample_r: int = 0  # Additional importance-sampled tokens (0 = disabled)

    # Shared set size cap
    max_K: int = 512  # Maximum shared set size

    # Pruning strategy when union exceeds max_K
    prune_by: str = "mass"  # "mass" (sum of teacher probs) or "max" (max prob)


class SharedSubsetKLLoss(BaseLoss):
    """
    Memory-efficient KL loss using shared candidate vocabulary.

    For each sequence+chunk:
    1. Build candidate set S by union of per-position importance samples
    2. Compute student logits ONLY for S: H @ W[S].T → [chunk_T, K]
    3. Compute position-specific KL on each position's support within S

    Parameters
    ----------
    top_m : int
        Number of importance-sampled teacher tokens per position to include.
    max_K : int
        Maximum size of shared candidate set (caps memory).
    sample_r : int
        Additional importance-sampled tokens per position (0 = disabled).
    reduction : str
        How to reduce: "mean" or "sum".

    Example
    -------
    >>> loss_fn = SharedSubsetKLLoss(top_m=16, max_K=512)
    >>> # With chunk_T=128, this uses [128, 512] instead of [128, 50257]
    >>> # Memory reduction: ~100x for typical vocab sizes
    """

    def __init__(
        self,
        top_m: int = 16,
        max_K: int = 512,
        sample_r: int = 0,
        reduction: ReductionType = "mean",
        prune_by: str = "mass",
    ) -> None:
        super().__init__(reduction=reduction)
        self.config = SharedSubsetConfig(
            top_m=top_m,
            sample_r=sample_r,
            max_K=max_K,
            prune_by=prune_by,
        )

        # Diagnostics
        self._last_K: Optional[int] = None
        self._last_coverage: Optional[float] = None

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard forward - requires full student logits (not memory efficient).

        Use forward_with_lens() for the memory-efficient path.
        """
        # Fall back to standard top-k for this path
        with torch.no_grad():
            top_vals, top_idx = teacher_logits.topk(k=self.config.top_m, dim=-1)
            teacher_logprobs_k = F.log_softmax(top_vals, dim=-1)
            teacher_probs_k = teacher_logprobs_k.exp()

        student_logits_k = torch.gather(student_logits, -1, top_idx)
        student_logprobs_k = F.log_softmax(student_logits_k, dim=-1)

        kl = (teacher_probs_k * (teacher_logprobs_k - student_logprobs_k)).sum(dim=-1)
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
        Memory-efficient forward using shared candidate set.

        This is the key method - it NEVER materializes [B, T, V] student logits.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Activations [batch, seq, hidden] (should be a chunk, not full seq).
        teacher_logits : torch.Tensor
            Teacher logits [batch, seq, vocab].
        lens : BaseLens
            Lens module (used to get unembed weights).
        layer : LayerId
            Layer identifier.
        attention_mask : Optional[torch.Tensor]
            Mask [batch, seq].
        """
        B, T, V = teacher_logits.shape
        device = teacher_logits.device

        # Get unembed weight matrix from lens
        W, b = self._get_unembed_weights(lens)

        # Process each sequence independently (different candidate sets)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_count = torch.zeros((), device=device, dtype=torch.float32)

        for batch_idx in range(B):
            h_seq = hidden_states[batch_idx]  # [T, d]
            teacher_seq = teacher_logits[batch_idx]  # [T, V]
            mask_seq = attention_mask[batch_idx] if attention_mask is not None else None

            # Build shared candidate set for this sequence
            S, pos_cols, pos_probs = self._build_candidate_set(teacher_seq, device)
            K = S.numel()

            # Apply lens projection (LoRA transform)
            h_proj = self._apply_lens_projection(h_seq, lens, layer)  # [T, d]

            # Compute student logits ONLY for candidate set S
            # This is the key memory optimization: [T, K] instead of [T, V]
            W_S = W[S]  # [K, d]
            student_logits_S = h_proj @ W_S.T  # [T, K]
            if b is not None:
                student_logits_S = student_logits_S + b[S]

            # Compute position-specific KL
            seq_loss, seq_count = self._compute_position_kl_vectorized(
                student_logits_S=student_logits_S,
                pos_cols=pos_cols,
                pos_probs=pos_probs,
                mask_seq=mask_seq,
            )

            total_loss = total_loss + seq_loss
            total_count = total_count + seq_count

            # Store last K for diagnostics
            self._last_K = K

        if self.reduction == "mean":
            return total_loss / total_count.clamp_min(1.0)
        else:
            return total_loss

    def _get_unembed_weights(
        self, lens: "BaseLens"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract unembed weight matrix from lens."""
        # Unwrap DDP
        raw_lens = lens.module if hasattr(lens, 'module') else lens
        unembed = raw_lens.unembed

        # Handle different unembed structures
        if hasattr(unembed, 'lm_head'):
            W = unembed.lm_head.weight  # [V, d]
            b = getattr(unembed.lm_head, 'bias', None)
        elif hasattr(unembed, 'weight'):
            W = unembed.weight
            b = getattr(unembed, 'bias', None)
        else:
            raise ValueError("Cannot extract weight matrix from unembed")

        return W, b

    def _apply_lens_projection(
        self,
        h: torch.Tensor,
        lens: "BaseLens",
        layer,
    ) -> torch.Tensor:
        """Apply lens-specific projection (e.g., LoRA transform + layer norm)."""
        # Get the lens module (unwrap DDP if needed)
        raw_lens = lens.module if hasattr(lens, 'module') else lens

        # Apply per-layer projection if present (LoRA/Tuned lens)
        if hasattr(raw_lens, 'projections'):
            from loralens.lenses.types import canonical_layer_id
            lid = canonical_layer_id(layer)
            if lid in raw_lens.projections:
                proj = raw_lens.projections[lid]
                h = proj(h)  # [T, d] -> [T, d]

        # Apply layer norm if present in unembed
        unembed = raw_lens.unembed
        if hasattr(unembed, 'ln_f') and unembed.ln_f is not None:
            h = unembed.ln_f(h)
        elif hasattr(unembed, 'layer_norm') and unembed.layer_norm is not None:
            h = unembed.layer_norm(h)

        return h

    def _build_candidate_set(
        self,
        teacher_seq: torch.Tensor,  # [T, V]
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build shared candidate set S for a sequence.

        Returns
        -------
        S : torch.Tensor
            Shared candidate token indices [K], sorted.
        pos_cols : torch.Tensor
            Per-position column indices into S [T, m], padded with 0.
        pos_probs : torch.Tensor
            Per-position teacher probs [T, m], padded with 0.
        """
        T, V = teacher_seq.shape
        top_m = self.config.top_m
        sample_r = self.config.sample_r
        max_K = self.config.max_K

        with torch.no_grad():
            # Importance-sample per position from teacher distribution
            probs = F.softmax(teacher_seq.float(), dim=-1)  # [T, V]
            m = min(top_m + sample_r, V)
            if m <= 0:
                raise ValueError("top_m + sample_r must be > 0 to build candidate set")
            # Sample without replacement to get unique per-position candidates
            sampled_idx = torch.multinomial(probs, num_samples=m, replacement=False)  # [T, m]
            sampled_probs = torch.gather(probs, -1, sampled_idx)  # [T, m]

            # Build union of all sampled tokens
            all_candidates = sampled_idx.view(-1).unique()  # Unique token IDs

            # If union exceeds max_K, prune by aggregated mass
            if all_candidates.numel() > max_K:
                # Score = sum of teacher probs where token appears
                scores = torch.zeros(V, device=device, dtype=torch.float32)
                flat_idx = sampled_idx.view(-1)  # [T*m]
                flat_probs = sampled_probs.view(-1)  # [T*m]
                scores.scatter_add_(0, flat_idx, flat_probs)

                # Keep top-K by score
                _, top_K_idx = scores[all_candidates].topk(k=max_K)
                S = all_candidates[top_K_idx]
            else:
                S = all_candidates

            S = S.sort().values  # Sort for consistent ordering
            K = S.numel()

            # Build token-to-column mapping
            token_to_col = torch.full((V,), -1, dtype=torch.long, device=device)
            token_to_col[S] = torch.arange(K, device=device)

            # Map each position's top-m to columns in S
            pos_cols_raw = token_to_col[sampled_idx]  # [T, m], may have -1 for pruned

            # Create valid mask and handle pruned tokens
            valid_mask = pos_cols_raw >= 0  # [T, m]

            # Replace -1 with 0 (will be masked in loss computation)
            pos_cols = pos_cols_raw.clamp(min=0)  # [T, m]

            # Zero out probs for pruned tokens and renormalize
            pos_probs = sampled_probs * valid_mask.float()  # [T, m]
            pos_probs = pos_probs / pos_probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)

            # Track coverage
            self._last_coverage = float(valid_mask.float().mean())

        return S, pos_cols, pos_probs

    def _compute_position_kl_vectorized(
        self,
        student_logits_S: torch.Tensor,  # [T, K]
        pos_cols: torch.Tensor,  # [T, m]
        pos_probs: torch.Tensor,  # [T, m]
        mask_seq: Optional[torch.Tensor],  # [T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized KL computation over all positions.

        For each position, computes KL only over its m support tokens.
        """
        T, K = student_logits_S.shape
        m = pos_cols.shape[1]
        device = student_logits_S.device

        # Gather student logits for each position's support: [T, m]
        student_logits_m = torch.gather(student_logits_S, 1, pos_cols)

        # Renormalize student over support (log_softmax)
        # Mask out zero-prob entries before softmax
        valid_mask = pos_probs > 1e-10
        student_logits_m = student_logits_m.masked_fill(~valid_mask, float('-inf'))
        student_logprobs_m = F.log_softmax(student_logits_m, dim=-1)

        # Handle -inf from masked positions
        student_logprobs_m = student_logprobs_m.masked_fill(~valid_mask, 0.0)

        # Teacher log probs
        teacher_logprobs_m = (pos_probs + 1e-10).log()
        teacher_logprobs_m = teacher_logprobs_m.masked_fill(~valid_mask, 0.0)

        # KL per position: sum_i p_i * (log p_i - log q_i)
        kl_per_pos = (pos_probs * (teacher_logprobs_m - student_logprobs_m)).sum(dim=-1)  # [T]

        # Apply attention mask
        if mask_seq is not None:
            kl_per_pos = kl_per_pos * mask_seq.to(kl_per_pos.dtype)
            count = mask_seq.sum(dtype=torch.float32)
        else:
            count = torch.tensor(T, device=device, dtype=torch.float32)

        return kl_per_pos.sum(dtype=torch.float32), count

    @property
    def last_K(self) -> Optional[int]:
        """Size of last shared candidate set (for monitoring)."""
        return self._last_K

    @property
    def last_coverage(self) -> Optional[float]:
        """Fraction of valid candidates after pruning (for monitoring)."""
        return self._last_coverage

    def __repr__(self) -> str:
        return (
            f"SharedSubsetKLLoss(top_m={self.config.top_m}, "
            f"max_K={self.config.max_K}, reduction={self.reduction!r})"
        )


def verify_logits_match(
    hidden_states: torch.Tensor,
    lens: "BaseLens",
    layer,
    S: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Tuple[bool, float]:
    """
    Verify that shared subset logits match dense computation.

    This is for testing/validation that our GEMM produces exact results.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Activations [T, d].
    lens : BaseLens
        Lens module.
    layer : LayerId
        Layer identifier.
    S : torch.Tensor
        Candidate indices [K].
    rtol, atol : float
        Tolerance for allclose.

    Returns
    -------
    match : bool
        Whether logits match within tolerance.
    max_diff : float
        Maximum absolute difference.
    """
    raw_lens = lens.module if hasattr(lens, 'module') else lens

    # Get unembed
    unembed = raw_lens.unembed
    if hasattr(unembed, 'lm_head'):
        W = unembed.lm_head.weight
        b = getattr(unembed.lm_head, 'bias', None)
    else:
        W = unembed.weight
        b = getattr(unembed, 'bias', None)

    # Apply projection
    h = hidden_states.clone()
    if hasattr(raw_lens, 'projections'):
        from loralens.lenses.types import canonical_layer_id
        lid = canonical_layer_id(layer)
        if lid in raw_lens.projections:
            h = raw_lens.projections[lid](h)

    if hasattr(unembed, 'ln_f') and unembed.ln_f is not None:
        h = unembed.ln_f(h)

    # Dense computation
    dense_logits = h @ W.T  # [T, V]
    if b is not None:
        dense_logits = dense_logits + b
    dense_subset = dense_logits[:, S]  # [T, K]

    # Shared subset computation
    W_S = W[S]
    subset_logits = h @ W_S.T
    if b is not None:
        subset_logits = subset_logits + b[S]

    # Compare
    max_diff = (dense_subset - subset_logits).abs().max().item()
    match = torch.allclose(dense_subset, subset_logits, rtol=rtol, atol=atol)

    return match, max_diff
