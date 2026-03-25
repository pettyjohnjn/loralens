# src/loralens/losses/subset_kl.py
"""
Subset KL divergence loss for memory-efficient lens training.

Thin adapter around ``subset_kl.SubsetKLLoss`` and ``subset_kl.HajekKLLoss``
that conforms to the loralens ``BaseLoss`` interface.

Two modes are supported:
- ``"topk"`` (default): delegates to ``subset_kl.SubsetKLLoss``
- ``"hajek"``: delegates to ``subset_kl.HajekKLLoss``
"""

from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from subset_kl import SubsetKLLoss as _ExternalSubsetKL
from subset_kl import HajekKLLoss as _ExternalHajek
from subset_kl import ReductionType

from .base import BaseLoss

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
    k_head : Optional[int]
        Legacy alias for ``k``.

    Examples
    --------
    Simple top-k (recommended):
    >>> loss_fn = SubsetKLLoss(k=256)

    Hajek estimator:
    >>> loss_fn = SubsetKLLoss(k=128, mode="hajek", k_tail=64)
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
        # Legacy parameter
        k_head: Optional[int] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        self.k = k_head if k_head is not None else k
        self.k_head = self.k  # alias for backward compat
        self.k_tail = k_tail
        self.mode = mode
        self.tail_clip = tail_clip
        self.tail_oversample = tail_oversample

        # Build the delegate
        if mode == "hajek" and k_tail > 0:
            self._inner = _ExternalHajek(
                k_head=self.k,
                k_tail=k_tail,
                reduction=reduction,
                weight_clip=tail_clip,
                oversample=tail_oversample,
            )
        else:
            self._inner = _ExternalSubsetKL(k=self.k, reduction=reduction)

    # --- public helpers delegated to the external class ----

    def select_indices(self, teacher_logits: torch.Tensor):
        """Select top-k indices from teacher (delegated to subset_kl)."""
        if hasattr(self._inner, "select_indices"):
            return self._inner.select_indices(teacher_logits)
        raise AttributeError("select_indices not available in hajek mode")

    def forward_gathered(
        self,
        student_k: torch.Tensor,
        teacher_k: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL from pre-gathered k-way tensors."""
        if hasattr(self._inner, "forward_gathered"):
            return self._inner.forward_gathered(student_k, teacher_k, attention_mask)
        raise AttributeError("forward_gathered not available in hajek mode")

    # --- BaseLoss interface ----

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute subset KL given pre-computed student logits.

        ``labels`` is accepted for interface compatibility but unused.
        """
        return self._inner.forward(student_logits, teacher_logits, attention_mask)

    # --- lens-aware memory-efficient path ----

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
        Instead it calls ``lens.forward(..., vocab_indices=...)`` to compute
        student logits only for the selected subset.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Activations [batch, seq, hidden].
        teacher_logits : torch.Tensor
            Teacher logits [batch, seq, vocab].
        lens : BaseLens
            Lens module.
        layer
            Layer identifier.
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
        """Simple top-k with memory-efficient lens computation."""
        with torch.no_grad():
            top_vals, top_idx = teacher_logits.topk(k=self.k, dim=-1)
            teacher_logprobs_k = F.log_softmax(top_vals, dim=-1)
            teacher_probs_k = teacher_logprobs_k.exp()

        # Compute student logits ONLY for top-k indices via lens
        student_logits_k = lens.forward(
            hidden_states,
            layer=layer,
            vocab_indices=top_idx,
        ).logits

        student_logprobs_k = F.log_softmax(student_logits_k, dim=-1)
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
        """Hajek estimator with memory-efficient lens computation."""
        from .sampling import pps_sample_indices_batched, hajek_kl_estimate

        B, T, V = teacher_logits.shape

        teacher_flat = teacher_logits.view(B * T, V)

        with torch.no_grad():
            teacher_log_probs = F.log_softmax(teacher_flat, dim=-1)

            indices, inc_probs, mask, diagnostics = pps_sample_indices_batched(
                teacher_log_probs,
                k_head=self.k,
                k_tail=self.k_tail,
                oversample=self.tail_oversample,
            )

            teacher_sel = torch.gather(teacher_log_probs, -1, indices)

            S = indices.shape[-1]
            indices_3d = indices.view(B, T, S)

        # Compute student logits ONLY for selected indices via lens
        student_logits_subset = lens.forward(
            hidden_states,
            layer=layer,
            vocab_indices=indices_3d,
        ).logits

        student_sel = student_logits_subset.view(B * T, S)
        student_log_probs_sel = F.log_softmax(student_sel, dim=-1)

        kl_flat = hajek_kl_estimate(
            teacher_log_probs=teacher_sel,
            student_log_probs=student_log_probs_sel,
            inclusion_probs=inc_probs,
            sample_mask=mask,
            weight_clip=self.tail_clip,
        )

        kl = kl_flat.view(B, T)
        return self._apply_reduction(kl, attention_mask)

    def __repr__(self) -> str:
        return (
            f"SubsetKLLoss(k={self.k}, mode={self.mode!r}, "
            f"reduction={self.reduction!r})"
        )
