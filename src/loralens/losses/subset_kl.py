# src/loralens/losses/subset_kl.py
"""
Subset KL divergence loss for memory-efficient lens training.

Thin adapter that delegates the KL math to the ``subset_kl`` package and adapts
it to the loralens ``BaseLoss`` interface and the lens-aware (per-layer,
gathered) training path. Two estimators are supported:

- ``"topk"`` (default): top-k renormalized KL over the teacher's top-k tokens.
  Fast and recommended for most runs.
- ``"mc"``: exact KL on the top-k head plus an importance-weighted Monte Carlo
  estimate of the tail, sampled from the (normalized) teacher tail. Enable the
  tail term with ``k_tail > 0``.

The estimator implementations live in ``subset_kl`` (single source of truth);
this module only wires the lens's subset-logit outputs into them.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TYPE_CHECKING

import torch

from subset_kl import ReductionType
from subset_kl import SubsetKLLoss as _ExternalSubsetKL
from subset_kl import (
    compute_subset_mc_kl,
    select_head_tail_indices,
    select_topk_indices,
    subset_kl_from_gathered,
    subset_mc_kl_from_gathered,
)

from .base import BaseLoss

if TYPE_CHECKING:
    from loralens.lenses import BaseLens


SubsetMode = Literal["topk", "mc"]


class SubsetKLLoss(BaseLoss):
    """
    Memory-efficient subset KL loss.

    Computes KL divergence using only a subset of vocabulary tokens, avoiding
    full [B, T, V] materialisation.

    Parameters
    ----------
    k : int
        Number of top tokens (head) for both modes.
    mode : str
        ``"topk"`` (default): top-k renormalization — recommended.
        ``"mc"``: exact head KL + importance-weighted teacher-tail MC estimate.
    reduction : str
        How to reduce across tokens: ``"mean"`` or ``"sum"``.
    k_tail : int
        (``"mc"`` mode) Number of tail samples per token. Default 0 disables the
        tail term and behaves like ``"topk"``.

    Examples
    --------
    Simple top-k (recommended for most runs):

    >>> loss_fn = SubsetKLLoss(k=256)

    MC estimator with tail correction:

    >>> loss_fn = SubsetKLLoss(k=128, mode="mc", k_tail=64)
    """

    def __init__(
        self,
        k: int = 128,
        mode: SubsetMode = "topk",
        reduction: ReductionType = "mean",
        k_tail: int = 0,
    ) -> None:
        super().__init__(reduction=reduction)
        valid_modes = {"topk", "mc"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown subset KL mode {mode!r}; expected one of {sorted(valid_modes)}")
        self.k = k
        self.k_tail = k_tail
        self.mode = mode

        self._inner = _ExternalSubsetKL(k=self.k, reduction=reduction)

    # --- BaseLoss interface (full student logits) ----

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute subset KL given pre-computed (full) student logits.

        ``labels`` is accepted for interface compatibility but unused.
        """
        if self.mode == "mc":
            return compute_subset_mc_kl(
                student_logits,
                teacher_logits,
                k_head=self.k,
                k_tail=self.k_tail,
                attention_mask=attention_mask,
                reduction=self.reduction,
            )
        # "topk" — delegate to the external SubsetKLLoss
        return self._inner.forward(student_logits, teacher_logits, attention_mask)

    # --- lens-aware memory-efficient path ----

    def prepare_teacher_subset(self, teacher_logits: torch.Tensor) -> dict[str, Any]:
        """Precompute the teacher-side subset selection for reuse across layers."""
        if self.mode == "mc":
            indices, teacher_log_probs_selected, p_head = select_head_tail_indices(
                teacher_logits, k_head=self.k, k_tail=self.k_tail
            )
            return {
                "mode": "mc",
                "indices_3d": indices,
                "teacher_log_probs_selected": teacher_log_probs_selected,
                "p_head": p_head,
            }
        indices, teacher_logits_k = select_topk_indices(teacher_logits, self.k)
        return {"mode": "topk", "indices_3d": indices, "teacher_logits_k": teacher_logits_k}

    def forward_with_lens(
        self,
        hidden_states: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_subset: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Maximum memory efficiency: compute only k student logits.

        This path never materializes full [B, T, V] student logits. Instead it
        asks the lens for logits at the selected ``vocab_indices`` only, then
        hands those to the ``subset_kl`` estimators.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Activations [batch, seq, hidden].
        teacher_logits : Optional[torch.Tensor]
            Teacher logits [batch, seq, vocab]. Required when ``teacher_subset``
            is not provided.
        lens : BaseLens
            Lens module.
        layer
            Layer identifier.
        attention_mask : Optional[torch.Tensor]
            Mask [batch, seq].
        teacher_subset : Optional[dict]
            Pre-computed subset from ``prepare_teacher_subset()``. Pass this when
            reusing the same subset across multiple layers.
        """
        if teacher_subset is None:
            if teacher_logits is None:
                raise ValueError("teacher_logits is required when teacher_subset is not provided")
            teacher_subset = self.prepare_teacher_subset(teacher_logits)

        if teacher_subset["mode"] == "mc":
            return self._forward_with_lens_mc(hidden_states, lens, layer, attention_mask, teacher_subset)
        return self._forward_with_lens_topk(hidden_states, lens, layer, attention_mask, teacher_subset)

    def _forward_with_lens_topk(
        self,
        hidden_states: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor],
        teacher_subset: dict[str, Any],
    ) -> torch.Tensor:
        """Top-k renormalized KL on lens-computed subset logits."""
        student_logits_k = lens.forward(
            hidden_states,
            layer=layer,
            vocab_indices=teacher_subset["indices_3d"],
        ).logits
        return subset_kl_from_gathered(
            student_logits_k,
            teacher_subset["teacher_logits_k"],
            attention_mask,
            self.reduction,
        )

    def _forward_with_lens_mc(
        self,
        hidden_states: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor],
        teacher_subset: dict[str, Any],
    ) -> torch.Tensor:
        """Exact head KL plus teacher-tail MC estimate on lens-computed subset logits.

        The MC tail needs true full-vocabulary student log-probs, so the lens
        returns both the subset logits and the full-vocab logsumexp.
        """
        raw_lens = lens.module if hasattr(lens, "module") else lens
        student_logits_subset, student_logsumexp = raw_lens.compute_logits_subset_with_logsumexp(
            activations=hidden_states,
            vocab_indices=teacher_subset["indices_3d"],
            layer=layer,
        )
        return subset_mc_kl_from_gathered(
            student_logits_subset,
            teacher_subset["teacher_log_probs_selected"],
            k_head=self.k,
            k_tail=self.k_tail,
            p_head=teacher_subset["p_head"],
            student_log_normalizer=student_logsumexp.squeeze(-1),
            attention_mask=attention_mask,
            reduction=self.reduction,
        )

    def __repr__(self) -> str:
        return (
            f"SubsetKLLoss(k={self.k}, mode={self.mode!r}, "
            f"reduction={self.reduction!r})"
        )
