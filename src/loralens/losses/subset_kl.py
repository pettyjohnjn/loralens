# src/loralens/losses/subset_kl.py
"""
Subset KL divergence loss for memory-efficient lens training.

Thin adapter around ``subset_kl.SubsetKLLoss`` that conforms to the loralens
``BaseLoss`` interface. Two estimators are supported:

- ``"topk"`` (default): top-k renormalized KL over the teacher's top-k tokens.
  Fast and recommended for most runs; delegates to ``subset_kl.SubsetKLLoss``.
- ``"mc"``: exact KL on the top-k head plus an importance-weighted Monte Carlo
  estimate of the tail, sampled from the (normalized) teacher tail. Enable the
  tail term with ``k_tail > 0``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from subset_kl import SubsetKLLoss as _ExternalSubsetKL
from subset_kl import ReductionType

from .base import BaseLoss

if TYPE_CHECKING:
    from loralens.lenses import BaseLens


SubsetMode = Literal["topk", "mc"]


def _mc_teacher_subset(
    teacher_logits: torch.Tensor,
    k_head: int,
    k_tail: int,
) -> dict[str, torch.Tensor]:
    """Build a deterministic top-k head plus teacher-tail Monte Carlo samples.

    The tail proposal is the normalized teacher tail distribution (sampling with
    replacement), which gives the standard importance-weighted MC tail estimate.
    """
    N, V = teacher_logits.shape
    device = teacher_logits.device
    teacher_logits_f = teacher_logits.float()

    with torch.no_grad():
        teacher_log_probs = teacher_logits_f - teacher_logits_f.logsumexp(dim=-1, keepdim=True)
        teacher_probs = teacher_log_probs.exp()

        if k_head > 0:
            _, head_idx = teacher_logits_f.topk(k_head, dim=-1)
            head_log_probs = torch.gather(teacher_log_probs, 1, head_idx)
            p_head = torch.gather(teacher_probs, 1, head_idx).sum(dim=-1)
        else:
            head_idx = torch.empty(N, 0, device=device, dtype=torch.long)
            head_log_probs = torch.empty(N, 0, device=device, dtype=torch.float32)
            p_head = torch.zeros(N, device=device, dtype=torch.float32)

        if k_tail > 0:
            if k_head >= V:
                raise ValueError("Cannot sample tail when k_head covers the vocabulary")
            tail_probs = teacher_probs.clone()
            if k_head > 0:
                tail_probs.scatter_(1, head_idx, 0.0)
            tail_mass = tail_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            proposal_probs = tail_probs / tail_mass  # normalized teacher tail
            tail_idx = torch.multinomial(proposal_probs, num_samples=k_tail, replacement=True)
            tail_log_probs = torch.gather(teacher_log_probs, 1, tail_idx)
            tail_proposal_log_probs = torch.gather(
                proposal_probs.clamp_min(1e-45).log(), 1, tail_idx
            )
        else:
            tail_idx = torch.empty(N, 0, device=device, dtype=torch.long)
            tail_log_probs = torch.empty(N, 0, device=device, dtype=torch.float32)
            tail_proposal_log_probs = torch.empty(N, 0, device=device, dtype=torch.float32)

        indices = torch.cat([head_idx, tail_idx], dim=-1)

    return {
        "indices": indices,
        "teacher_head_log_probs": head_log_probs,
        "teacher_tail_log_probs": tail_log_probs,
        "tail_proposal_log_probs": tail_proposal_log_probs,
        "p_head": p_head,
    }


def _mc_tail_kl(
    teacher_head_log_probs: torch.Tensor,
    teacher_tail_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    p_head: torch.Tensor,
    k_head: int,
    k_tail: int,
    tail_proposal_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Exact head KL plus importance-weighted MC tail estimate.

    ``student_log_probs`` must be normalized over the full vocabulary (i.e.
    computed with the true logsumexp), so the head and tail terms compose into
    an unbiased estimate of the full KL.
    """
    if k_head > 0:
        student_head_log_probs = student_log_probs[:, :k_head]
        head_kl = (
            teacher_head_log_probs.exp()
            * (teacher_head_log_probs - student_head_log_probs)
        ).sum(dim=-1)
    else:
        head_kl = torch.zeros(
            student_log_probs.shape[0],
            device=student_log_probs.device,
            dtype=torch.float32,
        )

    if k_tail > 0:
        student_tail_log_probs = student_log_probs[:, k_head:]
        tail_terms = teacher_tail_log_probs - student_tail_log_probs
        weights = (teacher_tail_log_probs - tail_proposal_log_probs).exp()
        tail_estimate = (weights * tail_terms).sum(dim=-1) / k_tail
        return head_kl + tail_estimate
    return head_kl


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
        if self.mode == "mc":
            B, T, V = teacher_logits.shape
            subset = _mc_teacher_subset(
                teacher_logits=teacher_logits.view(B * T, V),
                k_head=self.k,
                k_tail=self.k_tail,
            )
            indices = subset["indices"]
            student_flat = student_logits.view(B * T, V)
            student_selected = torch.gather(student_flat, -1, indices)
            student_logsumexp = student_flat.float().logsumexp(dim=-1, keepdim=True)
            student_log_probs = student_selected.float() - student_logsumexp
            kl = _mc_tail_kl(
                teacher_head_log_probs=subset["teacher_head_log_probs"],
                teacher_tail_log_probs=subset["teacher_tail_log_probs"],
                student_log_probs=student_log_probs,
                p_head=subset["p_head"],
                k_head=self.k,
                k_tail=self.k_tail,
                tail_proposal_log_probs=subset["tail_proposal_log_probs"],
            )
            return self._apply_reduction(kl.view(B, T), attention_mask)

        # "topk" — delegate to external SubsetKLLoss
        return self._inner.forward(student_logits, teacher_logits, attention_mask)

    # --- lens-aware memory-efficient path ----

    def prepare_teacher_subset(self, teacher_logits: torch.Tensor) -> dict[str, Any]:
        """Precompute teacher-side subset selection for reuse across layers."""
        if self.mode == "mc":
            return self._prepare_teacher_subset_mc(teacher_logits)
        return self._prepare_teacher_subset_topk(teacher_logits)

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
        calls ``lens.forward(..., vocab_indices=...)`` to compute student logits
        only for the selected subset.

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
            subset = self.prepare_teacher_subset(teacher_logits)
        else:
            subset = teacher_subset

        if subset["mode"] == "mc":
            return self._forward_with_lens_mc(hidden_states, lens, layer, attention_mask, subset)
        return self._forward_with_lens_topk(hidden_states, lens, layer, attention_mask, subset)

    def _prepare_teacher_subset_topk(self, teacher_logits: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            top_vals, top_idx = teacher_logits.topk(k=self.k, dim=-1)
            teacher_logprobs_k = F.log_softmax(top_vals, dim=-1)
            teacher_probs_k = teacher_logprobs_k.exp()
        return {
            "mode": "topk",
            "indices_3d": top_idx,
            "teacher_logprobs_k": teacher_logprobs_k,
            "teacher_probs_k": teacher_probs_k,
        }

    def _prepare_teacher_subset_mc(self, teacher_logits: torch.Tensor) -> dict[str, Any]:
        B, T, V = teacher_logits.shape
        subset = _mc_teacher_subset(
            teacher_logits=teacher_logits.view(B * T, V),
            k_head=self.k,
            k_tail=self.k_tail,
        )
        indices = subset["indices"]
        S = indices.shape[-1]
        return {
            "mode": "mc",
            "indices": indices,
            "indices_3d": indices.view(B, T, S),
            "teacher_head_log_probs": subset["teacher_head_log_probs"],
            "teacher_tail_log_probs": subset["teacher_tail_log_probs"],
            "tail_proposal_log_probs": subset["tail_proposal_log_probs"],
            "p_head": subset["p_head"],
        }

    def _forward_with_lens_topk(
        self,
        hidden_states: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_subset: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Simple top-k with memory-efficient lens computation."""
        if teacher_subset is None:
            raise ValueError("teacher_subset is required")

        student_logits_k = lens.forward(
            hidden_states,
            layer=layer,
            vocab_indices=teacher_subset["indices_3d"],
        ).logits

        student_logprobs_k = F.log_softmax(student_logits_k, dim=-1)
        kl = (
            teacher_subset["teacher_probs_k"]
            * (teacher_subset["teacher_logprobs_k"] - student_logprobs_k)
        ).sum(dim=-1)
        return self._apply_reduction(kl, attention_mask)

    def _forward_with_lens_mc(
        self,
        hidden_states: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_subset: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Exact head KL plus teacher-tail MC estimate."""
        if teacher_subset is None:
            raise ValueError("teacher_subset is required")

        B, T, S = teacher_subset["indices_3d"].shape
        raw_lens = lens.module if hasattr(lens, "module") else lens

        student_logits_subset, student_logsumexp = raw_lens.compute_logits_subset_with_logsumexp(
            activations=hidden_states,
            vocab_indices=teacher_subset["indices_3d"],
            layer=layer,
        )
        student_log_probs = (
            student_logits_subset.view(B * T, S).float()
            - student_logsumexp.view(B * T, 1).float()
        )

        kl_flat = _mc_tail_kl(
            teacher_head_log_probs=teacher_subset["teacher_head_log_probs"],
            teacher_tail_log_probs=teacher_subset["teacher_tail_log_probs"],
            student_log_probs=student_log_probs,
            p_head=teacher_subset["p_head"],
            k_head=self.k,
            k_tail=self.k_tail,
            tail_proposal_log_probs=teacher_subset["tail_proposal_log_probs"],
        )

        return self._apply_reduction(kl_flat.view(B, T), attention_mask)

    def __repr__(self) -> str:
        return (
            f"SubsetKLLoss(k={self.k}, mode={self.mode!r}, "
            f"reduction={self.reduction!r})"
        )
