# src/loralens/losses/subset_kl.py
"""
Subset KL divergence loss for memory-efficient lens training.

Thin adapter around ``subset_kl.SubsetKLLoss`` plus local lens-aware
head/tail estimators
that conforms to the loralens ``BaseLoss`` interface.

Six modes are supported:
- ``"topk"`` (default): delegates to ``subset_kl.SubsetKLLoss``
- ``"frankenstein"``: legacy clipped self-normalized PPS estimator
- ``"hajek"``: exact head plus self-normalized teacher-tail Hajek estimator
- ``"mc"``: exact head plus teacher-tail Monte Carlo estimator
- ``"k2"``: top-k head plus teacher-tail Schulman K2 penalty
- ``"k3"``: top-k head plus teacher-tail Schulman K3 estimator
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


SubsetMode = Literal["topk", "frankenstein", "hajek", "mc", "k2", "k3"]
TailProposal = Literal["target", "teacher", "mixed", "tempered"]


def _tail_proposal_from_probs(
    tail_probs: torch.Tensor,
    tail_mass: torch.Tensor,
    proposal: TailProposal,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    if proposal == "teacher":
        proposal = "target"
    if proposal not in {"target", "mixed", "tempered"}:
        raise ValueError("tail proposal must be target, teacher, mixed, or tempered")
    q_target = tail_probs / tail_mass
    if proposal == "target":
        return q_target
    q_explore = tail_probs.pow(tau)
    q_explore = q_explore / q_explore.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q_explore if proposal == "tempered" else alpha * q_target + (1.0 - alpha) * q_explore
    return q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _prepare_hajek_teacher_subset(
    teacher_logits: torch.Tensor,
    k_head: int,
    k_tail: int,
    oversample: int,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Build a head-tail subset without materializing extra full-vocab tensors."""
    N, V = teacher_logits.shape
    device = teacher_logits.device
    teacher_logits_f = teacher_logits.float()

    with torch.no_grad():
        teacher_logsumexp = teacher_logits_f.logsumexp(dim=-1, keepdim=True)
        teacher_probs = (teacher_logits_f - teacher_logsumexp).exp()

        if k_head > 0:
            _, top_idx = teacher_logits_f.topk(k_head, dim=-1)
            top_probs = torch.gather(teacher_probs, 1, top_idx)
        else:
            top_idx = torch.empty(N, 0, device=device, dtype=torch.long)
            top_probs = torch.empty(N, 0, device=device, dtype=teacher_probs.dtype)

        tail_probs = teacher_probs
        if k_head > 0:
            tail_probs.scatter_(1, top_idx, 0.0)
            tail_mass = (1.0 - top_probs.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
        else:
            tail_mass = tail_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        tail_probs = tail_probs / tail_mass

        num_draws = k_tail * oversample
        tail_sample_count = min(num_draws, max(V - k_head, 0))
        if tail_sample_count > 0 and (tail_mass > 1e-10).any():
            sampled_idx = torch.multinomial(
                tail_probs,
                num_samples=tail_sample_count,
                replacement=True,
            )
            all_idx = torch.cat([top_idx, sampled_idx], dim=-1)
            all_is_head = torch.cat(
                [
                    torch.ones_like(top_idx, dtype=torch.bool),
                    torch.zeros_like(sampled_idx, dtype=torch.bool),
                ],
                dim=-1,
            )

            sort_order = all_idx.argsort(dim=-1)
            all_idx_sorted = torch.gather(all_idx, 1, sort_order)
            all_is_head_sorted = torch.gather(all_is_head, 1, sort_order)

            diff = torch.diff(
                all_idx_sorted,
                dim=-1,
                prepend=all_idx_sorted[:, :1] - 1,
            )
            keep_mask = diff != 0
            counts = keep_mask.sum(dim=-1)
            S_max = min(k_head + k_tail, int(counts.max().item()))
            positions = keep_mask.cumsum(dim=-1) - 1
            valid = keep_mask & (positions < S_max)

            indices = torch.zeros(N, S_max, device=device, dtype=torch.long)
            mask = torch.zeros(N, S_max, device=device, dtype=torch.bool)
            row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(all_idx_sorted)
            indices[row_idx[valid], positions[valid]] = all_idx_sorted[valid]
            mask[row_idx[valid], positions[valid]] = True

            head_counts = torch.zeros(N, S_max, device=device, dtype=torch.int32)
            clipped_positions = positions.clamp(min=0, max=max(S_max - 1, 0))
            head_src = all_is_head_sorted.to(torch.int32) * (positions < S_max).to(torch.int32)
            head_counts.scatter_add_(1, clipped_positions, head_src)
            is_head = head_counts > 0
        else:
            indices = top_idx
            mask = torch.ones(N, top_idx.shape[-1], device=device, dtype=torch.bool)
            is_head = torch.ones_like(indices, dtype=torch.bool)
            S_max = indices.shape[-1]

        teacher_log_probs_sel = torch.gather(teacher_logits_f, -1, indices) - teacher_logsumexp
        tail_probs_sel = torch.gather(tail_probs, -1, indices) if indices.numel() else teacher_log_probs_sel.new_zeros(indices.shape)

        inclusion_probs = torch.ones_like(teacher_log_probs_sel, dtype=torch.float32)
        if num_draws > 0 and indices.numel():
            tail_inclusion = 1.0 - torch.pow((1.0 - tail_probs_sel).clamp_min(0.0), num_draws)
            inclusion_probs = torch.where(is_head, torch.ones_like(tail_inclusion), tail_inclusion)
            inclusion_probs = inclusion_probs.clamp_min(1e-8)

        valid_inclusion = inclusion_probs[mask]
        diagnostics = {
            "num_unique_indices": float(mask.sum().item() / max(N, 1)),
            "mean_inclusion_prob": valid_inclusion.mean().item() if valid_inclusion.numel() else 1.0,
            "min_inclusion_prob": valid_inclusion.min().item() if valid_inclusion.numel() else 1.0,
            "max_importance_weight": (1.0 / valid_inclusion.min()).item() if valid_inclusion.numel() else 1.0,
            "tail_sample_count": float(tail_sample_count),
        }

    return {
        "indices": indices,
        "teacher_log_probs_sel": teacher_log_probs_sel,
        "inclusion_probs": inclusion_probs,
        "mask": mask,
        "diagnostics": diagnostics,
    }


def _prepare_k2_teacher_subset(
    teacher_logits: torch.Tensor,
    k_head: int,
    k_tail: int,
    tail_proposal: TailProposal = "target",
    tail_proposal_alpha: float = 0.8,
    tail_proposal_tau: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Build deterministic head plus with-replacement samples from the teacher tail."""
    N, V = teacher_logits.shape
    device = teacher_logits.device
    teacher_logits_f = teacher_logits.float()

    with torch.no_grad():
        teacher_logsumexp = teacher_logits_f.logsumexp(dim=-1, keepdim=True)
        teacher_log_probs = teacher_logits_f - teacher_logsumexp
        teacher_probs = teacher_log_probs.exp()

        if k_head > 0:
            _, head_idx = teacher_logits_f.topk(k_head, dim=-1)
            head_log_probs = torch.gather(teacher_log_probs, 1, head_idx)
            head_logprobs_k = head_log_probs - torch.logsumexp(
                head_log_probs, dim=-1, keepdim=True
            )
            head_probs_k = head_logprobs_k.exp()
            p_head = torch.gather(teacher_probs, 1, head_idx).sum(dim=-1)
        else:
            head_idx = torch.empty(N, 0, device=device, dtype=torch.long)
            head_log_probs = torch.empty(N, 0, device=device, dtype=torch.float32)
            head_logprobs_k = torch.empty(N, 0, device=device, dtype=torch.float32)
            head_probs_k = torch.empty(N, 0, device=device, dtype=torch.float32)
            p_head = torch.zeros(N, device=device, dtype=torch.float32)

        if k_tail > 0:
            if k_head >= V:
                raise ValueError("Cannot sample K2 tail when k_head covers the vocabulary")
            tail_probs = teacher_probs.clone()
            if k_head > 0:
                tail_probs.scatter_(1, head_idx, 0.0)
            tail_mass = tail_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            proposal_probs = _tail_proposal_from_probs(
                tail_probs,
                tail_mass,
                proposal=tail_proposal,
                alpha=tail_proposal_alpha,
                tau=tail_proposal_tau,
            )
            tail_idx = torch.multinomial(
                proposal_probs,
                num_samples=k_tail,
                replacement=True,
            )
            tail_log_probs = torch.gather(teacher_log_probs, 1, tail_idx)
            tail_proposal_log_probs = torch.gather(
                proposal_probs.clamp_min(1e-45).log(), 1, tail_idx
            )
        else:
            tail_idx = torch.empty(N, 0, device=device, dtype=torch.long)
            tail_log_probs = torch.empty(N, 0, device=device, dtype=torch.float32)
            tail_proposal_log_probs = torch.empty(N, 0, device=device, dtype=torch.float32)

        indices = torch.cat([head_idx, tail_idx], dim=-1)
        teacher_tail_log_probs = tail_log_probs

    return {
        "indices": indices,
        "teacher_head_log_probs": head_log_probs,
        "teacher_head_logprobs_k": head_logprobs_k,
        "teacher_head_probs_k": head_probs_k,
        "teacher_tail_log_probs": teacher_tail_log_probs,
        "tail_proposal_log_probs": tail_proposal_log_probs,
        "p_head": p_head,
    }


def _hajek_kl_from_student_log_probs(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    inclusion_probs: torch.Tensor,
    mask: torch.Tensor,
    weight_clip: float,
) -> tuple[torch.Tensor, float]:
    """Self-normalized head-tail KL using true student log-probabilities."""
    weights = (1.0 / inclusion_probs).clamp(max=weight_clip) * mask.float()
    teacher_probs = teacher_log_probs.exp()
    kl_terms = teacher_probs * (teacher_log_probs - student_log_probs) * weights
    numerator = (kl_terms * mask.float()).sum(dim=-1)
    denominator = (teacher_probs * weights * mask.float()).sum(dim=-1).clamp_min(1e-8)
    kl = numerator / denominator

    w_normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    variance_proxy = (w_normalized.square() * mask.float()).sum().item() / max(mask.shape[0], 1)
    return kl, variance_proxy


def _exact_head_tail_kl_from_student_log_probs(
    teacher_head_log_probs: torch.Tensor,
    teacher_tail_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    p_head: torch.Tensor,
    k_head: int,
    k_tail: int,
    tail_proposal_log_probs: Optional[torch.Tensor] = None,
    tail_estimator: Literal["mc", "hajek"] = "mc",
) -> torch.Tensor:
    """Exact head KL plus MC or Hajek tail estimate."""
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
        if tail_proposal_log_probs is None:
            tail_estimate = (1.0 - p_head.detach()) * tail_terms.sum(dim=-1) / k_tail
        elif tail_estimator == "mc":
            weights = (teacher_tail_log_probs - tail_proposal_log_probs).exp()
            tail_estimate = (weights * tail_terms).sum(dim=-1) / k_tail
        else:
            tail_mass = (1.0 - p_head).clamp_min(1e-12)
            log_conditional = teacher_tail_log_probs - tail_mass.unsqueeze(-1).log()
            weights = (log_conditional - tail_proposal_log_probs).exp()
            tail_estimate = tail_mass.detach() * (weights * tail_terms).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-12)
        return head_kl + tail_estimate
    return head_kl


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
        "frankenstein": Legacy clipped self-normalized PPS estimator.
        "hajek": Exact head plus self-normalized teacher-tail Hajek estimate.
        "mc": Exact head plus teacher-tail Monte Carlo estimate.
        "k2": Top-k head plus teacher-tail Schulman K2 penalty.
        "k3": Top-k head plus teacher-tail Schulman K3 estimator.
    reduction : str
        How to reduce: "mean" or "sum".
    k_tail : int
        (head/tail modes only) Number of tail samples. Default 0 = no tail.
    tail_clip : float
        (frankenstein mode only) Maximum importance weight.
    tail_oversample : int
        (frankenstein mode only) Oversample factor for PPS sampling.
    k_head : Optional[int]
        Legacy alias for ``k``.

    Examples
    --------
    Simple top-k (recommended):
    >>> loss_fn = SubsetKLLoss(k=256)

    Proper Hajek estimator:
    >>> loss_fn = SubsetKLLoss(k=128, mode="hajek", k_tail=64)

    Legacy Frankenstein estimator:
    >>> loss_fn = SubsetKLLoss(k=128, mode="frankenstein", k_tail=64)
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
        tail_proposal: TailProposal = "target",
        tail_proposal_alpha: float = 0.8,
        tail_proposal_tau: float = 0.7,
        # Legacy parameter
        k_head: Optional[int] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        valid_modes = {"topk", "frankenstein", "hajek", "mc", "k2", "k3"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown subset KL mode {mode!r}; expected one of {sorted(valid_modes)}")
        self.k = k_head if k_head is not None else k
        self.k_head = self.k  # alias for backward compat
        self.k_tail = k_tail
        self.mode = mode
        self.tail_clip = tail_clip
        self.tail_oversample = tail_oversample
        self.tail_proposal = tail_proposal
        self.tail_proposal_alpha = tail_proposal_alpha
        self.tail_proposal_tau = tail_proposal_tau

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
        if self.mode == "frankenstein" and self.k_tail > 0:
            B, T, V = teacher_logits.shape
            subset = _prepare_hajek_teacher_subset(
                teacher_logits=teacher_logits.view(B * T, V),
                k_head=self.k,
                k_tail=self.k_tail,
                oversample=self.tail_oversample,
            )
            indices = subset["indices"]
            S = indices.shape[-1]
            student_flat = student_logits.view(B * T, V)
            student_selected = torch.gather(student_flat, -1, indices)
            student_logsumexp = student_flat.float().logsumexp(dim=-1, keepdim=True)
            student_log_probs = student_selected.float() - student_logsumexp

            kl_flat, _ = _hajek_kl_from_student_log_probs(
                teacher_log_probs=subset["teacher_log_probs_sel"],
                student_log_probs=student_log_probs.view(B * T, S),
                inclusion_probs=subset["inclusion_probs"],
                mask=subset["mask"],
                weight_clip=self.tail_clip,
            )
            return self._apply_reduction(kl_flat.view(B, T), attention_mask)

        if self.mode in ("hajek", "mc"):
            B, T, V = teacher_logits.shape
            subset = _prepare_k2_teacher_subset(
                teacher_logits=teacher_logits.view(B * T, V),
                k_head=self.k,
                k_tail=self.k_tail,
                tail_proposal=self.tail_proposal,
                tail_proposal_alpha=self.tail_proposal_alpha,
                tail_proposal_tau=self.tail_proposal_tau,
            )
            indices = subset["indices"]
            student_flat = student_logits.view(B * T, V)
            student_selected = torch.gather(student_flat, -1, indices)
            student_logsumexp = student_flat.float().logsumexp(dim=-1, keepdim=True)
            student_log_probs = student_selected.float() - student_logsumexp
            kl = _exact_head_tail_kl_from_student_log_probs(
                teacher_head_log_probs=subset["teacher_head_log_probs"],
                teacher_tail_log_probs=subset["teacher_tail_log_probs"],
                student_log_probs=student_log_probs,
                p_head=subset["p_head"],
                k_head=self.k,
                k_tail=self.k_tail,
                tail_proposal_log_probs=subset["tail_proposal_log_probs"],
                tail_estimator=self.mode,
            )
            return self._apply_reduction(kl.view(B, T), attention_mask)

        if self.mode in ("k2", "k3"):
            B, T, V = teacher_logits.shape
            subset = _prepare_k2_teacher_subset(
                teacher_logits=teacher_logits.view(B * T, V),
                k_head=self.k,
                k_tail=self.k_tail,
            )
            indices = subset["indices"]
            student_flat = student_logits.view(B * T, V)
            student_selected = torch.gather(student_flat, -1, indices)

            if self.k > 0:
                student_head_logprobs = F.log_softmax(
                    student_selected[:, : self.k].float(), dim=-1
                )
                head_kl = (
                    subset["teacher_head_probs_k"]
                    * (subset["teacher_head_logprobs_k"] - student_head_logprobs)
                ).sum(dim=-1)
            else:
                head_kl = torch.zeros(B * T, device=student_logits.device, dtype=torch.float32)

            if self.k_tail > 0:
                student_logsumexp = student_flat.float().logsumexp(dim=-1, keepdim=True)
                student_log_probs = student_selected.float() - student_logsumexp
                teacher_tail_log_probs = subset["teacher_tail_log_probs"]
                student_tail_log_probs = student_log_probs[:, self.k :]
                if self.mode == "k2":
                    tail_penalty = (
                        teacher_tail_log_probs - student_tail_log_probs
                    ).square().sum(dim=-1)
                else:
                    log_ratio = student_tail_log_probs - teacher_tail_log_probs
                    tail_penalty = (torch.expm1(log_ratio) - log_ratio).sum(dim=-1)
                kl = head_kl + (1.0 - subset["p_head"].detach()) * tail_penalty / self.k_tail
            else:
                kl = head_kl

            return self._apply_reduction(kl.view(B, T), attention_mask)

        return self._inner.forward(student_logits, teacher_logits, attention_mask)

    # --- lens-aware memory-efficient path ----

    def prepare_teacher_subset(self, teacher_logits: torch.Tensor) -> dict[str, Any]:
        """Precompute teacher-side subset selection for reuse across layers."""
        if self.mode == "frankenstein" and self.k_tail > 0:
            return self._prepare_teacher_subset_frankenstein(teacher_logits)
        if self.mode in ("hajek", "mc", "k2", "k3"):
            return self._prepare_teacher_subset_k2(teacher_logits)
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
        if teacher_subset is None:
            if teacher_logits is None:
                raise ValueError("teacher_logits is required when teacher_subset is not provided")
            subset = self.prepare_teacher_subset(teacher_logits)
        else:
            subset = teacher_subset
        if subset["mode"] == "frankenstein":
            return self._forward_with_lens_frankenstein(
                hidden_states, lens, layer, attention_mask, subset
            )
        if subset["mode"] in ("hajek", "mc"):
            return self._forward_with_lens_exact_head_tail(
                hidden_states, lens, layer, attention_mask, subset
            )
        if subset["mode"] in ("k2", "k3"):
            return self._forward_with_lens_k2(
                hidden_states, lens, layer, attention_mask, subset
            )
        return self._forward_with_lens_topk(
            hidden_states, lens, layer, attention_mask, subset
        )

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

    def _prepare_teacher_subset_frankenstein(self, teacher_logits: torch.Tensor) -> dict[str, Any]:
        B, T, V = teacher_logits.shape
        teacher_flat = teacher_logits.view(B * T, V)

        subset = _prepare_hajek_teacher_subset(
            teacher_logits=teacher_flat,
            k_head=self.k,
            k_tail=self.k_tail,
            oversample=self.tail_oversample,
        )
        indices = subset["indices"]
        S = indices.shape[-1]
        return {
            "mode": "frankenstein",
            "indices": indices,
            "indices_3d": indices.view(B, T, S),
            "teacher_sel": subset["teacher_log_probs_sel"],
            "inclusion_probs": subset["inclusion_probs"],
            "mask": subset["mask"],
            "diagnostics": subset["diagnostics"],
        }

    def _prepare_teacher_subset_k2(self, teacher_logits: torch.Tensor) -> dict[str, Any]:
        B, T, V = teacher_logits.shape
        teacher_flat = teacher_logits.view(B * T, V)

        subset = _prepare_k2_teacher_subset(
            teacher_logits=teacher_flat,
            k_head=self.k,
            k_tail=self.k_tail,
            tail_proposal=self.tail_proposal,
            tail_proposal_alpha=self.tail_proposal_alpha,
            tail_proposal_tau=self.tail_proposal_tau,
        )
        indices = subset["indices"]
        S = indices.shape[-1]
        return {
            "mode": self.mode,
            "indices": indices,
            "indices_3d": indices.view(B, T, S),
            "teacher_head_log_probs": subset["teacher_head_log_probs"],
            "teacher_head_logprobs_k": subset["teacher_head_logprobs_k"],
            "teacher_head_probs_k": subset["teacher_head_probs_k"],
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
            raise ValueError("teacher_subset is required for top-k forward path")

        # Compute student logits ONLY for top-k indices via lens
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

    def _forward_with_lens_frankenstein(
        self,
        hidden_states: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_subset: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Legacy Frankenstein estimator with memory-efficient lens computation."""
        if teacher_subset is None:
            raise ValueError("teacher_subset is required for Frankenstein forward path")

        B, T, S = teacher_subset["indices_3d"].shape

        # Compute selected student logits and the full-vocab logsumexp.
        raw_lens = lens.module if hasattr(lens, "module") else lens
        student_logits_subset, student_logsumexp = raw_lens.compute_logits_subset_with_logsumexp(
            activations=hidden_states,
            vocab_indices=teacher_subset["indices_3d"],
            layer=layer,
        )
        student_log_probs = student_logits_subset.view(B * T, S).float() - student_logsumexp.view(B * T, 1).float()

        kl_flat, _ = _hajek_kl_from_student_log_probs(
            teacher_log_probs=teacher_subset["teacher_sel"],
            student_log_probs=student_log_probs,
            inclusion_probs=teacher_subset["inclusion_probs"],
            mask=teacher_subset["mask"],
            weight_clip=self.tail_clip,
        )

        kl = kl_flat.view(B, T)
        return self._apply_reduction(kl, attention_mask)

    def _forward_with_lens_exact_head_tail(
        self,
        hidden_states: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_subset: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Exact head KL plus teacher-tail MC/Hajek estimate."""
        if teacher_subset is None:
            raise ValueError("teacher_subset is required for Hajek/MC forward path")

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

        kl_flat = _exact_head_tail_kl_from_student_log_probs(
            teacher_head_log_probs=teacher_subset["teacher_head_log_probs"],
            teacher_tail_log_probs=teacher_subset["teacher_tail_log_probs"],
            student_log_probs=student_log_probs,
            p_head=teacher_subset["p_head"],
            k_head=self.k,
            k_tail=self.k_tail,
            tail_proposal_log_probs=teacher_subset.get("tail_proposal_log_probs"),
            tail_estimator=self.mode,
        )

        return self._apply_reduction(kl_flat.view(B, T), attention_mask)

    def _forward_with_lens_k2(
        self,
        hidden_states: torch.Tensor,
        lens: "BaseLens",
        layer,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_subset: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Top-k head KL plus Schulman K2/K3 penalty on teacher-tail samples."""
        if teacher_subset is None:
            raise ValueError("teacher_subset is required for K2/K3 forward path")

        B, T, S = teacher_subset["indices_3d"].shape
        raw_lens = lens.module if hasattr(lens, "module") else lens

        if self.k_tail > 0:
            student_logits_subset, student_logsumexp = raw_lens.compute_logits_subset_with_logsumexp(
                activations=hidden_states,
                vocab_indices=teacher_subset["indices_3d"],
                layer=layer,
            )
            student_log_probs = (
                student_logits_subset.view(B * T, S).float()
                - student_logsumexp.view(B * T, 1).float()
            )
            student_logits_head = student_logits_subset[..., : self.k]
        else:
            student_logits_subset = lens.forward(
                hidden_states,
                layer=layer,
                vocab_indices=teacher_subset["indices_3d"],
            ).logits
            student_logits_head = student_logits_subset
            student_log_probs = None

        if self.k > 0:
            student_head_logprobs = F.log_softmax(student_logits_head.float(), dim=-1)
            head_kl = (
                teacher_subset["teacher_head_probs_k"].view(B, T, self.k)
                * (
                    teacher_subset["teacher_head_logprobs_k"].view(B, T, self.k)
                    - student_head_logprobs
                )
            ).sum(dim=-1)
        else:
            head_kl = torch.zeros(
                B, T, device=hidden_states.device, dtype=hidden_states.dtype
            )

        if self.k_tail > 0:
            tail_student_log_probs = student_log_probs[:, self.k :]
            teacher_tail_log_probs = teacher_subset["teacher_tail_log_probs"]
            if teacher_subset["mode"] == "k2":
                tail_penalty = (
                    teacher_tail_log_probs - tail_student_log_probs
                ).square().sum(dim=-1)
            else:
                log_ratio = tail_student_log_probs - teacher_tail_log_probs
                tail_penalty = (torch.expm1(log_ratio) - log_ratio).sum(dim=-1)
            tail_penalty = tail_penalty.view(B, T)
            p_head = teacher_subset["p_head"].view(B, T)
            kl = head_kl + (1.0 - p_head.detach()) * tail_penalty / self.k_tail
        else:
            kl = head_kl

        return self._apply_reduction(kl, attention_mask)

    def __repr__(self) -> str:
        return (
            f"SubsetKLLoss(k={self.k}, mode={self.mode!r}, "
            f"reduction={self.reduction!r})"
        )
