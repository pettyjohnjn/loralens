"""Safety-net tests guarding the cleanup refactor.

These cover the code paths the refactor touches but that were previously
untested: the LoRA subset-logit computation, the memory-efficient
``forward_with_lens`` path the trainer actually uses (topk + mc), and the
bidirectional ortho penalty. Pure-tensor + tiny nn.Linear unembed, so they
run fast on CPU with no model download.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from loralens.lenses import LoRALens
from loralens.lenses.bidir_lora_lens import BidirLoRALens
from loralens.losses import SubsetKLLoss


def _make_lens(cls=LoRALens, hidden=8, vocab=64, layers=(0, 1, 2), r=4, seed=0):
    torch.manual_seed(seed)
    unembed = nn.Linear(hidden, vocab)
    return cls(layer_ids=list(layers), hidden_size=hidden, unembed=unembed, r=r), unembed


class TestLoRASubsetLogits:
    """compute_logits_subset(_with_logsumexp) must match the dense reference."""

    def test_shared_indices_match_dense(self):
        lens, _ = _make_lens()
        # Perturb the LoRA B so the projection is non-trivial (not identity).
        with torch.no_grad():
            for proj in lens.projections.values():
                proj.lora_B.weight.normal_(std=0.1)

        h = torch.randn(2, 5, 8)
        idx = torch.tensor([1, 7, 13, 30, 63])

        dense = lens.compute_logits(h, layer=1)            # [B, T, V]
        subset = lens.compute_logits_subset(h, idx, layer=1)
        torch.testing.assert_close(subset, dense[..., idx], atol=1e-4, rtol=1e-4)

    def test_per_position_indices_match_dense(self):
        lens, _ = _make_lens()
        h = torch.randn(2, 5, 8)
        idx = torch.randint(0, 64, (2, 5, 6))

        dense = lens.compute_logits(h, layer=0)
        subset = lens.compute_logits_subset(h, idx, layer=0)
        torch.testing.assert_close(
            subset, torch.gather(dense, -1, idx), atol=1e-4, rtol=1e-4
        )

    def test_logsumexp_path_matches_dense(self):
        lens, _ = _make_lens()
        h = torch.randn(2, 5, 8)
        idx = torch.randint(0, 64, (2, 5, 6))

        dense = lens.compute_logits(h, layer=2)
        subset, lse = lens.compute_logits_subset_with_logsumexp(h, idx, layer=2)

        torch.testing.assert_close(
            subset, torch.gather(dense, -1, idx), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            lse, dense.float().logsumexp(dim=-1, keepdim=True), atol=1e-4, rtol=1e-4
        )


class TestForwardWithLens:
    """The memory-efficient path used by the trainer for subset KL."""

    def test_topk_matches_manual(self):
        lens, _ = _make_lens()
        loss_fn = SubsetKLLoss(k=8, mode="topk", reduction="mean")

        h = torch.randn(2, 5, 8)
        teacher = torch.randn(2, 5, 64)
        mask = torch.ones(2, 5)

        loss = loss_fn.forward_with_lens(
            hidden_states=h, teacher_logits=teacher, lens=lens, layer=1, attention_mask=mask
        )
        assert loss.ndim == 0 and torch.isfinite(loss) and loss >= 0

        # Manual top-k renormalized KL reference.
        top_vals, top_idx = teacher.topk(8, dim=-1)
        t_logp = F.log_softmax(top_vals, dim=-1)
        student_k = torch.gather(lens.compute_logits(h, layer=1), -1, top_idx)
        s_logp = F.log_softmax(student_k, dim=-1)
        ref = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
        torch.testing.assert_close(loss, ref, atol=1e-4, rtol=1e-4)

    def test_mc_runs_and_finite(self):
        lens, _ = _make_lens()
        loss_fn = SubsetKLLoss(k=8, mode="mc", k_tail=4, reduction="mean")

        h = torch.randn(2, 5, 8)
        teacher = torch.randn(2, 5, 64)
        mask = torch.ones(2, 5)

        subset = loss_fn.prepare_teacher_subset(teacher)
        loss = loss_fn.forward_with_lens(
            hidden_states=h, teacher_logits=None, lens=lens, layer=0,
            attention_mask=mask, teacher_subset=subset,
        )
        assert loss.ndim == 0 and torch.isfinite(loss)


class TestBidirOrtho:
    """BidirLoRALens read path + orthogonality penalty."""

    def test_read_path_inherited(self):
        bidir, _ = _make_lens(cls=BidirLoRALens)
        h = torch.randn(2, 5, 8)
        out = bidir(h, layer=1)
        assert out.logits.shape == (2, 5, 64)

    def test_ortho_zero_at_init(self):
        # At default init B=0 => M.T = I => penalty == 0.
        bidir, _ = _make_lens(cls=BidirLoRALens)
        h = torch.randn(2, 5, 8)
        pen = bidir.compute_orthogonality_penalty(h, layer=1)
        assert pen.ndim == 0
        assert pen.item() == pytest.approx(0.0, abs=1e-6)

    def test_ortho_nonneg_when_perturbed(self):
        bidir, _ = _make_lens(cls=BidirLoRALens)
        with torch.no_grad():
            for proj in bidir.projections.values():
                proj.lora_B.weight.normal_(std=0.5)
        h = torch.randn(2, 5, 8)
        pen = bidir.compute_orthogonality_penalty(h, layer=1)
        assert pen.item() >= 0.0
