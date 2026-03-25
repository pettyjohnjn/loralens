# tests/losses/test_losses.py
"""Tests for loss functions (using external subset-kl and local adapters)."""

import torch
import torch.nn.functional as F
import pytest

from loralens.losses import (
    create_loss,
    BaseLoss,
    KLDivergenceLoss,
    SubsetKLLoss,
    CrossEntropyLoss,
)


class TestKLDivergenceLoss:
    """Tests for KL divergence loss (delegated to subset-kl)."""

    def test_basic_computation(self):
        loss_fn = KLDivergenceLoss()

        batch, seq, vocab = 2, 4, 10
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        mask = torch.ones(batch, seq)

        loss = loss_fn(student, teacher, mask)

        assert loss.ndim == 0
        assert loss >= 0

    def test_zero_when_equal(self):
        loss_fn = KLDivergenceLoss()

        logits = torch.randn(2, 4, 10)
        mask = torch.ones(2, 4)

        loss = loss_fn(logits, logits, mask)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_masking(self):
        loss_fn = KLDivergenceLoss(reduction="none")

        student = torch.randn(2, 4, 10)
        teacher = torch.randn(2, 4, 10)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float)

        loss = loss_fn(student, teacher, mask)

        # KL loss with reduction="none" won't apply mask itself,
        # but the external impl should still return per-token values
        assert loss.shape == (2, 4)

    def test_chunked_matches_full(self):
        batch, seq, vocab = 2, 8, 10
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        mask = torch.ones(batch, seq)

        loss_full = KLDivergenceLoss()(student, teacher, mask)
        loss_chunked = KLDivergenceLoss(chunk_size=4)(student, teacher, mask)

        assert torch.allclose(loss_full, loss_chunked, atol=1e-5)

    def test_labels_param_ignored(self):
        """Test that labels parameter is accepted but ignored."""
        loss_fn = KLDivergenceLoss()
        batch, seq, vocab = 2, 4, 10
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        mask = torch.ones(batch, seq)
        labels = torch.randint(0, vocab, (batch, seq))

        loss_with = loss_fn(student, teacher, mask, labels=labels)
        loss_without = loss_fn(student, teacher, mask)

        assert torch.allclose(loss_with, loss_without)


class TestSubsetKLLoss:
    """Tests for subset KL loss (delegated to subset-kl)."""

    def test_basic_computation(self):
        loss_fn = SubsetKLLoss(k=10)

        batch, seq, vocab = 2, 4, 100
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        mask = torch.ones(batch, seq)

        loss = loss_fn(student, teacher, mask)

        assert loss.ndim == 0
        assert loss > -0.1

    def test_approximates_full_kl(self):
        batch, seq, vocab = 4, 8, 50
        student = torch.randn(batch, seq, vocab) * 2
        teacher = torch.randn(batch, seq, vocab) * 2
        mask = torch.ones(batch, seq)

        full_loss = KLDivergenceLoss()(student, teacher, mask)
        subset_loss = SubsetKLLoss(k=40)(student, teacher, mask)

        ratio = subset_loss / full_loss
        assert 0.3 < ratio < 3.0

    def test_legacy_k_head_param(self):
        """Test backward compat with k_head parameter."""
        loss_fn = SubsetKLLoss(k_head=50)
        assert loss_fn.k == 50


class TestCrossEntropyLoss:
    """Tests for cross-entropy loss (local, not from subset-kl)."""

    def test_basic_computation(self):
        loss_fn = CrossEntropyLoss()

        batch, seq, vocab = 2, 4, 10
        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        mask = torch.ones(batch, seq)

        loss = loss_fn(logits, logits, mask, labels=labels)

        assert loss.ndim == 0
        assert loss > 0

    def test_matches_pytorch(self):
        batch, seq, vocab = 2, 4, 10
        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        our_loss = CrossEntropyLoss()(logits, logits, None, labels=labels)
        pt_loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))

        assert torch.allclose(our_loss, pt_loss, atol=1e-5)


class TestFactory:
    """Tests for loss factory."""

    def test_create_kl(self):
        loss = create_loss("kl")
        assert isinstance(loss, KLDivergenceLoss)

    def test_create_subset_kl(self):
        loss = create_loss("subset_kl", k=50)
        assert isinstance(loss, SubsetKLLoss)
        assert loss.k == 50

    def test_create_ce(self):
        loss = create_loss("ce")
        assert isinstance(loss, CrossEntropyLoss)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_loss("unknown_loss")


class TestHooksReExport:
    """Test that hookbox re-exports work through loralens.hooks."""

    def test_imports(self):
        from loralens.hooks import (
            ActivationCollector,
            HookManager,
            ActivationHook,
            BaseHook,
        )
        # Just verify imports succeed
        assert ActivationCollector is not None
        assert HookManager is not None


class TestOpsWrapper:
    """Test that ops wrapper loads correctly."""

    def test_availability_check(self):
        from loralens.ops import indexed_logits_available
        # Should return bool (True if CUDA ext installed, False otherwise)
        result = indexed_logits_available()
        assert isinstance(result, bool)
