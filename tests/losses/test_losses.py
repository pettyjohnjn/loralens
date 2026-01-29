# tests/losses/test_losses.py
"""Tests for loss functions."""

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
    """Tests for KL divergence loss."""

    def test_basic_computation(self):
        """Test basic KL computation."""
        loss_fn = KLDivergenceLoss()
        
        batch, seq, vocab = 2, 4, 10
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        mask = torch.ones(batch, seq)
        
        loss = loss_fn(student, teacher, mask)
        
        assert loss.ndim == 0
        assert loss >= 0

    def test_zero_when_equal(self):
        """Test KL is zero when distributions match."""
        loss_fn = KLDivergenceLoss()
        
        logits = torch.randn(2, 4, 10)
        mask = torch.ones(2, 4)
        
        loss = loss_fn(logits, logits, mask)
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_masking(self):
        """Test that mask is respected."""
        loss_fn = KLDivergenceLoss(reduction="none")
        
        student = torch.randn(2, 4, 10)
        teacher = torch.randn(2, 4, 10)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float)
        
        loss = loss_fn(student, teacher, mask)
        
        # Masked positions should have zero loss
        assert (loss[:, 2:] == 0).all()

    def test_chunked_matches_full(self):
        """Test chunked computation matches full."""
        batch, seq, vocab = 2, 8, 10
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        mask = torch.ones(batch, seq)
        
        loss_full = KLDivergenceLoss()(student, teacher, mask)
        loss_chunked = KLDivergenceLoss(chunk_size=4)(student, teacher, mask)
        
        assert torch.allclose(loss_full, loss_chunked, atol=1e-5)


class TestSubsetKLLoss:
    """Tests for subset KL loss."""

    def test_basic_computation(self):
        """Test subset KL computes without error."""
        loss_fn = SubsetKLLoss(k_head=10, k_tail=5)
        
        batch, seq, vocab = 2, 4, 100
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        mask = torch.ones(batch, seq)
        
        loss = loss_fn(student, teacher, mask)
        
        assert loss.ndim == 0
        # Should be approximately non-negative (small numerical errors ok)
        assert loss > -0.1

    def test_approximates_full_kl(self):
        """Test subset KL approximates full KL."""
        batch, seq, vocab = 4, 8, 50
        student = torch.randn(batch, seq, vocab) * 2
        teacher = torch.randn(batch, seq, vocab) * 2
        mask = torch.ones(batch, seq)
        
        full_loss = KLDivergenceLoss()(student, teacher, mask)
        subset_loss = SubsetKLLoss(k_head=40, k_tail=10)(student, teacher, mask)
        
        # Should be in same ballpark
        ratio = subset_loss / full_loss
        assert 0.3 < ratio < 3.0


class TestCrossEntropyLoss:
    """Tests for cross-entropy loss."""

    def test_basic_computation(self):
        """Test CE computation."""
        loss_fn = CrossEntropyLoss()
        
        batch, seq, vocab = 2, 4, 10
        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        mask = torch.ones(batch, seq)
        
        # CE uses labels, teacher_logits unused
        loss = loss_fn(logits, logits, mask, labels=labels)
        
        assert loss.ndim == 0
        assert loss > 0

    def test_matches_pytorch(self):
        """Test matches PyTorch CE."""
        batch, seq, vocab = 2, 4, 10
        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        
        our_loss = CrossEntropyLoss()(logits, logits, None, labels=labels)
        pt_loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
        
        assert torch.allclose(our_loss, pt_loss, atol=1e-5)


class TestFactory:
    """Tests for loss factory."""

    def test_create_kl(self):
        """Test creating KL loss."""
        loss = create_loss("kl")
        assert isinstance(loss, KLDivergenceLoss)

    def test_create_subset_kl(self):
        """Test creating subset KL loss."""
        loss = create_loss("subset_kl", k_head=50, k_tail=25)
        assert isinstance(loss, SubsetKLLoss)
        assert loss.k_head == 50
        assert loss.k_tail == 25

    def test_create_ce(self):
        """Test creating CE loss."""
        loss = create_loss("ce")
        assert isinstance(loss, CrossEntropyLoss)

    def test_unknown_raises(self):
        """Test unknown loss raises."""
        with pytest.raises(ValueError):
            create_loss("unknown_loss")
