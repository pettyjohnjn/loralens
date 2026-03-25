# tests/lenses/test_lenses.py
"""Tests for lenses module."""

import torch
import torch.nn as nn
import pytest

from loralens.lenses import (
    create_lens,
    BaseLens,
    LogitLens,
    TunedLens,
    LoRALens,
)


class TestLogitLens:
    """Tests for LogitLens."""

    def test_shapes(self):
        """Test output shapes."""
        unembed = nn.Linear(8, 100)
        lens = LogitLens(unembed=unembed)
        
        x = torch.randn(2, 4, 8)
        out = lens(x)
        
        assert out.logits.shape == (2, 4, 100)

    def test_no_trainable_params(self):
        """Test logit lens has no trainable params."""
        unembed = nn.Linear(8, 100)
        lens = LogitLens(unembed=unembed)
        
        assert lens.num_trainable_parameters() == 0


class TestTunedLens:
    """Tests for TunedLens."""

    def test_shapes(self):
        """Test output shapes."""
        unembed = nn.Linear(8, 100)
        lens = TunedLens(
            layer_ids=[0, 1, 2],
            hidden_size=8,
            unembed=unembed,
        )
        
        x = torch.randn(2, 4, 8)
        out = lens(x, layer=1)
        
        assert out.logits.shape == (2, 4, 100)

    def test_identity_init(self):
        """Test identity initialization matches logit lens."""
        unembed = nn.Linear(8, 100)
        
        logit = LogitLens(unembed=unembed)
        tuned = TunedLens(
            layer_ids=[0],
            hidden_size=8,
            unembed=unembed,
            init_identity=True,
        )
        
        x = torch.randn(2, 4, 8)
        
        logit_out = logit(x).logits
        tuned_out = tuned(x, layer=0).logits
        
        assert torch.allclose(logit_out, tuned_out, atol=1e-5)

    def test_layer_ids(self):
        """Test layer_ids property."""
        unembed = nn.Linear(8, 100)
        lens = TunedLens(
            layer_ids=[0, 1, 2],
            hidden_size=8,
            unembed=unembed,
        )
        
        assert lens.layer_ids == ["0", "1", "2"]

    def test_missing_layer_raises(self):
        """Test missing layer raises KeyError."""
        unembed = nn.Linear(8, 100)
        lens = TunedLens(
            layer_ids=[0, 1],
            hidden_size=8,
            unembed=unembed,
        )
        
        x = torch.randn(2, 4, 8)
        
        with pytest.raises(KeyError):
            lens(x, layer=99)


class TestLoRALens:
    """Tests for LoRALens."""

    def test_shapes(self):
        """Test output shapes."""
        unembed = nn.Linear(8, 100)
        lens = LoRALens(
            layer_ids=[0, 1, 2],
            hidden_size=8,
            unembed=unembed,
            r=4,
        )
        
        x = torch.randn(2, 4, 8)
        out = lens(x, layer=1)
        
        assert out.logits.shape == (2, 4, 100)

    def test_identity_init_matches_logit(self):
        """Test identity init matches logit lens."""
        unembed = nn.Linear(8, 100)
        
        logit = LogitLens(unembed=unembed)
        lora = LoRALens(
            layer_ids=[0],
            hidden_size=8,
            unembed=unembed,
            r=4,
        )
        
        x = torch.randn(2, 4, 8)
        
        logit_out = logit(x).logits
        lora_out = lora(x, layer=0).logits
        
        assert torch.allclose(logit_out, lora_out, atol=1e-5)

    def test_parameter_efficiency(self):
        """Test LoRA has fewer params than tuned."""
        hidden_size = 64
        unembed = nn.Linear(hidden_size, 1000)
        
        tuned = TunedLens(
            layer_ids=list(range(12)),
            hidden_size=hidden_size,
            unembed=unembed,
        )
        
        lora = LoRALens(
            layer_ids=list(range(12)),
            hidden_size=hidden_size,
            unembed=unembed,
            r=8,
        )
        
        tuned_params = tuned.num_trainable_parameters()
        lora_params = lora.num_trainable_parameters()
        
        # LoRA should have significantly fewer params
        assert lora_params < tuned_params / 2

    def test_gradients_flow(self):
        """Test gradients flow through LoRA."""
        unembed = nn.Linear(8, 100)
        lens = LoRALens(
            layer_ids=[0],
            hidden_size=8,
            unembed=unembed,
            r=4,
        )
        lens.train()
        
        x = torch.randn(2, 4, 8, requires_grad=True)
        out = lens(x, layer=0)
        out.logits.sum().backward()
        
        # Check LoRA params have gradients
        proj = lens.projections["0"]
        assert proj.lora_A.weight.grad is not None
        assert proj.lora_B.weight.grad is not None


class TestFactory:
    """Tests for lens factory."""

    def test_create_logit(self):
        """Test creating logit lens."""
        unembed = nn.Linear(8, 100)
        lens = create_lens("logit", unembed=unembed)
        assert isinstance(lens, LogitLens)

    def test_create_tuned(self):
        """Test creating tuned lens."""
        unembed = nn.Linear(8, 100)
        lens = create_lens(
            "tuned",
            layer_ids=[0, 1],
            hidden_size=8,
            unembed=unembed,
        )
        assert isinstance(lens, TunedLens)

    def test_create_lora(self):
        """Test creating LoRA lens."""
        unembed = nn.Linear(8, 100)
        lens = create_lens(
            "lora",
            layer_ids=[0, 1],
            hidden_size=8,
            unembed=unembed,
            r=4,
        )
        assert isinstance(lens, LoRALens)
        assert lens.r == 4

    def test_unknown_raises(self):
        """Test unknown lens raises."""
        with pytest.raises(ValueError):
            create_lens("unknown_lens")



