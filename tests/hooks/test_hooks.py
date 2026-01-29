# tests/hooks/test_hooks.py
"""Tests for hooks module."""

import torch
import torch.nn as nn
import pytest

from loralens.hooks import (
    ActivationHook,
    HookManager,
    ActivationCollector,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


class TestActivationHook:
    """Tests for ActivationHook."""

    def test_callback_called(self):
        """Test callback is invoked."""
        model = SimpleModel()
        captured = []
        
        hook = ActivationHook(
            name="test",
            on_activation=lambda x, n, m: captured.append(x.clone()),
        )
        hook.register(model.layer1)
        
        x = torch.randn(2, 4)
        _ = model(x)
        
        assert len(captured) == 1
        assert captured[0].shape == (2, 4)
        
        hook.unregister()

    def test_transform_applied(self):
        """Test transform function is applied."""
        model = SimpleModel()
        
        hook = ActivationHook(
            name="scale",
            transform_fn=lambda x, n, m: x * 0.0,  # Zero out
        )
        hook.register(model.layer1)
        
        x = torch.randn(2, 4)
        out_with_hook = model(x)
        
        hook.unregister()
        
        # Output should be different (zeroed intermediate)
        out_no_hook = model(x)
        assert not torch.allclose(out_with_hook, out_no_hook)

    def test_unregister(self):
        """Test hook can be unregistered."""
        model = SimpleModel()
        call_count = [0]
        
        hook = ActivationHook(
            name="counter",
            on_activation=lambda x, n, m: call_count.__setitem__(0, call_count[0] + 1),
        )
        hook.register(model.layer1)
        
        _ = model(torch.randn(2, 4))
        assert call_count[0] == 1
        
        hook.unregister()
        
        _ = model(torch.randn(2, 4))
        assert call_count[0] == 1  # No additional calls


class TestHookManager:
    """Tests for HookManager."""

    def test_add_hook(self):
        """Test adding hook by name."""
        model = SimpleModel()
        manager = HookManager(model)
        
        hook = ActivationHook(name="test")
        manager.add_hook("layer1", hook)
        
        assert "test" in manager.list_hooks()
        assert len(manager) == 1
        
        manager.remove_all()

    def test_add_by_predicate(self):
        """Test adding hooks by predicate."""
        model = SimpleModel()
        manager = HookManager(model)
        
        count = manager.add_activation_hooks(
            predicate=lambda name, mod: isinstance(mod, nn.Linear),
            name_prefix="linear",
        )
        
        assert count == 2
        assert len(manager) == 2
        
        manager.remove_all()

    def test_remove_hook(self):
        """Test removing specific hook."""
        model = SimpleModel()
        manager = HookManager(model)
        
        hook = ActivationHook(name="removeme")
        manager.add_hook("layer1", hook)
        
        assert manager.remove_hook("removeme")
        assert "removeme" not in manager.list_hooks()

    def test_remove_all(self):
        """Test removing all hooks."""
        model = SimpleModel()
        manager = HookManager(model)
        
        manager.add_activation_hooks(
            predicate=lambda n, m: True,
        )
        
        removed = manager.remove_all()
        assert removed > 0
        assert len(manager) == 0


class TestActivationCollector:
    """Tests for ActivationCollector."""

    def test_context_manager(self):
        """Test as context manager."""
        model = SimpleModel()
        collector = ActivationCollector(model)
        
        with collector:
            assert collector._attached
        
        assert not collector._attached

    def test_manual_attach_detach(self):
        """Test manual attach/detach."""
        model = SimpleModel()
        collector = ActivationCollector(model)
        
        collector.attach()
        assert collector._attached
        
        collector.detach()
        assert not collector._attached
