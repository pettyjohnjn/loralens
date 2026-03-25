import pytest
import torch
import torch.nn as nn

from loralens.hooks.hook_manager import HookManager
from loralens.hooks.activation_hook import ActivationHook


class ToyTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


class ToyModel(nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([ToyTransformerBlock() for _ in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_add_hook_by_module_name_and_remove():
    model = ToyModel()
    manager = HookManager(model)

    seen = {"count": 0}

    def on_activation(x, module_name, module):
        seen["count"] += 1

    hook = ActivationHook(
        name="hook:blocks.0.attn",
        on_activation=on_activation,
        module_name="blocks.0.attn",
    )

    manager.add_hook("blocks.0.attn", hook)
    assert "hook:blocks.0.attn" in manager.list_hooks()

    x = torch.randn(1, 8)
    _ = model(x)
    assert seen["count"] == 1

    manager.remove_hook("hook:blocks.0.attn")
    assert "hook:blocks.0.attn" not in manager.list_hooks()

    _ = model(x)
    assert seen["count"] == 1


def test_add_activation_hooks_by_predicate():
    model = ToyModel(num_layers=3)
    manager = HookManager(model)

    def predicate(name, module):
        return name.endswith(".mlp")

    seen = {"names": []}

    def on_activation(x, module_name, module):
        seen["names"].append(module_name)

    manager.add_activation_hooks(
        predicate,
        on_activation=on_activation,
        name_prefix="mlp_hook",
    )

    x = torch.randn(1, 8)
    _ = model(x)

    assert set(seen["names"]) == {
        "blocks.0.mlp",
        "blocks.1.mlp",
        "blocks.2.mlp",
    }

    assert len(manager.list_hooks()) == 3

    manager.remove_all()
    assert manager.list_hooks() == []


def test_hook_name_collision_raises():
    model = ToyModel()
    manager = HookManager(model)

    def predicate(name, module):
        return name == "blocks.0.attn"

    def transform_fn(x, module_name, module):
        return x

    manager.add_activation_hooks(
        predicate,
        transform_fn=transform_fn,
        name_prefix="collision",
    )

    with pytest.raises(ValueError):
        manager.add_activation_hooks(
            predicate,
            transform_fn=transform_fn,
            name_prefix="collision",
        )