import torch
import torch.nn as nn

from loralens.hooks.activation_hook import ActivationHook


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def test_activation_hook_transform_fn_applies():
    model = TinyModel()
    called = {"transform": False}
    scale = 2.0

    def transform_fn(x, module_name, module):
        called["transform"] = True
        return x * scale

    hook = ActivationHook(
        name="test_hook",
        transform_fn=transform_fn,
        module_name="linear",
    )
    hook.register(model.linear)

    x = torch.ones(1, 4)
    out = model(x)

    hook.unregister()
    out_no_hook = model(x)

    assert called["transform"] is True
    assert torch.allclose(out, out_no_hook * scale, atol=1e-5)


def test_activation_hook_on_activation_called():
    model = TinyModel()
    seen = {"count": 0, "module_names": []}

    def on_activation(x, module_name, module):
        seen["count"] += 1
        seen["module_names"].append(module_name)

    hook = ActivationHook(
        name="callback_hook",
        on_activation=on_activation,
        module_name="linear",
    )
    hook.register(model.linear)

    x = torch.randn(2, 4)
    _ = model(x)

    assert seen["count"] == 1
    assert seen["module_names"] == ["linear"]


def test_activation_hook_handles_non_tensor_outputs():
    class TupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            y = self.linear(x)
            return x, y  # non-tensor (tuple) output

    model = TupleModel()
    seen = {"count": 0}

    def on_activation(outputs, module_name, module):
        seen["count"] += 1
        # At this hook location, we expect the final model output (a tuple)
        assert isinstance(outputs, tuple)

    hook = ActivationHook(
        name="tuple_hook",
        on_activation=on_activation,
        module_name="tuple_model",
    )
    # IMPORTANT CHANGE: hook the whole model, not model.linear
    hook.register(model)

    x = torch.randn(2, 4)
    _ = model(x)

    assert seen["count"] == 1