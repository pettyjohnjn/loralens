import torch
import torch.nn as nn

from loralens.hooks.base_hook import BaseHook


class DummyHook(BaseHook):
    def register(self, module: nn.Module) -> None:
        def fn(mod, inp, out):
            return out
        self._handle = module.register_forward_hook(fn)
        self.module = module

    def __repr__(self) -> str:
        return f"DummyHook(name={self.name})"


def test_base_hook_unregister_clears_handle_and_module():
    model = nn.Linear(4, 4)
    hook = DummyHook(name="dummy")
    hook.register(model)

    assert hook._handle is not None
    assert hook.module is model

    hook.unregister()

    assert hook._handle is None
    assert hook.module is None