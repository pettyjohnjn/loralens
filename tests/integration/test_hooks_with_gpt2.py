import torch
import torch.nn as nn
import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer

from loralens.hooks.hook_manager import HookManager


@pytest.fixture(scope="module")
def gpt2_model_and_tokenizer():
    # tiny model keeps tests fast
    model_name = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


@pytest.mark.slow
def test_mlp_hooks_hit_each_block(gpt2_model_and_tokenizer):
    model, tokenizer = gpt2_model_and_tokenizer
    manager = HookManager(model)

    # For GPT-2, the MLP output projection is transformer.h[i].mlp.c_proj
    def predicate(name: str, module: nn.Module) -> bool:
        return name.endswith(".mlp.c_proj")

    seen = {"names": [], "shapes": []}

    def on_activation(x, module_name: str, module: nn.Module):
        seen["names"].append(module_name)
        seen["shapes"].append(tuple(x.shape))

    manager.add_activation_hooks(
        predicate,
        on_activation=on_activation,
        name_prefix="mlp_lens",
    )

    text = "Hello world"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    with torch.no_grad():
        _ = model(input_ids)

    n_layer = model.config.n_layer

    # one activation per block.mlp.c_proj
    assert len(seen["names"]) == n_layer
    assert len(manager.list_hooks()) == n_layer

    # names should correspond to transformer.h.{i}.mlp.c_proj
    for i in range(n_layer):
        expected = f"transformer.h.{i}.mlp.c_proj"
        assert expected in seen["names"][i]

    # shapes must be [batch, seq, hidden]
    batch, seq = input_ids.shape
    hidden = model.config.n_embd
    for shape in seen["shapes"]:
        assert shape == (batch, seq, hidden)

    manager.remove_all()
    assert manager.list_hooks() == []


@pytest.mark.slow
def test_attn_output_hook_matches_manual_forward(gpt2_model_and_tokenizer):
    model, tokenizer = gpt2_model_and_tokenizer
    model.eval()

    manager = HookManager(model)

    captured = {"hook_out": None}

    def on_activation(x, module_name: str, module: nn.Module):
        # capture the tensor the hook sees (first block attn.c_proj)
        captured["hook_out"] = x.detach().clone()

    def predicate(name: str, module: nn.Module) -> bool:
        # hook exactly the first block's attention output projection
        return name == "transformer.h.0.attn.c_proj"

    manager.add_activation_hooks(
        predicate,
        on_activation=on_activation,
        name_prefix="attn_out",
    )

    text = "Hello world"
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    with torch.no_grad():
        _ = model(input_ids)

    assert captured["hook_out"] is not None
    hook_out = captured["hook_out"]

    # Manually recompute the same tensor by forwarding through the embedding
    # and the first block, with a one-off hook on attn.c_proj
    block0 = model.transformer.h[0]
    wte = model.transformer.wte
    wpe = model.transformer.wpe

    # standard GPT-2 positional embedding usage
    pos = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    hidden_states = wte(input_ids) + wpe(pos)

    manual_captured = {}

    def manual_hook(mod, inp, out):
        manual_captured["out"] = out.detach().clone()
        return out

    handle = block0.attn.c_proj.register_forward_hook(manual_hook)

    with torch.no_grad():
        _ = block0(hidden_states)

    handle.remove()

    manual_out = manual_captured["out"]

    assert manual_out.shape == hook_out.shape
    assert torch.allclose(hook_out, manual_out, atol=1e-5)

    manager.remove_all()
    assert manager.list_hooks() == []