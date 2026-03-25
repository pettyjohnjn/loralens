import torch
import torch.nn as nn

from loralens.lenses import LogitLens, LoRALens


def _subset_logits_from_linear(
    activations: torch.Tensor,
    linear: nn.Linear,
    vocab_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute logits for only a subset of vocab indices using the linear layer
    without constructing full-vocab logits.

    activations: [B, T, H]
    vocab_indices: [K]
    returns: [B, T, K]
    """
    batch_size, seq_len, hidden_size = activations.shape
    k = vocab_indices.numel()

    weight_subset = linear.weight[vocab_indices]  # [K, H]
    bias_subset = linear.bias[vocab_indices] if linear.bias is not None else None

    flat = activations.reshape(batch_size * seq_len, hidden_size)  # [B*T, H]
    flat_logits_subset = flat @ weight_subset.t()                  # [B*T, K]
    if bias_subset is not None:
        flat_logits_subset = flat_logits_subset + bias_subset

    return flat_logits_subset.reshape(batch_size, seq_len, k)


def test_lora_lens_shapes_and_identity_matches_logit_lens_initially():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size = 2, 3, 4, 9

    readout = nn.Linear(hidden_size, vocab_size, bias=True)

    logit_lens = LogitLens(readout=readout, vocab_size=vocab_size)

    layer_ids = ["0", "1"]
    lora_lens = LoRALens(
        layer_ids=layer_ids,
        hidden_size=hidden_size,
        readout=readout,
        vocab_size=vocab_size,
        r=4,
        alpha=1.0,
        dropout=0.0,
        freeze_base=True,
        init_identity_base=True,
    )

    activations = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    layer_id = "0"

    lora_out = lora_lens(activations, layer=layer_id, return_logits=True)
    logit_out = logit_lens(activations, return_logits=True)

    assert lora_out.logits is not None
    assert logit_out.logits is not None

    # Shapes
    assert lora_out.logits.shape == (batch_size, seq_len, vocab_size)

    # At initialization, LoRA delta is zero (B is zero), base linear is identity,
    # so LoRALens == LogitLens.
    assert torch.allclose(
        lora_out.logits,
        logit_out.logits,
        atol=1e-6,
        rtol=0.0,
    )

    # Gradient sanity check:
    # - activations must have grad
    # - LoRA params for the used layer ("0") must have grad
    loss = lora_out.logits.sum()
    loss.backward()

    assert activations.grad is not None

    proj_used = lora_lens.projections[str(layer_id)]
    assert proj_used.lora_A.weight.grad is not None
    assert proj_used.lora_B.weight.grad is not None
    # base is frozen (requires_grad=False) when freeze_base=True, so we do not
    # assert anything about its grad here.


def test_lora_lens_vocab_subset_forward_matches_full():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size = 2, 3, 4, 17

    readout = nn.Linear(hidden_size, vocab_size, bias=True)
    layer_ids = [0]

    lora_lens = LoRALens(
        layer_ids=layer_ids,
        hidden_size=hidden_size,
        readout=readout,
        vocab_size=vocab_size,
        r=2,
        alpha=1.0,
        dropout=0.0,
        freeze_base=True,
        init_identity_base=True,
    )

    activations = torch.randn(batch_size, seq_len, hidden_size)
    layer_id = 0

    # Full logits via lens
    full_logits = lora_lens(activations, layer=layer_id, return_logits=True).logits
    assert full_logits is not None
    assert full_logits.shape == (batch_size, seq_len, vocab_size)

    # Compute projected activations for this layer (includes LoRA delta)
    proj = lora_lens.projections[str(layer_id)]
    flat = activations.view(batch_size * seq_len, hidden_size)
    projected_flat = proj(flat)
    projected = projected_flat.view(batch_size, seq_len, hidden_size)

    # Subset of vocab indices
    vocab_indices = torch.tensor([0, 3, 7, 12, 16], dtype=torch.long)

    subset_logits = _subset_logits_from_linear(projected, readout, vocab_indices)
    assert subset_logits.shape == (batch_size, seq_len, vocab_indices.numel())

    # Equality with slicing full logits
    assert torch.allclose(
        full_logits[..., vocab_indices],
        subset_logits,
        atol=1e-6,
        rtol=0.0,
    )