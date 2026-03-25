import torch
import torch.nn as nn

from loralens.lenses import LogitLens, TunedLens


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


def test_tuned_lens_shapes_and_identity_matches_logit_lens():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size = 2, 3, 4, 7

    readout = nn.Linear(hidden_size, vocab_size, bias=True)

    # Logit lens baseline
    logit_lens = LogitLens(readout=readout, vocab_size=vocab_size)

    # Tuned lens with identity-initialized per-layer projections
    layer_ids = ["0", "1", 2]
    tuned_lens = TunedLens(
        layer_ids=layer_ids,
        hidden_size=hidden_size,
        readout=readout,
        vocab_size=vocab_size,
        init_identity=True,
    )

    activations = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    layer_id = "1"

    tuned_out = tuned_lens(activations, layer=layer_id, return_logits=True)
    logit_out = logit_lens(activations, return_logits=True)

    assert tuned_out.logits is not None
    assert logit_out.logits is not None

    # Shapes
    assert tuned_out.logits.shape == (batch_size, seq_len, vocab_size)

    # Because projections are identity, tuned lens == logit lens at init
    assert torch.allclose(
        tuned_out.logits,
        logit_out.logits,
        atol=1e-6,
        rtol=0.0,
    )

    # Gradient flow sanity check:
    # - activations must have grad
    # - projection for the used layer must have grad
    loss = tuned_out.logits.sum()
    loss.backward()

    assert activations.grad is not None

    proj_used = tuned_lens.projections[str(layer_id)]
    assert proj_used.weight.grad is not None
    if proj_used.bias is not None and proj_used.bias.requires_grad:
        assert proj_used.bias.grad is not None


def test_tuned_lens_vocab_subset_forward_matches_full():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size = 2, 3, 4, 13

    readout = nn.Linear(hidden_size, vocab_size, bias=True)
    layer_ids = [0, 1]

    tuned_lens = TunedLens(
        layer_ids=layer_ids,
        hidden_size=hidden_size,
        readout=readout,
        vocab_size=vocab_size,
        init_identity=True,
    )

    activations = torch.randn(batch_size, seq_len, hidden_size)
    layer_id = 0

    # Full logits via lens
    full_logits = tuned_lens(activations, layer=layer_id, return_logits=True).logits
    assert full_logits is not None
    assert full_logits.shape == (batch_size, seq_len, vocab_size)

    # Compute projected activations only once for this layer
    proj = tuned_lens.projections[str(layer_id)]
    flat = activations.view(batch_size * seq_len, hidden_size)
    projected_flat = proj(flat)
    projected = projected_flat.view(batch_size, seq_len, hidden_size)

    # Select vocab subset
    vocab_indices = torch.tensor([1, 4, 8, 12], dtype=torch.long)

    subset_logits = _subset_logits_from_linear(projected, readout, vocab_indices)
    assert subset_logits.shape == (batch_size, seq_len, vocab_indices.numel())

    # Equality with slicing full logits
    assert torch.allclose(
        full_logits[..., vocab_indices],
        subset_logits,
        atol=1e-6,
        rtol=0.0,
    )