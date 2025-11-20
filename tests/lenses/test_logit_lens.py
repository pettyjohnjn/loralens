import torch
import torch.nn as nn

from loralens.lenses import LogitLens


def test_logit_lens_shapes_and_grad():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size = 2, 3, 4, 5

    readout = nn.Linear(hidden_size, vocab_size, bias=True)
    lens = LogitLens(readout=readout, vocab_size=vocab_size)

    activations = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)

    out = lens(activations, return_logits=True, return_loss=False)
    assert out.logits is not None

    logits = out.logits
    assert logits.shape == (batch_size, seq_len, vocab_size)

    loss = logits.sum()
    loss.backward()

    # Gradients should flow to activations and readout params
    assert activations.grad is not None
    assert readout.weight.grad is not None
    assert readout.bias.grad is not None


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


def test_logit_lens_vocab_subset_forward_matches_full():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size = 2, 3, 4, 11

    readout = nn.Linear(hidden_size, vocab_size, bias=True)
    lens = LogitLens(readout=readout, vocab_size=vocab_size)

    activations = torch.randn(batch_size, seq_len, hidden_size)

    # Full logits via the lens
    full_logits = lens(activations, return_logits=True).logits
    assert full_logits is not None

    # Choose a subset of vocab indices
    vocab_indices = torch.tensor([0, 2, 5, 7, 10], dtype=torch.long)

    # Efficient subset computation (no full [V] compute required in principle)
    subset_logits = _subset_logits_from_linear(activations, readout, vocab_indices)

    # Check shapes
    assert subset_logits.shape == (batch_size, seq_len, vocab_indices.numel())

    # Check numerical equality with sliced full logits
    assert torch.allclose(
        full_logits[..., vocab_indices],
        subset_logits,
        atol=1e-6,
        rtol=0.0,
    )