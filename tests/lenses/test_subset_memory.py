import torch
import torch.nn as nn
import pytest

from loralens.lenses import LogitLens, TunedLens, LoRALens


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
    weight_subset = linear.weight[vocab_indices]  # [K, H]
    bias_subset = linear.bias[vocab_indices] if linear.bias is not None else None

    flat = activations.reshape(batch_size * seq_len, hidden_size)  # [B*T, H]
    flat_logits_subset = flat @ weight_subset.t()                  # [B*T, K]
    if bias_subset is not None:
        flat_logits_subset = flat_logits_subset + bias_subset

    return flat_logits_subset.reshape(batch_size, seq_len, -1)


def test_subset_forward_uses_fewer_output_elements():
    """
    Deterministic check: subset forward should produce fewer output elements
    than a full-vocab forward. This is a proxy for memory usage.
    """
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size = 4, 128, 64
    full_vocab_size = 50_000
    subset_size = 512

    readout = nn.Linear(hidden_size, full_vocab_size, bias=True)
    lens = LogitLens(readout=readout, vocab_size=full_vocab_size)

    activations = torch.randn(batch_size, seq_len, hidden_size)

    # Full forward
    full_logits = lens(activations, return_logits=True).logits
    assert full_logits is not None

    # Subset forward
    vocab_indices = torch.arange(subset_size, dtype=torch.long)
    subset_logits = _subset_logits_from_linear(activations, readout, vocab_indices)

    # Check shapes
    assert full_logits.shape == (batch_size, seq_len, full_vocab_size)
    assert subset_logits.shape == (batch_size, seq_len, subset_size)

    # Element counts and raw bytes
    full_numel = full_logits.numel()
    subset_numel = subset_logits.numel()
    assert subset_numel < full_numel

    elem_size = full_logits.element_size()
    full_bytes = full_numel * elem_size
    subset_bytes = subset_numel * elem_size

    # Subset should use proportionally less memory than full vocab
    assert subset_bytes * 2 <= full_bytes  # arbitrary but strict: at least 2x smaller


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_subset_forward_uses_less_cuda_memory():
    """
    GPU-only sanity check: peak CUDA memory for subset forward should be
    noticeably lower than for a full forward, all else equal.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    batch_size, seq_len, hidden_size = 8, 256, 128
    full_vocab_size = 100_000
    subset_size = 1_000

    readout = nn.Linear(hidden_size, full_vocab_size, bias=True).to(device)
    lens = LogitLens(readout=readout, vocab_size=full_vocab_size).to(device)

    activations = torch.randn(batch_size, seq_len, hidden_size, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    full_logits = lens(activations, return_logits=True).logits
    assert full_logits is not None
    full_mem = torch.cuda.max_memory_allocated(device)

    # Force free as much as possible between runs
    del full_logits
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    vocab_indices = torch.arange(subset_size, dtype=torch.long, device=device)
    subset_logits = _subset_logits_from_linear(activations, readout, vocab_indices)
    subset_mem = torch.cuda.max_memory_allocated(device)

    # Subset forward should allocate significantly less memory than full forward
    # (tolerance because allocator behavior can vary).
    assert subset_mem < full_mem