# src/loralens/lenses/_unembed.py
"""
Shared helpers for projecting transformed hidden states through a frozen
unembed module.

The per-layer lenses (TunedLens, LoRALens, BidirLoRALens) all differ only in how
they transform a hidden state before the unembed; the unembed-side machinery —
applying the final norm, locating the weight/bias, and computing a memory-
efficient vocabulary *subset* — is identical. It lives here so each lens reduces
to "apply my transform, then call the shared projector".

An ``unembed`` is expected to expose either ``lm_head`` (an ``nn.Linear``) or a
raw ``weight``/``bias`` (e.g. a bare ``nn.Linear``), and optionally a final norm
as ``layer_norm`` or ``ln_f``. When neither weight form is present, the helpers
fall back to calling ``unembed(flat)`` directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def unembed_weight_bias(
    unembed: nn.Module,
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Return ``(weight [V, d], bias [V] or None)`` for the unembed.

    Returns ``None`` if the module exposes no raw weight matrix, signalling that
    callers should fall back to ``unembed(flat)``.
    """
    if hasattr(unembed, "lm_head"):
        return unembed.lm_head.weight, unembed.lm_head.bias
    if hasattr(unembed, "weight"):
        return unembed.weight, getattr(unembed, "bias", None)
    return None


def unembed_weight_dtype(unembed: nn.Module) -> Optional[torch.dtype]:
    """Dtype expected by the frozen unembed weights, or None if unavailable."""
    wb = unembed_weight_bias(unembed)
    return wb[0].dtype if wb is not None else None


def apply_unembed_norm(unembed: nn.Module, flat: torch.Tensor) -> torch.Tensor:
    """Apply the unembed's final norm (``layer_norm`` or ``ln_f``) if present."""
    norm = getattr(unembed, "layer_norm", None) or getattr(unembed, "ln_f", None)
    return norm(flat) if norm is not None else flat


def subset_logits_from_flat(
    unembed: nn.Module,
    flat: torch.Tensor,
    vocab_indices: torch.Tensor,
    batch: int,
    seq: int,
) -> torch.Tensor:
    """Project transformed activations to a vocabulary *subset*.

    Memory efficient: uses the fused ``indexed_logits`` kernel for per-position
    indices on CUDA, avoiding an [N, V] materialisation.

    Parameters
    ----------
    flat : torch.Tensor
        Already transformed activations, shape [batch * seq, hidden].
    vocab_indices : torch.Tensor
        ``[k]`` shared indices or ``[batch, seq, k]`` per-position indices.

    Returns
    -------
    torch.Tensor
        Logits ``[batch, seq, k]`` for the requested indices.
    """
    flat = apply_unembed_norm(unembed, flat)
    wb = unembed_weight_bias(unembed)
    if wb is None:
        full_logits = unembed(flat).view(batch, seq, -1)
        if vocab_indices.dim() == 1:
            return full_logits[..., vocab_indices]
        return torch.gather(full_logits, -1, vocab_indices)

    W, b = wb
    # The norm commonly upcasts to fp32; the subset kernels require flat and W to
    # share a dtype, so cast back here.
    if flat.dtype != W.dtype:
        flat = flat.to(W.dtype)

    if vocab_indices.dim() == 1:
        k = vocab_indices.shape[0]
        logits_flat = flat @ W[vocab_indices].T  # [B*T, k]
        if b is not None:
            logits_flat = logits_flat + b[vocab_indices]
        return logits_flat.view(batch, seq, k)

    # Per-position indices: use the fused kernel to avoid [N, V].
    from loralens.ops import indexed_logits, indexed_logits_available

    N = batch * seq
    k = vocab_indices.shape[-1]
    idx_flat = vocab_indices.view(N, k).contiguous()

    if indexed_logits_available() and flat.is_cuda:
        logits_flat = indexed_logits(H=flat.contiguous(), W=W.contiguous(), idx=idx_flat, bias=b)
    else:
        full_logits = flat @ W.T  # [N, V] — the materialisation we avoid on CUDA
        if b is not None:
            full_logits = full_logits + b
        logits_flat = torch.gather(full_logits, dim=1, index=idx_flat.long())

    return logits_flat.view(batch, seq, k)


def subset_logits_and_logsumexp_from_flat(
    unembed: nn.Module,
    flat: torch.Tensor,
    vocab_indices: torch.Tensor,
    batch: int,
    seq: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project transformed activations to a subset *and* the full-vocab logsumexp.

    A single ``flat @ W.T`` matmul serves both the logsumexp and the subset
    gather — one large CUBLAS call beats a chunked logsumexp loop. The MC subset
    KL estimator needs the true normalized student log-probs this provides.

    Returns ``(subset_logits [batch, seq, k], logsumexp [batch, seq, 1])``.
    """
    flat = apply_unembed_norm(unembed, flat)
    wb = unembed_weight_bias(unembed)
    if wb is None:
        full_logits = unembed(flat).view(batch, seq, -1)
        logsumexp = full_logits.float().logsumexp(dim=-1, keepdim=True)
        if vocab_indices.dim() == 1:
            subset_logits = full_logits[..., vocab_indices]
        else:
            subset_logits = torch.gather(full_logits, -1, vocab_indices)
        return subset_logits, logsumexp

    W, b = wb
    if flat.dtype != W.dtype:
        flat = flat.to(W.dtype)

    N = batch * seq
    full_logits = flat @ W.T  # [N, V]
    if b is not None:
        full_logits = full_logits + b

    logsumexp = full_logits.float().logsumexp(dim=-1, keepdim=True)  # [N, 1]

    if vocab_indices.dim() == 1:
        k = vocab_indices.shape[0]
        subset_logits = full_logits[:, vocab_indices].view(batch, seq, k)
    else:
        k = vocab_indices.shape[-1]
        idx_flat = vocab_indices.view(N, k).contiguous()
        subset_logits = torch.gather(full_logits, dim=1, index=idx_flat.long()).view(batch, seq, k)

    return subset_logits, logsumexp.view(batch, seq, 1)
