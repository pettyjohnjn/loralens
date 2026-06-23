# src/loralens/ops/indexed_logits.py
"""
Wrapper for the indexed_logits CUDA extension.

This provides a memory-efficient way to compute::

    out[i, j] = dot(H[i, :], W[idx[i, j], :]) + bias[idx[i, j]]

without materializing [N, k, d] or [N, V] intermediate tensors.

The extension is optional - if it is not available (or the inputs are not on
CUDA), we fall back to a dense matmul + gather, which is correct but not
memory-efficient.
"""

from __future__ import annotations

from typing import Optional
import logging

import torch

logger = logging.getLogger(__name__)

# Try to import the CUDA extension.
_INDEXED_LOGITS_AVAILABLE = False
_indexed_logits_fn = None

try:
    from indexed_logits import indexed_logits as _indexed_logits_fn
    _INDEXED_LOGITS_AVAILABLE = True
    logger.info("indexed_logits CUDA extension loaded successfully")
except ImportError:
    logger.warning(
        "indexed_logits CUDA extension not available. "
        "Falling back to dense matmul + gather. "
        "Install with: pip install indexed_logits"
    )


def indexed_logits_available() -> bool:
    """Check if the indexed_logits CUDA extension is available."""
    return _INDEXED_LOGITS_AVAILABLE


def _gather_bias(bias: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather bias values for the selected indices: bias[idx] -> [N, k]."""
    return torch.gather(bias.unsqueeze(0).expand(idx.shape[0], -1), dim=1, index=idx.long())


def indexed_logits(
    H: torch.Tensor,
    W: torch.Tensor,
    idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute indexed logits: ``out[i, j] = dot(H[i, :], W[idx[i, j], :]) + bias[idx[i, j]]``.

    Memory-efficient when the fused CUDA kernel is available: avoids
    materializing [N, k, d] or [N, V]. Falls back to a dense matmul + gather
    otherwise.

    Args:
        H: Hidden states [N, d], float16/bfloat16, CUDA (for the fused path).
        W: Weight matrix [V, d], same dtype as H.
        idx: Token indices [N, k]. Cast to int32 for the fused kernel.
        bias: Optional bias [V], same dtype as W.

    Returns:
        Logits [N, k], same dtype as H.
    """
    H = H.contiguous()
    W = W.contiguous()
    idx = idx.contiguous()

    if H.dtype != W.dtype:
        raise ValueError(f"H and W must have same dtype: got {H.dtype} vs {W.dtype}")

    use_fused = _INDEXED_LOGITS_AVAILABLE and H.is_cuda and W.is_cuda and idx.is_cuda
    if use_fused:
        try:
            out = _indexed_logits_fn(H, W, idx.to(torch.int32))
        except Exception as e:  # pragma: no cover - kernel-specific failure
            logger.warning("indexed_logits fused kernel failed: %s. Using fallback.", e)
            out = torch.gather(H @ W.T, dim=1, index=idx.long())
    else:
        out = torch.gather(H @ W.T, dim=1, index=idx.long())

    if bias is not None:
        out = out + _gather_bias(bias, idx)

    return out
