# src/loralens/ops/indexed_logits.py
"""
Wrapper for the indexed_logits CUDA extension.

This provides a memory-efficient way to compute:
    out[i,j] = dot(H[i,:], W[idx[i,j],:])

Without materializing [N, k, d] or [N, V] intermediate tensors.

The extension is optional - if not available, we fall back to 
the standard (less memory-efficient) implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import the CUDA extension
_INDEXED_LOGITS_AVAILABLE = False
_indexed_logits_fn = None

try:
    from indexed_logits import indexed_logits as _indexed_logits_fn
    _INDEXED_LOGITS_AVAILABLE = True
    logger.info("indexed_logits CUDA extension loaded successfully")
except ImportError:
    logger.warning(
        "indexed_logits CUDA extension not available. "
        "Falling back to standard implementation. "
        "Install with: pip install git+https://github.com/pettyjohnjn/indexed_logits.git"
    )


def indexed_logits_available() -> bool:
    """Check if the indexed_logits CUDA extension is available."""
    return _INDEXED_LOGITS_AVAILABLE


@dataclass
class IndexedLogitsConfig:
    """Configuration for indexed logits computation."""
    # Whether to use fused CUDA kernel (if available)
    use_fused: bool = True
    # Whether to fall back to standard impl if fused fails
    allow_fallback: bool = True
    # Force int32 indices (recommended for CUDA kernel)
    force_int32_indices: bool = True


def _ensure_contiguous_and_dtype(
    H: torch.Tensor,
    W: torch.Tensor, 
    idx: torch.Tensor,
    force_int32: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ensure tensors meet the CUDA extension requirements.
    
    Requirements:
    - H: [N, d], float16/bfloat16, CUDA, contiguous
    - W: [V, d], same dtype as H, CUDA, contiguous
    - idx: [N, k], int32 preferred, CUDA, contiguous
    """
    # Contiguity
    if not H.is_contiguous():
        H = H.contiguous()
    if not W.is_contiguous():
        W = W.contiguous()
    if not idx.is_contiguous():
        idx = idx.contiguous()
    
    # Index dtype
    if force_int32 and idx.dtype != torch.int32:
        idx = idx.to(torch.int32)
    
    # Dtype match check
    if H.dtype != W.dtype:
        raise ValueError(f"H and W must have same dtype: got {H.dtype} vs {W.dtype}")
    
    # Device check
    if not (H.is_cuda and W.is_cuda and idx.is_cuda):
        raise ValueError("All tensors must be on CUDA device")
    
    return H, W, idx


def _indexed_logits_fallback(
    H: torch.Tensor,
    W: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    Fallback implementation using standard PyTorch ops.
    
    This computes full logits then gathers - NOT memory efficient,
    but provides a reference implementation.
    
    Args:
        H: [N, d] hidden states
        W: [V, d] weight matrix  
        idx: [N, k] indices
        
    Returns:
        [N, k] logits for selected indices
    """
    N, d = H.shape
    N2, k = idx.shape
    assert N == N2, f"Batch size mismatch: H has {N}, idx has {N2}"
    
    # Full matmul then gather
    full_logits = H @ W.T  # [N, V]
    return torch.gather(full_logits, dim=1, index=idx.long())  # [N, k]


def indexed_logits(
    H: torch.Tensor,
    W: torch.Tensor,
    idx: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    config: Optional[IndexedLogitsConfig] = None,
) -> torch.Tensor:
    """
    Compute indexed logits: out[i,j] = dot(H[i,:], W[idx[i,j],:]) + bias[idx[i,j]]
    
    This is memory-efficient: avoids materializing [N, k, d] or [N, V].
    
    Args:
        H: Hidden states [N, d], float16 or bfloat16, CUDA
        W: Weight matrix [V, d], same dtype, CUDA
        idx: Token indices [N, k], int32 preferred, CUDA
        bias: Optional bias [V], same dtype as W
        config: Optional configuration
        
    Returns:
        Logits [N, k], same dtype as H
        
    Example:
        >>> N, d, V, k = 4096, 1024, 50000, 128
        >>> H = torch.randn(N, d, dtype=torch.bfloat16, device='cuda')
        >>> W = torch.randn(V, d, dtype=torch.bfloat16, device='cuda')
        >>> idx = torch.randint(0, V, (N, k), dtype=torch.int32, device='cuda')
        >>> out = indexed_logits(H, W, idx)  # [N, k]
    """
    if config is None:
        config = IndexedLogitsConfig()
    
    # Prepare tensors
    H, W, idx = _ensure_contiguous_and_dtype(H, W, idx, config.force_int32_indices)
    
    # Try fused kernel
    if config.use_fused and _INDEXED_LOGITS_AVAILABLE:
        try:
            out = _indexed_logits_fn(H, W, idx)
            
            # Add bias if present (need to gather from bias)
            if bias is not None:
                bias_k = torch.gather(bias.unsqueeze(0).expand(idx.shape[0], -1), 
                                      dim=1, index=idx.long())
                out = out + bias_k
            
            return out
        except Exception as e:
            if not config.allow_fallback:
                raise
            logger.warning(f"indexed_logits fused kernel failed: {e}. Using fallback.")
    
    # Fallback
    out = _indexed_logits_fallback(H, W, idx)
    
    if bias is not None:
        bias_k = torch.gather(bias.unsqueeze(0).expand(idx.shape[0], -1),
                              dim=1, index=idx.long())
        out = out + bias_k
    
    return out


def indexed_logits_with_lora(
    H: torch.Tensor,
    W_base: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    idx: torch.Tensor,
    scale: float = 1.0,
    bias: Optional[torch.Tensor] = None,
    config: Optional[IndexedLogitsConfig] = None,
) -> torch.Tensor:
    """
    Compute indexed logits with LoRA decomposition.
    
    Computes: out[i,j] = H[i] @ W_base[idx[i,j]] + scale * (H[i] @ B.T @ A[idx[i,j]])
    
    This avoids materializing W_eff = W_base + scale * A @ B.
    
    For a LoRA lens where:
        - W_base is the frozen unembed weight [V, d]
        - A is lora_A weight [V, r] (maps from rank to vocab)
        - B is lora_B weight [r, d] (maps from hidden to rank)
        - scale = alpha / r
    
    Args:
        H: Hidden states [N, d]
        W_base: Base weight [V, d]
        lora_A: LoRA A matrix [V, r] - note: this is the "output" LoRA matrix
        lora_B: LoRA B matrix [r, d] - note: this is the "input" LoRA matrix  
        idx: Token indices [N, k]
        scale: LoRA scale factor (alpha / r)
        bias: Optional bias [V]
        config: Optional configuration
        
    Returns:
        Logits [N, k]
        
    Note:
        Standard LoRA formulation for output projection:
            y = x @ W_base.T + scale * x @ B.T @ A.T
        
        For indexed computation:
            y[idx] = x @ W_base[idx].T + scale * (x @ B.T) @ A[idx].T
                   = indexed_logits(x, W_base, idx) + scale * indexed_logits(x @ B.T, A, idx)
    """
    if config is None:
        config = IndexedLogitsConfig()
    
    # Ensure inputs are ready
    H, W_base, idx = _ensure_contiguous_and_dtype(H, W_base, idx, config.force_int32_indices)
    
    # Base contribution: H @ W_base[idx].T
    if config.use_fused and _INDEXED_LOGITS_AVAILABLE:
        try:
            base_logits = _indexed_logits_fn(H, W_base, idx)
        except Exception as e:
            if not config.allow_fallback:
                raise
            logger.warning(f"indexed_logits base failed: {e}. Using fallback.")
            base_logits = _indexed_logits_fallback(H, W_base, idx)
    else:
        base_logits = _indexed_logits_fallback(H, W_base, idx)
    
    # LoRA contribution: scale * (H @ B.T) @ A[idx].T
    # First compute Z = H @ B.T  [N, r]
    Z = (H @ lora_B.T).contiguous()
    
    # Ensure lora_A is contiguous and same dtype
    lora_A = lora_A.contiguous()
    if lora_A.dtype != Z.dtype:
        lora_A = lora_A.to(Z.dtype)
    
    # Then compute indexed logits from Z using A as weight
    if config.use_fused and _INDEXED_LOGITS_AVAILABLE:
        try:
            lora_logits = _indexed_logits_fn(Z, lora_A, idx)
        except Exception as e:
            if not config.allow_fallback:
                raise
            logger.warning(f"indexed_logits lora failed: {e}. Using fallback.")
            lora_logits = _indexed_logits_fallback(Z, lora_A, idx)
    else:
        lora_logits = _indexed_logits_fallback(Z, lora_A, idx)
    
    # Combine
    out = base_logits + scale * lora_logits
    
    # Add bias if present
    if bias is not None:
        bias_k = torch.gather(bias.unsqueeze(0).expand(idx.shape[0], -1),
                              dim=1, index=idx.long())
        out = out + bias_k
    
    return out


# Correctness verification utilities

def verify_indexed_logits_correctness(
    N: int = 128,
    d: int = 256,
    V: int = 1000,
    k: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    """
    Verify indexed_logits produces correct results against dense reference.
    
    Returns dict with:
        - forward_match: bool
        - forward_max_err: float
        - grad_H_match: bool (if extension supports backward)
        - grad_W_match: bool (if extension supports backward)
    """
    results = {}
    
    # Setup
    H = torch.randn(N, d, dtype=dtype, device=device, requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device=device, requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device=device)
    
    # Dense reference
    H_ref = H.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    
    full_logits = H_ref @ W_ref.T  # [N, V]
    dense_out = torch.gather(full_logits, dim=1, index=idx.long())
    
    # Fused result
    fused_out = indexed_logits(H, W, idx)
    
    # Forward check
    forward_err = (dense_out - fused_out).abs().max().item()
    results["forward_max_err"] = forward_err
    results["forward_match"] = forward_err < atol
    
    # Backward check (if possible)
    try:
        # Backward on dense
        loss_dense = dense_out.sum()
        loss_dense.backward()
        grad_H_ref = H_ref.grad.clone()
        grad_W_ref = W_ref.grad.clone()
        
        # Backward on fused
        loss_fused = fused_out.sum()
        loss_fused.backward()
        grad_H_fused = H.grad
        grad_W_fused = W.grad
        
        if grad_H_fused is not None:
            grad_H_err = (grad_H_ref - grad_H_fused).abs().max().item()
            results["grad_H_max_err"] = grad_H_err
            results["grad_H_match"] = grad_H_err < atol
        
        if grad_W_fused is not None:
            # Note: grad_W might be in fp32 for numerical stability
            grad_W_err = (grad_W_ref.float() - grad_W_fused.float()).abs().max().item()
            results["grad_W_max_err"] = grad_W_err
            results["grad_W_match"] = grad_W_err < atol * 10  # looser tolerance for accumulated grads
            
    except Exception as e:
        results["backward_error"] = str(e)
    
    return results
