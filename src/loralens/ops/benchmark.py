# src/loralens/ops/benchmark.py
"""
Benchmarking utilities for indexed_logits performance comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    forward_ms: float
    backward_ms: float
    total_ms: float
    peak_memory_mb: float
    allocated_memory_mb: float
    
    def __repr__(self) -> str:
        return (
            f"{self.name}: fwd={self.forward_ms:.2f}ms bwd={self.backward_ms:.2f}ms "
            f"total={self.total_ms:.2f}ms peak={self.peak_memory_mb:.1f}MB"
        )


def _time_cuda_ms(fn, warmup: int = 3, repeats: int = 10) -> float:
    """Time a CUDA function in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    
    # Actual timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return sum(times) / len(times)


def benchmark_indexed_logits(
    N: int = 8192,
    d: int = 4096,
    V: int = 50257,
    k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    warmup: int = 3,
    repeats: int = 10,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Benchmark indexed_logits vs dense baseline.
    
    Args:
        N: Batch size (B*T flattened)
        d: Hidden dimension
        V: Vocabulary size
        k: Number of indices per position
        dtype: Tensor dtype
        device: Device
        warmup: Warmup iterations
        repeats: Measurement iterations
        
    Returns:
        Tuple of (fused_result, dense_result)
    """
    from loralens.ops import indexed_logits, indexed_logits_available
    
    # Setup
    H = torch.randn(N, d, dtype=dtype, device=device, requires_grad=True)
    W = torch.randn(V, d, dtype=dtype, device=device, requires_grad=True)
    idx = torch.randint(0, V, (N, k), dtype=torch.int32, device=device)
    grad_out = torch.randn(N, k, dtype=dtype, device=device)
    
    results = []
    
    # Benchmark fused path
    if indexed_logits_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        def fused_forward():
            return indexed_logits(H, W, idx)
        
        def fused_backward():
            out = indexed_logits(H, W, idx)
            out.backward(grad_out)
            H.grad = None
            W.grad = None
        
        fwd_ms = _time_cuda_ms(fused_forward, warmup, repeats)
        bwd_ms = _time_cuda_ms(fused_backward, warmup, repeats)
        
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        alloc_mem = torch.cuda.memory_allocated(device) / 1024 / 1024
        
        results.append(BenchmarkResult(
            name="indexed_logits (fused)",
            forward_ms=fwd_ms,
            backward_ms=bwd_ms - fwd_ms,  # Approximate backward time
            total_ms=bwd_ms,
            peak_memory_mb=peak_mem,
            allocated_memory_mb=alloc_mem,
        ))
    else:
        results.append(None)
    
    # Benchmark dense baseline
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    H_dense = H.detach().clone().requires_grad_(True)
    W_dense = W.detach().clone().requires_grad_(True)
    
    def dense_forward():
        full_logits = H_dense @ W_dense.T  # [N, V]
        return torch.gather(full_logits, dim=1, index=idx.long())
    
    def dense_backward():
        full_logits = H_dense @ W_dense.T
        out = torch.gather(full_logits, dim=1, index=idx.long())
        out.backward(grad_out)
        H_dense.grad = None
        W_dense.grad = None
    
    fwd_ms = _time_cuda_ms(dense_forward, warmup, repeats)
    bwd_ms = _time_cuda_ms(dense_backward, warmup, repeats)
    
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    alloc_mem = torch.cuda.memory_allocated(device) / 1024 / 1024
    
    results.append(BenchmarkResult(
        name="dense + gather (baseline)",
        forward_ms=fwd_ms,
        backward_ms=bwd_ms - fwd_ms,
        total_ms=bwd_ms,
        peak_memory_mb=peak_mem,
        allocated_memory_mb=alloc_mem,
    ))
    
    return tuple(results)


def benchmark_training_step(
    model_name: str = "gpt2",
    batch_size: int = 4,
    seq_len: int = 512,
    k: int = 128,
    use_fused: bool = True,
) -> dict:
    """
    Benchmark a full training step with subset KL loss.
    
    This provides end-to-end timing and memory measurement.
    """
    # This would require the full training setup
    # Placeholder for now
    raise NotImplementedError(
        "Full training step benchmark requires model loading. "
        "Use benchmark_indexed_logits for microbenchmarks."
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark indexed_logits")
    parser.add_argument("--N", type=int, default=8192, help="Batch size (B*T)")
    parser.add_argument("--d", type=int, default=4096, help="Hidden dim")
    parser.add_argument("--V", type=int, default=50257, help="Vocab size")
    parser.add_argument("--k", type=int, default=128, help="Top-k")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = parser.parse_args()
    
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    
    print(f"Benchmarking indexed_logits: N={args.N}, d={args.d}, V={args.V}, k={args.k}")
    print(f"dtype={args.dtype}, device=cuda")
    print()
    
    fused, dense = benchmark_indexed_logits(
        N=args.N, d=args.d, V=args.V, k=args.k, dtype=dtype
    )
    
    if fused:
        print(fused)
    else:
        print("Fused kernel not available")
    print(dense)
    
    if fused:
        speedup = dense.total_ms / fused.total_ms
        mem_reduction = dense.peak_memory_mb / fused.peak_memory_mb
        print()
        print(f"Speedup: {speedup:.2f}x")
        print(f"Memory reduction: {mem_reduction:.2f}x")
