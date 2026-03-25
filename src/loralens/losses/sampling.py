# src/loralens/losses/sampling.py
"""
Sampling utilities for subset KL divergence estimation.

Re-exports from the ``subset_kl`` package. See that package for
full documentation and implementation details.
"""

from subset_kl import (
    pps_sample_indices_batched,
    hajek_kl_estimate,
    SamplingDiagnostics,
)


def head_tail_kl(*args, **kwargs):
    """Simplified head-tail KL interface (delegates to subset_kl)."""
    from subset_kl import compute_subset_kl
    return compute_subset_kl(*args, **kwargs)


__all__ = [
    "pps_sample_indices_batched",
    "hajek_kl_estimate",
    "head_tail_kl",
    "SamplingDiagnostics",
]
