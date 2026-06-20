# src/loralens/losses/sampling.py
"""
Sampling utilities for subset KL divergence estimation.

Re-exports from the ``subset_kl`` package. See that package for
full documentation and implementation details.
"""

from subset_kl import (
    pps_sample_indices_batched,
    subset_mc_kl_from_gathered,
    SamplingDiagnostics,
)

__all__ = [
    "pps_sample_indices_batched",
    "subset_mc_kl_from_gathered",
    "SamplingDiagnostics",
]
