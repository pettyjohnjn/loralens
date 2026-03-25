# src/loralens/training/distributed.py
"""Distributed training utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DDPState:
    """State container for distributed data parallel training."""
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        """Whether this is the main (rank 0) process."""
        return self.rank == 0

    def __repr__(self) -> str:
        return (
            f"DDPState(enabled={self.enabled}, rank={self.rank}, "
            f"world_size={self.world_size}, device={self.device})"
        )


def init_ddp(backend: str = "nccl", enabled: bool = True) -> DDPState:
    """Initialize distributed training."""
    want_ddp = (
        enabled
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "LOCAL_RANK" in os.environ
    )

    if not want_ddp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DDPState(enabled=False, rank=0, world_size=1, local_rank=0, device=device)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    dist.barrier()

    return DDPState(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def shutdown_ddp(state: DDPState) -> None:
    """Clean up distributed training."""
    if not state.enabled:
        return
    dist.barrier()
    dist.destroy_process_group()


def all_reduce_sum(x: torch.Tensor, state: DDPState) -> torch.Tensor:
    """Sum tensor across all processes."""
    if not state.enabled:
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def all_reduce_mean(x: torch.Tensor, state: DDPState) -> torch.Tensor:
    """Average tensor across all processes."""
    if not state.enabled:
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y / state.world_size


def barrier(state: DDPState) -> None:
    """Synchronize all processes."""
    if state.enabled:
        dist.barrier()
