from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist


@dataclass
class DDPState:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_ddp(backend: str = "nccl", enabled: bool = True) -> DDPState:
    want = (
        enabled
        and ("RANK" in os.environ)
        and ("WORLD_SIZE" in os.environ)
        and ("LOCAL_RANK" in os.environ)
    )
    if not want:
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
    if not state.enabled:
        return
    dist.barrier()
    dist.destroy_process_group()


def all_reduce_sum(x: torch.Tensor, state: DDPState) -> torch.Tensor:
    if not state.enabled:
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def all_gather_vec(vec: torch.Tensor, state: DDPState) -> List[torch.Tensor]:
    if not state.enabled:
        return [vec]
    outs = [torch.zeros_like(vec) for _ in range(state.world_size)]
    dist.all_gather(outs, vec)
    return outs