# src/loralens/training/model_shard.py
"""
Model sharding for large teacher models.

Single-node: HuggingFace ``device_map`` splits layers across local GPUs.
Multi-node: PyTorch FSDP shards all parameters across ranks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class ModelShardState:
    """Tracks how the teacher model is distributed across GPUs."""

    enabled: bool = False
    lens_device: torch.device = field(default_factory=lambda: torch.device("cuda:0"))
    device_map: Optional[Dict[str, Any]] = None
    num_gpus: int = 1
    model_dtype: torch.dtype = torch.bfloat16
    gpu_memory_used: Optional[Dict[int, str]] = None

    @property
    def is_sharded(self) -> bool:
        """True when model lives on more than one device."""
        return self.enabled and self.num_gpus > 1

    def __repr__(self) -> str:
        if not self.enabled:
            return "ModelShardState(enabled=False)"
        return (
            f"ModelShardState(num_gpus={self.num_gpus}, "
            f"dtype={self.model_dtype}, lens_device={self.lens_device})"
        )

def disabled_shard_state(device: torch.device) -> ModelShardState:
    """Return a no-op shard state for the single-GPU path."""
    return ModelShardState(enabled=False, lens_device=device)

def _build_max_memory(
    gpu_ids: List[int],
    per_gpu_max_memory: Optional[str] = None,
    cpu_offload_gb: int = 0,
) -> Dict:
    """
    Build ``max_memory`` dict for ``from_pretrained``.

    Parameters
    ----------
    gpu_ids : list[int]
        CUDA device indices the model may use.
    per_gpu_max_memory : str, optional
        Per-GPU limit like ``"38GiB"``. Defaults to ~90% of total.
    cpu_offload_gb : int
        Allow this many GB of CPU RAM for overflow (0 to disable).
    """
    max_mem = {}
    for gid in gpu_ids:
        if per_gpu_max_memory is not None:
            max_mem[gid] = per_gpu_max_memory
        else:
            total = torch.cuda.get_device_properties(gid).total_memory
            # Reserve 10% headroom for activations / lens
            usable_gib = int(total * 0.9 / (1024 ** 3))
            max_mem[gid] = f"{usable_gib}GiB"
    if cpu_offload_gb > 0:
        max_mem["cpu"] = f"{cpu_offload_gb}GiB"
    return max_mem
def load_sharded_model(
    model_name: str,
    *,
    model_revision: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    gpu_ids: Optional[List[int]] = None,
    per_gpu_max_memory: Optional[str] = None,
    cpu_offload_gb: int = 0,
    lens_device: Optional[torch.device] = None,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
) -> Tuple[nn.Module, ModelShardState]:
    """
    Load a HuggingFace model sharded across local GPUs via device_map="auto".

    All parameters are frozen. Requires ``accelerate``.
    """
    from transformers import AutoModelForCausalLM

    # Resolve GPU list
    if gpu_ids is None:
        num_visible = torch.cuda.device_count()
        gpu_ids = list(range(num_visible))
    num_gpus = len(gpu_ids)

    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available for model sharding.")

    # Default lens device = first GPU
    if lens_device is None:
        lens_device = torch.device("cuda", gpu_ids[0])

    logger.info(
        f"Loading sharded model: {model_name} "
        f"(dtype={dtype}, gpus={gpu_ids}, lens_device={lens_device})"
    )

    # Build memory constraints
    max_memory = _build_max_memory(gpu_ids, per_gpu_max_memory, cpu_offload_gb)
    logger.info(f"Max memory map: {max_memory}")

    # Load with device_map
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "max_memory": max_memory,
        "trust_remote_code": trust_remote_code,
    }
    if model_revision is not None:
        model_kwargs["revision"] = model_revision
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Log device placement
    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        # Count unique devices
        unique_devices = set(str(v) for v in device_map.values())
        logger.info(f"Model distributed across devices: {sorted(unique_devices)}")
        _log_device_summary(device_map)
    else:
        device_map = None

    # Estimate memory usage
    gpu_mem = {}
    for gid in gpu_ids:
        allocated = torch.cuda.memory_allocated(gid)
        gpu_mem[gid] = f"{allocated / 1e9:.1f}GB"
    logger.info(f"GPU memory after loading: {gpu_mem}")

    shard_state = ModelShardState(
        enabled=True,
        lens_device=lens_device,
        device_map=device_map,
        num_gpus=num_gpus,
        model_dtype=dtype,
        gpu_memory_used=gpu_mem,
    )

    return model, shard_state
def _log_device_summary(device_map: Dict[str, Any]) -> None:
    """Log a concise summary of layer→device placement."""
    from collections import Counter

    dev_counts = Counter(str(v) for v in device_map.values())
    for dev, count in sorted(dev_counts.items()):
        logger.info(f"  device {dev}: {count} module(s)")

def gather_to_device(
    tensors: Tuple[torch.Tensor, ...],
    target_device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """
    Move a tuple of tensors to *target_device*, skipping if already there.

    Used to gather hidden states (which may live on different GPUs
    when using ``device_map``) onto the lens device.
    """
    result = []
    for t in tensors:
        if t.device != target_device:
            result.append(t.to(target_device, non_blocking=True))
        else:
            result.append(t)
    return tuple(result)

def _get_transformer_layer_cls(model: nn.Module) -> Optional[type]:
    """
    Auto-detect the transformer block class for FSDP wrapping.

    FSDP needs to know which modules are the repeating "layers" so it
    can shard at the right granularity (one all-gather per layer).
    """
    # Common HuggingFace architectures
    _KNOWN_LAYER_CLASSES = [
        "LlamaDecoderLayer",
        "GPT2Block",
        "OPTDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "GemmaDecoderLayer",
        "PhiDecoderLayer",
        "FalconDecoderLayer",
        "BloomBlock",
        "GPTNeoXLayer",
        "GPTJBlock",
        "MPTBlock",
    ]

    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name in _KNOWN_LAYER_CLASSES:
            return module.__class__

    # Fallback: look for any module class that appears multiple times
    # at the same depth (likely transformer blocks)
    from collections import Counter
    child_classes = Counter()
    for child in model.modules():
        child_classes[child.__class__] += 1
    # Find a class with many instances (transformer blocks repeat N times)
    for cls, count in child_classes.most_common():
        if count >= 4 and cls != nn.Linear and cls != nn.LayerNorm:
            if any(hasattr(cls, attr) for attr in ["self_attn", "attention", "attn"]):
                return cls

    return None
def load_sharded_model_multinode(
    model_name: str,
    *,
    local_rank: int,
    model_revision: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    _preloaded_model: Optional[nn.Module] = None,
) -> Tuple[nn.Module, ModelShardState]:
    """
    Load a HuggingFace model sharded across nodes via PyTorch FSDP.

    Must be called after ``torch.distributed.init_process_group()``.
    If ``_preloaded_model`` is provided, skips from_pretrained and wraps directly.
    """
    from functools import partial
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM

    import torch.distributed as dist
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    logger.info(
        f"[rank {rank}/{world_size}] Loading FSDP sharded model: "
        f"{model_name} (dtype={dtype}, cpu_offload={cpu_offload})"
    )

    # Load model to CPU first (FSDP will shard and move to GPU)
    if _preloaded_model is not None:
        model = _preloaded_model
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            total_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
            logger.info(
                f"Using preloaded model: {total_params / 1e9:.1f}B params, "
                f"{total_gb:.1f}GB, sharding across {world_size} ranks"
            )
    else:
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
            # Load to CPU first; FSDP will shard and move to GPU
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }
        if model_revision is not None:
            model_kwargs["revision"] = model_revision
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Freeze everything — teacher is inference-only
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            total_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
            logger.info(
                f"Model loaded to CPU: {total_params / 1e9:.1f}B params, "
                f"{total_gb:.1f}GB, sharding across {world_size} ranks"
            )

    # Detect transformer layer class for FSDP wrapping
    layer_cls = _get_transformer_layer_cls(model)
    if layer_cls is not None:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )
        if rank == 0:
            logger.info(f"FSDP auto-wrap on: {layer_cls.__name__}")
    else:
        # Fallback: wrap by parameter count
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=1_000_000,
        )
        if rank == 0:
            logger.info("FSDP auto-wrap fallback: size_based (1M params)")


    # We do NOT use FSDP's MixedPrecision because:
    # 1. The model is already loaded in the target dtype (bf16/fp16)
    # 2. FSDP MixedPrecision casts *inputs* to param_dtype, which causes
    #    dynamically-created tensors (e.g. Llama's causal mask) to be in
    #    bf16, and torch.triu doesn't support bf16 on PyTorch 2.0/CUDA 11.7.
    # Since params are already bf16, FSDP just shards them as-is.

    # Wrap with FSDP
    fsdp_kwargs: Dict[str, Any] = {
        "auto_wrap_policy": auto_wrap_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": local_rank,
        "sync_module_states": True,  # Broadcast rank-0 weights to all ranks
        "forward_prefetch": True,    # Prefetch next layer during current forward
        "limit_all_gathers": True,   # Limit concurrent all-gathers for memory
    }

    if cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    model = FSDP(model, **fsdp_kwargs)

    if rank == 0:
        _total_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        logger.info(f"FSDP wrapping complete. Per-rank shard: ~{_total_gb / world_size:.1f}GB")

    shard_state = ModelShardState(
        enabled=True,
        lens_device=device,
        device_map=None,
        num_gpus=world_size,
        model_dtype=dtype,
    )

    return model, shard_state


def extract_unembed_from_fsdp(
    model: nn.Module,
    unembed: nn.Module,
    target_device: torch.device,
) -> nn.Module:
    """Clone unembed weights from an FSDP-wrapped model via summon_full_params."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    # Summon full parameters on all ranks temporarily
    with FSDP.summon_full_params(model, writeback=False, recurse=True):
        cloned = unembed.clone_to_device(target_device)

    logger.info(f"Extracted unembed from FSDP to {target_device}")
    return cloned
