# src/loralens/training/config.py
"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class TrainConfig:
    """
    Configuration for lens training.

    Groups all hyperparameters in a single place for easy
    serialization and experiment tracking.
    """

    # Model
    model_name: str = "gpt2"
    model_revision: Optional[str] = None
    tokenizer_name: Optional[str] = None

    # Data
    data_source: Literal["text", "pile"] = "pile"
    text_paths: List[Path] = field(default_factory=list)
    text_mode: Literal["lines", "whole", "jsonl"] = "lines"
    text_json_field: str = "text"
    pile_split: str = "train"

    # Sequence
    max_seq_len: int = 1024
    stride: Optional[int] = None
    drop_remainder: bool = True
    document_separated: bool = True
    max_docs: Optional[int] = None

    # Lens
    lens_type: Literal["logit", "tuned", "lora"] = "lora"
    lora_rank: int = 16
    lora_alpha: Optional[float] = None
    lora_init: Literal["default_lora", "mean_shift", "ridge_svd"] = "default_lora"
    lora_init_calibration_tokens: int = 50000
    lora_init_ridge_lambda: float = 1e-3
    lora_init_ridge_lambda_scale: Literal["trace_xxt_over_d", "absolute"] = "trace_xxt_over_d"
    lora_init_stats_dtype: Literal["float32", "float64"] = "float32"
    lora_init_jitter: float = 1e-6
    lora_init_svd_metric: Literal["residual", "unembed"] = "residual"
    lora_init_normalization: Literal["none", "per_dim_std"] = "none"
    activation_site_preset: Literal[
        "residual",
        "llama_expanded",
        "gpt2_expanded",
        "gpt2_attention",
    ] = "residual"

    # Loss
    loss_type: Literal["kl", "subset_kl", "shared_subset_kl", "ce"] = "kl"
    kl_chunk_size: Optional[int] = 128
    subset_kl_k: int = 128  # Top-k tokens for subset KL
    subset_kl_mode: Literal["topk", "frankenstein", "hajek", "mc", "k2", "k3"] = "topk"
    subset_kl_k_tail: int = 0  # Tail samples for head/tail subset KL
    subset_kl_tail_clip: float = 50.0  # Max importance weight for Frankenstein mode
    subset_kl_tail_oversample: int = 4  # PPS oversample factor for Frankenstein mode
    subset_kl_tail_proposal: Literal["target", "teacher", "mixed", "tempered"] = "target"
    subset_kl_tail_proposal_alpha: float = 0.8
    subset_kl_tail_proposal_tau: float = 0.7
    # Shared subset KL params
    shared_subset_top_m: int = 16  # Per-position candidates
    shared_subset_max_K: int = 512  # Max shared set size
    token_shift: Optional[int] = None  # Auto: KL->0, CE->1

    # Optimization
    per_gpu_batch_size: int = 4
    num_steps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    lr_schedule: Literal["constant", "linear", "cosine"] = "constant"
    min_lr_ratio: float = 0.0
    grad_clip_norm: Optional[float] = 1.0

    # Gradient accumulation
    tokens_per_step: Optional[int] = 262144
    max_microsteps: int = 512

    # Mixed precision
    amp_enabled: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    # Distributed
    ddp_enabled: bool = True
    ddp_backend: str = "nccl"

    # Model parallelism (for large teacher models)
    model_parallel: bool = False
    model_dtype: str = "bf16"  # "bf16", "fp16", "fp32"
    model_gpu_ids: Optional[List[int]] = None  # None = all visible GPUs
    per_gpu_max_memory: Optional[str] = None  # e.g. "38GiB"
    cpu_offload_gb: int = 0  # CPU RAM for weight overflow
    attn_implementation: Optional[str] = None  # "flash_attention_2", "sdpa", etc.

    # Multi-node model parallelism (PyTorch FSDP)
    multinode_model_parallel: bool = False  # FSDP across nodes
    fsdp_offload_to_cpu: bool = False  # Offload sharded params to CPU

    # Logging
    log_every: int = 10
    log_memory: bool = True

    # Checkpointing
    save_every: int = 500
    save_initial_checkpoint: bool = False
    output_dir: Path = Path("./checkpoints")
    resume_checkpoint: Optional[Path] = None

    # Misc
    seed: int = 0

    def __post_init__(self):
        """Validate and set defaults."""
        if self.lens_type == "lora" and self.lora_alpha is None:
            self.lora_alpha = float(self.lora_rank)

        if self.token_shift is None:
            self.token_shift = 0 if self.loss_type in ("kl", "subset_kl", "shared_subset_kl") else 1

        if self.stride is None:
            self.stride = self.max_seq_len

        self.output_dir = Path(self.output_dir)
        if self.resume_checkpoint is not None:
            self.resume_checkpoint = Path(self.resume_checkpoint)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, list) and v and isinstance(v[0], Path):
                d[k] = [str(p) for p in v]
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """Create from dictionary."""
        if "output_dir" in d:
            d["output_dir"] = Path(d["output_dir"])
        if d.get("resume_checkpoint") is not None:
            d["resume_checkpoint"] = Path(d["resume_checkpoint"])
        if "text_paths" in d:
            d["text_paths"] = [Path(p) for p in d["text_paths"]]
        return cls(**d)
