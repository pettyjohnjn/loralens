# src/loralens/training/config.py
"""Training configuration."""

from __future__ import annotations

import re
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
    lens_type: Literal["logit", "tuned", "lora", "bidir_lora"] = "lora"
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

    # Write loss (bidirectional training, only active when lens_type="bidir_lora")
    write_loss_type: Literal["none", "ortho"] = "none"
    write_loss_weight: float = 0.1

    # Loss
    loss_type: Literal["kl", "subset_kl", "ce"] = "kl"
    kl_chunk_size: Optional[int] = 128
    subset_kl_k: int = 128  # Top-k tokens for subset KL
    subset_kl_mode: Literal["topk", "mc"] = "topk"
    subset_kl_k_tail: int = 0  # Tail samples for mc subset KL (0 = head only)
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
    run_tag: Optional[str] = None  # Short label appended to canonical path (e.g., "ablation1")

    def __post_init__(self):
        """Validate and set defaults."""
        if self.lens_type in ("lora", "bidir_lora") and self.lora_alpha is None:
            self.lora_alpha = float(self.lora_rank)

        if self.token_shift is None:
            self.token_shift = 0 if self.loss_type in ("kl", "subset_kl") else 1

        if self.stride is None:
            self.stride = self.max_seq_len

        self.output_dir = Path(self.output_dir)
        if self.resume_checkpoint is not None:
            self.resume_checkpoint = Path(self.resume_checkpoint)

    def canonical_run_path(self) -> Path:
        """
        Return a canonical relative path for this run, derived from its attributes.

        Intended as a subdirectory under a base checkpoints root::

            output_dir = Path("./checkpoints") / config.canonical_run_path()

        The hierarchy is::

            {model}/{lens}/{loss}/{sites}/{init}/{tag}

        where ``{init}`` is omitted for non-LoRA lens types and ``{tag}`` is
        either the ``run_tag`` field or ``seed{seed}`` when ``run_tag`` is None.
        """
        # Model: last path component of HF name, lower-cased and slug-ified
        model_slug = self.model_name.split("/")[-1].lower()
        model_slug = re.sub(r"[^a-z0-9]+", "-", model_slug).strip("-")

        # Lens
        if self.lens_type == "lora":
            lens_slug = f"lora-r{self.lora_rank}"
        elif self.lens_type == "bidir_lora":
            write_tag = f"-{self.write_loss_type}" if self.write_loss_type != "none" else ""
            lens_slug = f"bidir-r{self.lora_rank}{write_tag}"
        else:
            lens_slug = self.lens_type  # "tuned" or "logit"

        # Loss
        if self.loss_type == "subset_kl":
            loss_slug = f"subset_kl-{self.subset_kl_mode}-k{self.subset_kl_k}"
            if self.subset_kl_k_tail > 0:
                loss_slug += f"-tail{self.subset_kl_k_tail}"
        else:
            loss_slug = self.loss_type  # "kl" or "ce"

        # Activation sites
        site_slug = self.activation_site_preset

        parts: list[str] = [model_slug, lens_slug, loss_slug, site_slug]

        # LoRA initialization strategy
        if self.lens_type in ("lora", "bidir_lora"):
            init_map = {
                "default_lora": "init-default",
                "mean_shift": "init-mean_shift",
                "ridge_svd": "init-ridge",
            }
            parts.append(init_map.get(self.lora_init, f"init-{self.lora_init}"))

        # Run tag: user-provided label or auto-generated from seed
        parts.append(self.run_tag if self.run_tag else f"seed{self.seed}")

        result = Path(parts[0])
        for p in parts[1:]:
            result = result / p
        return result

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
        """Create from dictionary, ignoring unrecognised keys for forward/backward compat."""
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        if "output_dir" in d:
            d["output_dir"] = Path(d["output_dir"])
        if d.get("resume_checkpoint") is not None:
            d["resume_checkpoint"] = Path(d["resume_checkpoint"])
        if "text_paths" in d:
            d["text_paths"] = [Path(p) for p in d["text_paths"]]
        return cls(**d)
