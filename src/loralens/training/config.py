# src/loralens/training/config.py
"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union


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
    lora_alpha: float = 1.0
    
    # Loss
    loss_type: Literal["kl", "subset_kl", "shared_subset_kl", "ce"] = "kl"
    kl_chunk_size: Optional[int] = 128
    subset_kl_k: int = 128  # Top-k tokens for subset KL
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
    
    # Logging
    log_every: int = 10
    log_memory: bool = True
    
    # Checkpointing
    save_every: int = 500
    output_dir: Path = Path("./checkpoints")
    
    # Misc
    seed: int = 0
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.token_shift is None:
            self.token_shift = 0 if self.loss_type in ("kl", "subset_kl", "shared_subset_kl") else 1
        
        if self.stride is None:
            self.stride = self.max_seq_len
        
        self.output_dir = Path(self.output_dir)
    
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
        if "text_paths" in d:
            d["text_paths"] = [Path(p) for p in d["text_paths"]]
        return cls(**d)
