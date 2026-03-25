<<<<<<< HEAD
# src/loralens/training/__init__.py
"""Training module - orchestrates lens training."""

from .config import TrainConfig
from .trainer import LensTrainer
from .unembed import HFUnembed, get_model_config
from .distributed import DDPState, init_ddp, shutdown_ddp
from .amp import AMPContext
from .model_shard import (
    ModelShardState,
    load_sharded_model,
    load_sharded_model_multinode,
    disabled_shard_state,
)

__all__ = [
    "TrainConfig",
    "LensTrainer",
    "HFUnembed",
    "get_model_config",
    "DDPState",
    "init_ddp",
    "shutdown_ddp",
    "AMPContext",
    "ModelShardState",
    "load_sharded_model",
    "load_sharded_model_multinode",
    "disabled_shard_state",
]
=======
from .loop import Train, LossChoice

__all__ = ["Train", "LossChoice"]
>>>>>>> origin/main
