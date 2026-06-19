# src/loralens/training/__init__.py
"""Training module - orchestrates lens training."""

from .config import TrainConfig
from .activation_sites import (
    ActivationSitePlan,
    adapt_activation_site_plan_for_model,
    build_activation_site_plan,
)
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
    "ActivationSitePlan",
    "adapt_activation_site_plan_for_model",
    "build_activation_site_plan",
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
