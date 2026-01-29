# src/loralens/training/__init__.py
"""
Training module - Orchestrates lens training.

This module provides the training loop and utilities. It delegates
to specialized components for activation collection, loss computation,
and lens inference.

Example usage:
    
    # Create components
    collector = ActivationCollector(model)
    loss_fn = create_loss("subset_kl", k_head=100)
    lens = create_lens("lora", ...)
    
    # Create trainer
    trainer = LensTrainer(
        model=model,
        lens=lens,
        loss_fn=loss_fn,
        collector=collector,
        config=config,
    )
    
    # Train
    trainer.train()
"""

from .config import TrainConfig
from .trainer import LensTrainer
from .unembed import HFUnembed, get_model_config
from .distributed import DDPState, init_ddp, shutdown_ddp
from .amp import AMPContext

__all__ = [
    "TrainConfig",
    "LensTrainer",
    "HFUnembed",
    "get_model_config",
    "DDPState",
    "init_ddp",
    "shutdown_ddp",
    "AMPContext",
]
