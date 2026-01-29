# src/loralens/__init__.py
"""
LoRA Lens - Scalable lens-based interpretability for large language models.

This package provides:
- Hooks: Standalone activation capture system
- Losses: Pluggable loss functions (KL, subset KL, CE)
- Lenses: Neural network modules (logit, tuned, LoRA)
- Ops: Custom CUDA operations (indexed_logits)
- Training: Thin orchestration layer

Example usage:

    from loralens.hooks import ActivationCollector
    from loralens.losses import create_loss
    from loralens.lenses import create_lens
    from loralens.training import LensTrainer, TrainConfig
    
    # Create components
    collector = ActivationCollector(model)
    loss_fn = create_loss("subset_kl", k_head=100, k_tail=50)
    lens = create_lens("lora", layer_ids=range(12), hidden_size=768, unembed=unembed, r=16)
    
    # Train
    trainer = LensTrainer(model, lens, loss_fn, collector, config, ddp_state, amp_ctx)
    trainer.train(dataloader_factory, optimizer)
"""

__version__ = "0.2.0"

# Convenient imports
from loralens.hooks import ActivationCollector, HookManager
from loralens.losses import create_loss, BaseLoss
from loralens.lenses import create_lens, BaseLens, LogitLens, TunedLens, LoRALens
from loralens.training import LensTrainer, TrainConfig, HFUnembed, get_model_config

# Ops (optional - may not be available without CUDA extension)
try:
    from loralens.ops import indexed_logits, indexed_logits_available
except ImportError:
    indexed_logits = None
    indexed_logits_available = lambda: False

__all__ = [
    # Version
    "__version__",
    # Hooks
    "ActivationCollector",
    "HookManager",
    # Losses
    "create_loss",
    "BaseLoss",
    # Lenses
    "create_lens",
    "BaseLens",
    "LogitLens",
    "TunedLens",
    "LoRALens",
    # Ops
    "indexed_logits",
    "indexed_logits_available",
    # Training
    "LensTrainer",
    "TrainConfig",
    "HFUnembed",
    "get_model_config",
]
