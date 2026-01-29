# LoRA Lens

**Scalable lens-based interpretability for large language models.**

## Architecture

This codebase is organized around **clean separation of concerns**:

```
loralens/
├── hooks/          # Standalone activation capture system
├── losses/         # Pluggable loss functions  
├── lenses/         # Neural network modules
├── training/       # Thin orchestration layer
└── cli/            # Command-line interface
```

### Hooks Module

The hooks module provides a **standalone** system for capturing activations:

```python
from loralens.hooks import ActivationCollector

# Collect activations from any model
collector = ActivationCollector(model)
with collector:
    data = collector.collect(input_ids, attention_mask)
    
for layer_id, hidden in data.iter_layers():
    print(f"Layer {layer_id}: {hidden.shape}")
```

### Losses Module

Losses are **pluggable objects** that can be swapped without touching the training loop:

```python
from loralens.losses import create_loss

# Standard KL divergence
loss_fn = create_loss("kl", chunk_size=128)

# Memory-efficient subset KL
loss_fn = create_loss("subset_kl", k_head=100, k_tail=50)

# Cross-entropy
loss_fn = create_loss("ce", label_smoothing=0.1)
```

### Lenses Module

Lenses map hidden states to vocabulary space:

```python
from loralens.lenses import create_lens

# Logit lens (no trainable params)
lens = create_lens("logit", unembed=unembed)

# Tuned lens (full-rank)
lens = create_lens("tuned", layer_ids=range(12), hidden_size=768, unembed=unembed)

# LoRA lens (parameter-efficient)
lens = create_lens("lora", layer_ids=range(12), hidden_size=768, unembed=unembed, r=16)
```

### Training Module

The trainer is a **thin orchestration layer** that delegates to components:

```python
from loralens import LensTrainer, TrainConfig

trainer = LensTrainer(
    model=model,
    lens=lens,
    loss_fn=loss_fn,
    collector=collector,
    config=config,
    ddp_state=ddp_state,
    amp_ctx=amp_ctx,
)

trainer.train(dataloader_factory, optimizer)
```

## Installation

```bash
pip install -e .
pip install -e ".[dev]"  # With dev dependencies
```

## CLI Usage

```bash
# Basic training
loralens train --model_name gpt2 --lens_type lora --loss_type kl

# With subset KL for memory efficiency
loralens train --model_name gpt2 --lens_type lora --loss_type subset_kl \
    --subset_kl_head_k 100 --subset_kl_tail_k 50

# Distributed training
torchrun --nproc_per_node=4 -m loralens.cli.main train \
    --model_name meta-llama/Llama-2-7b-hf \
    --lens_type lora --lora_rank 32 \
    --loss_type subset_kl \
    --amp --amp_dtype bf16
```

## Key Features

| Feature | Description |
|---------|-------------|
| **LoRA Lens** | Parameter-efficient lens using low-rank adaptation |
| **Subset KL** | Memory-efficient KL approximation for large vocabularies |
| **Clean Architecture** | Components can be tested and used independently |
| **Pluggable Losses** | Swap loss functions without changing training code |
| **Standalone Hooks** | Use activation collection for inference/analysis |

## Testing

```bash
pytest tests/
pytest tests/losses/  # Just losses
pytest tests/hooks/   # Just hooks
pytest tests/lenses/  # Just lenses
```

## License

MIT
