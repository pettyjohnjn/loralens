# LoRA Lens

**Scalable lens-based interpretability for large language models.**

## Architecture

This codebase is organized around **clean separation of concerns**, with
core infrastructure extracted into standalone packages:

```
loralens/
├── hooks/          # Re-exports from hookbox (activation capture)
├── losses/         # Adapters over subset-kl + local losses
├── lenses/         # Neural network modules (logit, tuned, LoRA)
├── ops/            # Wrapper for indexed_logits CUDA extension
├── training/       # Thin orchestration layer
├── data/           # Data loading and chunking
└── cli/            # Command-line interface
```

### External Dependencies

| Package | Role | Install |
|---------|------|---------|
| **hookbox** | Activation capture from transformer models | `pip install hookbox` |
| **subset-kl** | Memory-efficient KL divergence for large vocabs | `pip install subset-kl` |
| **indexed_logits** | CUDA kernel for `H @ W[indices].T` (optional) | `pip install indexed_logits` |

### What lives where

- **hookbox** handles all activation capture, hook management, and distributed
  model unwrapping. `loralens.hooks` is a thin re-export shim.

- **subset-kl** provides the core KL math: `SubsetKLLoss`, `KLDivergenceLoss`,
  `HajekKLLoss`, and functional APIs like `select_topk_indices` +
  `subset_kl_from_gathered`. `loralens.losses` wraps these with adapters that
  add the `labels` parameter for cross-entropy compatibility.

- **indexed_logits** is the CUDA extension for computing `out[i,j] =
  dot(H[i,:], W[idx[i,j],:])` without materialising `[N, k, d]` tensors.
  `loralens.ops` wraps it with fallback support.

- **loralens** itself provides: lenses (LogitLens, TunedLens, LoRALens),
  SharedSubsetKLLoss (tightly coupled to lenses), CrossEntropyLoss,
  training orchestration, data loading, and the CLI.

## Quick Start

```python
from loralens.hooks import ActivationCollector
from loralens.losses import create_loss
from loralens.lenses import create_lens
from loralens.training import LensTrainer, TrainConfig

# Create components
collector = ActivationCollector(model)
loss_fn = create_loss("subset_kl", k=256)
lens = create_lens("lora", layer_ids=range(12), hidden_size=768, unembed=unembed, r=16)

# Train
trainer = LensTrainer(model, lens, loss_fn, collector, config, ddp_state, amp_ctx)
trainer.train(dataloader_factory, optimizer)
```

## Installation

```bash
pip install loralens                # Core (includes hookbox + subset-kl)
pip install loralens[cuda]          # + indexed_logits CUDA extension
pip install loralens[dev]           # + test dependencies
```

## Training

```bash
loralens train --model_name gpt2 --lens_type lora --loss_type subset_kl
```

## Loss Functions

| Loss | Memory | Use Case |
|------|--------|----------|
| `kl` | O(B·T·V) | Baseline, small models |
| `subset_kl` | O(B·T·k) | Recommended default |
| `shared_subset_kl` | O(B·chunk·K) | Largest models |
| `ce` | O(B·T·V) | Label-based training |
