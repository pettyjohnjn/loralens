# LoRA Lens Refactoring Guide
## Splitting monolithic codebase into hookbox + subset-kl + indexed_logits

---

## Summary

This refactoring extracts core infrastructure into three standalone packages
while keeping loralens focused on lens-specific logic. The approach uses **shim
modules** so that all existing import paths continue to work without changes
to consuming code (trainer, CLI, lenses, tests).

### Before vs After

```
BEFORE (monolithic)                    AFTER (modular)
─────────────────                      ───────────────
loralens/                              loralens/
├── hooks/           (500+ lines)      ├── hooks/__init__.py    (25 lines, re-exports hookbox)
│   ├── base.py                        ├── losses/
│   ├── activation_hook.py             │   ├── base.py          (kept, has labels param)
│   ├── collector.py                   │   ├── kl.py            (70 lines, wraps subset_kl.KLDivergenceLoss)
│   └── manager.py                     │   ├── subset_kl.py     (130 lines, wraps subset_kl.SubsetKLLoss)
├── losses/                            │   ├── sampling.py      (25 lines, re-exports subset_kl)
│   ├── base.py                        │   ├── cross_entropy.py (unchanged)
│   ├── kl.py            (160 lines)   │   ├── shared_subset_kl.py (unchanged)
│   ├── subset_kl.py     (340 lines)   │   ├── factory.py       (unchanged)
│   ├── sampling.py      (300 lines)   │   └── __init__.py      (updated)
│   ├── cross_entropy.py               ├── lenses/              (unchanged)
│   ├── shared_subset_kl.py            ├── ops/                 (unchanged, already wraps indexed_logits)
│   ├── factory.py                     ├── training/            (unchanged, imports still work)
│   └── __init__.py                    ├── data/                (unchanged)
├── lenses/                            └── cli/                 (unchanged)
├── ops/
│   ├── indexed_logits.py
│   └── benchmark.py
├── training/
├── data/
└── cli/
```

---

## Files Changed

### 1. DELETED (moved to external packages)

| File | Replacement | Lines removed |
|------|-------------|---------------|
| `src/loralens/hooks/base.py` | `hookbox.BaseHook` | ~60 |
| `src/loralens/hooks/activation_hook.py` | `hookbox.ActivationHook` | ~100 |
| `src/loralens/hooks/collector.py` | `hookbox.ActivationCollector` | ~260 |
| `src/loralens/hooks/manager.py` | `hookbox.HookManager` | ~210 |
| `tests/hooks/__init__.py` | hookbox test suite | — |
| `tests/hooks/test_hooks.py` | hookbox test suite | ~170 |
| `src/loralens/ops/benchmark.py` | indexed_logits test suite | ~215 |

**Total removed: ~1,015 lines**

### 2. REPLACED (with thin adapters/shims)

| File | What changed | New size |
|------|-------------|----------|
| `src/loralens/hooks/__init__.py` | Now re-exports from `hookbox` | 25 lines |
| `src/loralens/losses/kl.py` | Wraps `subset_kl.KLDivergenceLoss`, adds `labels` param | 70 lines |
| `src/loralens/losses/subset_kl.py` | Wraps `subset_kl.SubsetKLLoss` + `HajekKLLoss` | 130 lines |
| `src/loralens/losses/sampling.py` | Re-exports from `subset_kl` | 25 lines |
| `src/loralens/losses/__init__.py` | Updated docstring and imports | 55 lines |
| `src/loralens/__init__.py` | Updated docstring, bumped version to 0.3.0 | 60 lines |

**Net reduction in these files: ~700 lines → ~365 lines**

### 3. UNCHANGED (imports still work via shims)

| File | Why unchanged |
|------|---------------|
| `src/loralens/losses/base.py` | Kept: has `labels` param not in `subset_kl.BaseLoss` |
| `src/loralens/losses/cross_entropy.py` | Kept: not in subset-kl package |
| `src/loralens/losses/shared_subset_kl.py` | Kept: tightly coupled to lens interface |
| `src/loralens/losses/factory.py` | Unchanged: registers adapter classes |
| `src/loralens/lenses/*` | Unchanged: `from loralens.ops import ...` still works |
| `src/loralens/ops/indexed_logits.py` | Unchanged: already wraps external `indexed_logits` |
| `src/loralens/training/trainer.py` | Unchanged: `from loralens.hooks/losses import ...` still works |
| `src/loralens/cli/main.py` | Unchanged: all imports go through shims |
| `src/loralens/data/*` | Unchanged: no dependency on hooks/losses/ops |
| `tests/lenses/*` | Unchanged |

### 4. UPDATED

| File | Change |
|------|--------|
| `pyproject.toml` | Added `hookbox` and `subset-kl` dependencies, optional `indexed_logits[cuda]` |
| `README.md` | Updated architecture docs, dependency table |
| `tests/losses/test_losses.py` | Updated `SubsetKLLoss` construction (use `k=` not `k_head=`/`k_tail=`) |

---

## Dependency Graph (After)

```
loralens
├── hookbox          (pip, required)
│   └── torch
├── subset-kl        (pip, required)
│   └── torch
├── indexed_logits   (pip, optional[cuda])
│   └── torch + CUDA
├── torch            (required)
├── transformers     (required)
├── datasets         (required)
└── numpy            (required)
```

---

## Key Design Decisions

### Why shim modules instead of direct imports?

The shim approach (`loralens.hooks` → re-exports from `hookbox`) means:

1. **Zero changes** to trainer.py, cli/main.py, lenses/*.py
2. **Backward-compatible** import paths: `from loralens.hooks import ActivationCollector` still works
3. **Single place** to swap implementations if needed
4. Consumers can also import directly: `from hookbox import ActivationCollector`

### Why keep loralens.losses.base.BaseLoss?

The external `subset_kl.BaseLoss` has signature:
```python
def forward(self, student_logits, teacher_logits, attention_mask=None)
```

But loralens needs:
```python
def forward(self, student_logits, teacher_logits, attention_mask=None, labels=None)
```

The `labels` parameter is needed by `CrossEntropyLoss` and the training loop
expects a uniform interface. The adapter classes accept `labels` but pass
only the KL-relevant args to the external implementations.

### Why keep SharedSubsetKLLoss in loralens?

`SharedSubsetKLLoss.forward_with_lens()` directly accesses lens internals:
- `lens.projections[layer_id]` (LoRA weights)
- `lens.unembed.lm_head.weight` (unembedding matrix)
- `lens.unembed.ln_f()` (layer norm)

This tight coupling to the lens interface makes it loralens-specific.
Moving it to subset-kl would create a circular dependency.

### Why keep ops/indexed_logits.py wrapper?

The wrapper adds value beyond the raw external package:
- `indexed_logits_available()` availability check
- `IndexedLogitsConfig` for controlling fallback behavior
- `_ensure_contiguous_and_dtype()` validation
- `_indexed_logits_fallback()` CPU/non-CUDA fallback
- `indexed_logits_with_lora()` for LoRA-specific computation
- Bias support (external package doesn't handle bias)

---

## Migration Checklist

- [x] Replace `loralens/hooks/` with hookbox re-export shim
- [x] Replace `loralens/losses/kl.py` with subset_kl adapter
- [x] Replace `loralens/losses/subset_kl.py` with subset_kl adapter
- [x] Replace `loralens/losses/sampling.py` with subset_kl re-exports
- [x] Update `pyproject.toml` dependencies
- [x] Update `__init__.py` version to 0.3.0
- [x] Update `README.md` architecture docs
- [x] Update tests to match new APIs
- [x] Delete `tests/hooks/` (now in hookbox)
- [x] Delete `src/loralens/hooks/{base,activation_hook,collector,manager}.py`
- [ ] Run full test suite to verify
- [ ] Update CI/CD to install hookbox + subset-kl
