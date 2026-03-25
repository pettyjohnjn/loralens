# src/loralens/training/unembed.py
"""Unembedding utilities for extracting LM heads from HuggingFace models."""

from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class HFUnembed(nn.Module):
    """
    Unembedding module that extracts and applies LM head from HF models.

    For GPT-2-like models: applies ln_f then lm_head.
    For LLaMA-like models: applies norm then lm_head.

    Parameters
    ----------
    model : nn.Module
        HuggingFace causal LM model.

    Notes
    -----
    The unembed parameters are always frozen - only the lens
    parameters are trained.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()

        self.layer_norm = None
        self.lm_head = None
        self.vocab_size = None

        # Try to find layer norm
        # GPT-2 style
        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            self.layer_norm = model.transformer.ln_f
        # LLaMA style
        elif hasattr(model, "model") and hasattr(model.model, "norm"):
            self.layer_norm = model.model.norm

        # Find LM head
        if hasattr(model, "lm_head"):
            self.lm_head = model.lm_head
        elif hasattr(model, "get_output_embeddings"):
            self.lm_head = model.get_output_embeddings()
        else:
            raise ValueError("Could not locate LM head in model.")

        # Get vocab size
        if hasattr(self.lm_head, "out_features"):
            self.vocab_size = self.lm_head.out_features
        elif hasattr(self.lm_head, "weight"):
            self.vocab_size = self.lm_head.weight.shape[0]

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        idx_subset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply layer norm (if present) and LM head.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape [batch, seq, hidden].
        idx_subset : Optional[torch.Tensor]
            If provided, only compute logits for these vocab indices.
            Shape [batch, seq, k] for per-position indices.

        Returns
        -------
        torch.Tensor
            Logits of shape [batch, seq, vocab] or [batch, seq, k] if idx_subset.
        """
        h = hidden_states

        if self.layer_norm is not None:
            h = self.layer_norm(h)

        if idx_subset is None:
            # Full vocabulary path
            return self.lm_head(h)

        # Efficient subset path with unique index deduplication
        # This optimization is crucial when many positions share similar top-k indices
        B, T, k = idx_subset.shape
        d = h.size(-1)

        h_flat = h.reshape(-1, d)              # [B*T, d]
        idx_flat = idx_subset.reshape(-1, k)   # [B*T, k]

        # Deduplicate indices - key optimization!
        # Many positions may share the same top-k indices
        uniq_idx, inverse = torch.unique(idx_flat, return_inverse=True)

        # Only gather unique rows from weight matrix
        W_sel = self.lm_head.weight[uniq_idx]  # [num_unique, d]

        # Compute logits for unique indices
        logits_sel = h_flat @ W_sel.T          # [B*T, num_unique]

        # Add bias if present
        if self.lm_head.bias is not None:
            logits_sel = logits_sel + self.lm_head.bias[uniq_idx]

        # Map back to per-position indices via inverse
        logits_k = torch.gather(
            logits_sel, 1, inverse.view(-1, k)
        ).view(B, T, k)

        return logits_k

    def __repr__(self) -> str:
        has_ln = self.layer_norm is not None
        return f"HFUnembed(vocab_size={self.vocab_size}, has_layer_norm={has_ln})"

    def clone_to_device(self, device: torch.device) -> "HFUnembed":
        """Create a standalone copy of the unembed weights on *device*."""
        import copy

        clone = object.__new__(HFUnembed)
        nn.Module.__init__(clone)

        if self.layer_norm is not None:
            clone.layer_norm = copy.deepcopy(self.layer_norm).to(device)
        else:
            clone.layer_norm = None

        clone.lm_head = copy.deepcopy(self.lm_head).to(device)
        clone.vocab_size = self.vocab_size

        for p in clone.parameters():
            p.requires_grad = False

        return clone


def get_model_config(model: nn.Module) -> Dict[str, Any]:
    """Extract num_layers, hidden_size, vocab_size from a HuggingFace model."""
    config = model.config

    # Number of layers
    if hasattr(config, "n_layer"):
        num_layers = config.n_layer
    elif hasattr(config, "num_hidden_layers"):
        num_layers = config.num_hidden_layers
    else:
        raise ValueError("Cannot determine number of layers from config.")

    # Hidden size
    if hasattr(config, "n_embd"):
        hidden_size = config.n_embd
    elif hasattr(config, "hidden_size"):
        hidden_size = config.hidden_size
    else:
        raise ValueError("Cannot determine hidden size from config.")

    # Vocab size
    if hasattr(config, "vocab_size"):
        vocab_size = config.vocab_size
    else:
        raise ValueError("Cannot determine vocab size from config.")

    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
    }
