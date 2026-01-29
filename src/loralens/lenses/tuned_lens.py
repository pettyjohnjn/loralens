# src/loralens/lenses/tuned_lens.py
"""
Tuned Lens - Per-layer learned translators.

Implements the tuned lens from Belrose et al. (2023): each layer
gets a learned affine translator that maps hidden states to a
"more final" representation before unembedding.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .base import BaseLens
from .types import LayerId, canonical_layer_id


class TunedLens(BaseLens):
    """
    Tuned lens with per-layer residual translators.
    
    For each layer, applies:
        h' = h + translator(h)
        logits = unembed(h')
    
    The translators are initialized to zero so the lens starts
    as an identity mapping (equivalent to logit lens).
    
    Parameters
    ----------
    layer_ids : Iterable[LayerId]
        Layer identifiers to create translators for.
    hidden_size : int
        Dimension of hidden states.
    unembed : nn.Module
        Unembedding module.
    vocab_size : Optional[int]
        Vocabulary size (inferred from unembed if not provided).
    bias : bool
        Whether translators have bias terms.
    init_identity : bool
        If True, initialize translators to zero (identity mapping).
        
    Notes
    -----
    Parameter count scales as O(num_layers * hidden_size^2), which
    can be prohibitive for large models. See LoRALens for a
    parameter-efficient alternative.
    """

    def __init__(
        self,
        layer_ids: Iterable[LayerId],
        hidden_size: int,
        unembed: nn.Module,
        vocab_size: Optional[int] = None,
        bias: bool = True,
        init_identity: bool = True,
    ) -> None:
        # Infer vocab size
        if vocab_size is None:
            if hasattr(unembed, "vocab_size"):
                vocab_size = unembed.vocab_size
            elif isinstance(unembed, nn.Linear):
                vocab_size = unembed.out_features
            else:
                raise ValueError("Cannot infer vocab_size; provide it explicitly.")
        
        super().__init__(vocab_size=vocab_size)
        
        self.hidden_size = hidden_size
        self.unembed = unembed
        
        # Freeze unembed
        for p in self.unembed.parameters():
            p.requires_grad = False
        
        # Create per-layer translators
        self._layer_ids = [canonical_layer_id(lid) for lid in layer_ids]
        
        self.translators = nn.ModuleDict({
            lid: nn.Linear(hidden_size, hidden_size, bias=bias)
            for lid in self._layer_ids
        })
        
        if init_identity:
            self._init_identity()

    def _init_identity(self) -> None:
        """Initialize translators to zero (identity mapping)."""
        for translator in self.translators.values():
            nn.init.zeros_(translator.weight)
            if translator.bias is not None:
                nn.init.zeros_(translator.bias)

    @property
    def layer_ids(self) -> List[str]:
        """List of layer IDs."""
        return list(self._layer_ids)

    def compute_logits(
        self,
        activations: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> torch.Tensor:
        """
        Apply translator and unembed.
        
        Parameters
        ----------
        activations : torch.Tensor
            Hidden states of shape [batch, seq, hidden].
        layer : LayerId
            Layer identifier (required).
        """
        if layer is None:
            raise ValueError("TunedLens requires a layer argument.")
        
        lid = canonical_layer_id(layer)
        if lid not in self.translators:
            raise KeyError(
                f"Layer {layer!r} not found. Available: {self.layer_ids}"
            )
        
        batch, seq, hidden = activations.shape
        
        # Apply translator with residual connection
        translator = self.translators[lid]
        flat = activations.view(batch * seq, hidden)
        flat = flat + translator(flat)  # Residual
        
        # Unembed
        flat_logits = self.unembed(flat)
        
        return flat_logits.view(batch, seq, -1)

    def compute_logits_subset(
        self,
        activations: torch.Tensor,
        vocab_indices: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> torch.Tensor:
        """
        Compute logits for a SUBSET of vocabulary only.
        
        Memory efficient: Uses fused CUDA kernel to avoid materializing
        [N, k, d] or [N, V] intermediate tensors.
        
        Parameters
        ----------
        activations : torch.Tensor
            Hidden states [batch, seq, hidden].
        vocab_indices : torch.Tensor
            Indices to compute, [batch, seq, k] or [k].
        layer : LayerId
            Layer identifier.
            
        Returns
        -------
        torch.Tensor
            Logits [batch, seq, k] for requested indices only.
        """
        if layer is None:
            raise ValueError("TunedLens requires a layer argument.")
        
        lid = canonical_layer_id(layer)
        if lid not in self.translators:
            raise KeyError(f"Layer {layer!r} not found.")
        
        batch, seq, hidden = activations.shape
        
        # Apply translator with residual
        translator = self.translators[lid]
        flat = activations.view(batch * seq, hidden)
        flat = flat + translator(flat)  # [B*T, d]
        
        # Apply layer norm if present
        if hasattr(self.unembed, 'layer_norm') and self.unembed.layer_norm is not None:
            flat = self.unembed.layer_norm(flat)
        elif hasattr(self.unembed, 'ln_f') and self.unembed.ln_f is not None:
            flat = self.unembed.ln_f(flat)
        
        # Get unembed weight matrix
        if hasattr(self.unembed, 'lm_head'):
            W = self.unembed.lm_head.weight  # [V, d]
            b = self.unembed.lm_head.bias    # [V] or None
        elif hasattr(self.unembed, 'weight'):
            W = self.unembed.weight
            b = getattr(self.unembed, 'bias', None)
        else:
            # Fallback to full computation
            full_logits = self.unembed(flat).view(batch, seq, -1)
            if vocab_indices.dim() == 1:
                return full_logits[..., vocab_indices]
            return torch.gather(full_logits, -1, vocab_indices)
        
        # Efficient subset computation
        if vocab_indices.dim() == 1:
            # [k] shared indices - simple case, use matmul
            k = vocab_indices.shape[0]
            W_subset = W[vocab_indices]  # [k, d]
            logits_flat = flat @ W_subset.T  # [B*T, k]
            if b is not None:
                logits_flat = logits_flat + b[vocab_indices]
            return logits_flat.view(batch, seq, k)
        else:
            # [batch, seq, k] per-position indices
            # Use fused indexed_logits kernel to avoid [N, V] materialization
            from loralens.ops import indexed_logits, indexed_logits_available
            
            N = batch * seq
            k = vocab_indices.shape[-1]
            idx_flat = vocab_indices.view(N, k).contiguous()
            
            if indexed_logits_available() and flat.is_cuda:
                # Use fused kernel - avoids [N, V] entirely!
                logits_flat = indexed_logits(
                    H=flat.contiguous(),
                    W=W.contiguous(),
                    idx=idx_flat,
                    bias=b,
                )
            else:
                # Fallback: full matmul then gather (memory inefficient)
                full_logits = flat @ W.T  # [N, V] - this is what we want to avoid
                if b is not None:
                    full_logits = full_logits + b
                logits_flat = torch.gather(full_logits, dim=1, index=idx_flat.long())
            
            return logits_flat.view(batch, seq, k)
