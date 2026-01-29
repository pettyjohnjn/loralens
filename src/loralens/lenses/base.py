# src/loralens/lenses/base.py
"""Base class for all lenses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
import torch.nn as nn

from .types import LayerId, LensOutput, canonical_layer_id


class BaseLens(nn.Module, ABC):
    """
    Abstract base class for all lenses.
    
    A lens maps intermediate hidden states to vocabulary-space logits.
    Subclasses implement `compute_logits()` with their specific
    parameterization (identity, tuned, LoRA, etc.).
    
    For memory efficiency, lenses also support `compute_logits_subset()`
    which only computes logits for a subset of vocabulary indices.
    
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self._vocab_size

    @property
    def layer_ids(self) -> List[str]:
        """
        List of layer IDs this lens supports.
        
        Override in subclasses with per-layer parameters.
        """
        return []

    @abstractmethod
    def compute_logits(
        self,
        activations: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> torch.Tensor:
        """
        Map activations to FULL vocabulary logits.
        
        Parameters
        ----------
        activations : torch.Tensor
            Hidden states of shape [batch, seq, hidden].
        layer : Optional[LayerId]
            Layer identifier (required for per-layer lenses).
            
        Returns
        -------
        torch.Tensor
            Logits of shape [batch, seq, vocab].
        """
        raise NotImplementedError

    def compute_logits_subset(
        self,
        activations: torch.Tensor,
        vocab_indices: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> torch.Tensor:
        """
        Map activations to logits for a SUBSET of vocabulary only.
        
        This is the memory-efficient path that avoids materializing
        full [batch, seq, vocab] tensors. The default implementation
        falls back to full computation + gather, but subclasses should
        override with efficient implementations.
        
        Parameters
        ----------
        activations : torch.Tensor
            Hidden states of shape [batch, seq, hidden].
        vocab_indices : torch.Tensor
            Which vocab indices to compute. Shape [batch, seq, k] for
            per-position indices, or [k] for shared indices.
        layer : Optional[LayerId]
            Layer identifier.
            
        Returns
        -------
        torch.Tensor
            Logits of shape [batch, seq, k] for requested indices only.
        """
        # Default: compute full then gather (inefficient, but correct)
        # Subclasses should override with efficient subset computation
        full_logits = self.compute_logits(activations, layer=layer)
        
        if vocab_indices.dim() == 1:
            # [k] shared indices -> simple indexing
            return full_logits[..., vocab_indices]
        else:
            # [batch, seq, k] per-position indices -> gather
            return torch.gather(full_logits, dim=-1, index=vocab_indices)

    def compute_logits_subset_with_logsumexp(
        self,
        activations: torch.Tensor,
        vocab_indices: torch.Tensor,
        layer: Optional[LayerId] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute subset logits AND logsumexp without storing full [B,T,V].
        
        This is the key method for memory-efficient subset KL training.
        Returns both the subset logits and the logsumexp normalization
        constant, but computes logsumexp in a streaming fashion without
        materializing full vocabulary logits in memory.
        
        Parameters
        ----------
        activations : torch.Tensor
            Hidden states [batch, seq, hidden].
        vocab_indices : torch.Tensor
            Indices to compute [batch, seq, k] or [k].
        layer : Optional[LayerId]
            Layer identifier.
            
        Returns
        -------
        subset_logits : torch.Tensor
            Logits for requested indices [batch, seq, k].
        logsumexp : torch.Tensor
            Log-sum-exp over full vocab [batch, seq, 1].
        """
        # Default implementation: compute full, extract logsumexp, then discard
        # Subclasses can override with chunked/streaming computation
        full_logits = self.compute_logits(activations, layer=layer)
        logsumexp = full_logits.logsumexp(dim=-1, keepdim=True)
        
        if vocab_indices.dim() == 1:
            subset_logits = full_logits[..., vocab_indices]
        else:
            subset_logits = torch.gather(full_logits, dim=-1, index=vocab_indices)
        
        # Note: full_logits will be garbage collected after this returns
        # The backward pass will recompute as needed
        return subset_logits, logsumexp

    def forward(
        self,
        activations: torch.Tensor,
        layer: Optional[LayerId] = None,
        vocab_indices: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> LensOutput:
        """
        Forward pass through the lens.
        
        Parameters
        ----------
        activations : torch.Tensor
            Hidden states of shape [batch, seq, hidden].
        layer : Optional[LayerId]
            Layer identifier.
        vocab_indices : Optional[torch.Tensor]
            If provided, only compute logits for these vocabulary indices.
            This is the memory-efficient path for subset KL training.
        return_logits : bool
            Whether to include logits in output.
            
        Returns
        -------
        LensOutput
            Container with logits and optional extras.
        """
        logits = None
        if return_logits:
            if vocab_indices is not None:
                logits = self.compute_logits_subset(activations, vocab_indices, layer=layer)
            else:
                logits = self.compute_logits(activations, layer=layer)
        
        return LensOutput(logits=logits)

    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        trainable = self.num_trainable_parameters()
        total = self.num_parameters()
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"params={trainable:,}/{total:,})"
        )
