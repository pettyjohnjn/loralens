from __future__ import annotations

import torch
import torch.nn as nn


class HFUnembed(nn.Module):
    """
    HF unembed that matches tuned-lens behavior for GPT-2-like models:
      logits = lm_head(ln_f(h)) if ln_f exists, else lm_head(h)
    """

    def __init__(self, model):
        super().__init__()
        self.ln_f = None

        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            self.ln_f = model.transformer.ln_f

        if hasattr(model, "lm_head"):
            self.lm_head = model.lm_head
        elif hasattr(model, "get_output_embeddings"):
            self.lm_head = model.get_output_embeddings()
        else:
            raise ValueError("Could not locate LM head (lm_head / output_embeddings).")

        if hasattr(self.lm_head, "out_features"):
            self.vocab_size = int(self.lm_head.out_features)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.ln_f is not None:
            h = self.ln_f(h)
        return self.lm_head(h)