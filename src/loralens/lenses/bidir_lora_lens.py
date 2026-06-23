# src/loralens/lenses/bidir_lora_lens.py
"""
BidirLoRALens — LoRA lens with an orthogonality penalty for the write direction.

Inherits all read functionality from LoRALens (compute_logits, checkpointing,
subset KL, etc.). Adds one write-direction regulariser that uses the existing
projection weights — no new parameters.

Read direction (inherited):
    h_L → h_L @ M.T + bias → unembed → logits
    where M.T = I + lora_A.weight.T @ lora_B.weight.T * scaling

Orthogonality penalty:
    ||h - h @ M.T @ M||^2  →  pushes M.T^{-1} ≈ M (transpose ≈ inverse), so the
    write direction is just the transpose of the read map. Cheap: two rank-r
    ops, no inversion.
"""

from __future__ import annotations

import torch

from .lora_lens import LoRALens
from .types import LayerId, canonical_layer_id


class BidirLoRALens(LoRALens):
    """
    Bidirectional LoRA lens.

    All read methods are inherited unchanged from LoRALens — existing
    checkpoints and evaluation code work without modification.

    New methods
    -----------
    compute_orthogonality_penalty(activations, layer)
        Scalar regulariser ``||h - h @ M.T @ M||^2``.
        Encourages M.T to be orthogonal (transpose ≈ inverse).
        Free to compute alongside the read loss.
    """

    def compute_orthogonality_penalty(
        self,
        activations: torch.Tensor,
        layer: LayerId,
    ) -> torch.Tensor:
        """
        Orthogonality penalty: ``||h - h @ M.T @ M||^2``.

        Encourages M.T to be orthogonal so that (M.T)^{-1} ≈ M (the plain
        transpose), making the write direction free to compute at inference.

        Derivation:
            M.T = I + A_w.T @ B_w.T * s
            M   = I + B_w * s @ A_w          (transpose of the rank-r update)

            h @ M.T @ M = (h @ M.T) @ M
                Mh   = h  + h  @ A_w.T @ B_w.T * s   (forward pass)
                MTMh = Mh + Mh @ B_w * s  @ A_w      (second pass, transposed map)

        Cost: two rank-r multiplies, O(N·d·r). No inversion required.

        Args:
            activations: [batch, seq, d] or [batch, d].
            layer: Layer identifier.

        Returns:
            Scalar penalty (mean over all tokens).
        """
        lid = canonical_layer_id(layer)
        module_key = self._module_keys.get(lid)
        if module_key is None or module_key not in self.projections:
            raise KeyError(f"Layer {layer!r} not found.")

        proj = self.projections[module_key]
        A_w = proj.lora_A.weight   # [r, d]
        B_w = proj.lora_B.weight   # [d, r]
        s = proj.scaling

        if activations.dim() == 3:
            b, t, d = activations.shape
            h = activations.reshape(b * t, d)
        else:
            h = activations

        h_f = h.float()
        A_f = A_w.float()
        B_f = B_w.float()
        dev = A_w.device

        # Disable AMP autocast so all matmuls stay in float32.
        with torch.autocast(device_type=dev.type, enabled=False):
            # h @ M.T  =  h + (h @ A_w.T) @ B_w.T * s
            hA = h_f @ A_f.T          # [N, r]:   h @ A_w.T   (A_w.T is [d, r])
            Mh = h_f + hA @ B_f.T * s # [N, d]:   h @ M.T

            # Mh @ M  =  Mh + (Mh @ B_w * s) @ A_w
            #   M = I + B_w * s @ A_w  (B_w is [d, r], A_w is [r, d])
            MhB = Mh @ B_f * s        # [N, r]:   Mh @ (B_w * s)
            MTMh = Mh + MhB @ A_f     # [N, d]:   Mh @ M

        # Normalize by ||h||² so the penalty is a dimensionless fraction of
        # hidden-state energy that isn't preserved — stays in [0, 1] regardless
        # of activation scale or LoRA initialization.
        h_energy = h_f.pow(2).mean().clamp(min=1e-8)
        return (h_f - MTMh).pow(2).mean() / h_energy
