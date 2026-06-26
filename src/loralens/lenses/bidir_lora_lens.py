# src/loralens/lenses/bidir_lora_lens.py
"""
BidirLoRALens — Bidirectional LoRA lens supporting both read and write.

Inherits all read functionality from LoRALens (compute_logits, checkpointing,
subset KL, etc.).  Adds two write-direction methods that use the existing
projection weights — no new parameters.

Read direction (inherited):
    h_L → h_L @ M.T + bias → unembed → logits
    where M.T = I + lora_A.weight.T @ lora_B.weight.T * scaling

Write direction (new):
    v ∈ R^V → W_U.T @ v → (M.T)^{-1} via Woodbury → δ_L ∈ R^d
    Inject δ_L at site L to steer the downstream model toward predicting v.

Orthogonality penalty (approach A):
    ||h - h @ M.T @ M||^2  →  pushes M.T^{-1} ≈ M (transpose ≈ inverse)
    Cheap: two rank-r ops, no inversion.

Write injection (approach B):
    Uses the Woodbury identity to invert (M.T) exactly in O(d·r + r³) cost.
    Requires a suffix forward pass through the frozen model to train.
"""

from __future__ import annotations

import torch

from ._unembed import unembed_weight_bias
from .lora_lens import LoRALens, LoRAProjection
from .types import LayerId


class BidirLoRALens(LoRALens):
    """
    Bidirectional LoRA lens.

    All read methods are inherited unchanged from LoRALens — existing
    checkpoints and evaluation code work without modification.

    New methods
    -----------
    compute_write_injection(vocab_logits, layer)
        Given a vocabulary distribution, return the injection vector δ_L
        such that, when added to h_L, the lens predicts that distribution.
        Uses Woodbury inverse: O(d·r + r³), no d×d materialisation.

    compute_orthogonality_penalty(activations, layer)
        Scalar regulariser ||h - h @ M.T @ M||^2.
        Encourages M.T to be orthogonal (transpose ≈ inverse).
        Free to compute alongside the read loss.
    """

    def _get_W_U(self) -> torch.Tensor:
        """Return lm_head weight [V, d] from the frozen unembed module."""
        wb = unembed_weight_bias(self.unembed)
        if wb is None:
            raise AttributeError(
                "Cannot locate lm_head weight in unembed module. "
                "Expected self.unembed.lm_head.weight or self.unembed.weight."
            )
        return wb[0]

    def _woodbury_inverse_row(
        self,
        proj: LoRAProjection,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply (M.T)^{-1} to a batch of row vectors h [N, d].

        The LoRAProjection forward for row vectors is:
            f(h) = h @ M.T + bias
            M.T  = I + A_w.T @ B_w.T * s          [d, d]

        where A_w = lora_A.weight [r, d], B_w = lora_B.weight [d, r], s = scaling.

        Woodbury identity for M.T = I + U @ V  (U = A_w.T [d,r], V = B_w.T*s [r,d]):
            (M.T)^{-1} = I - U @ (I + V @ U)^{-1} @ V

        Applied to row vectors from the right:
            h @ (M.T)^{-1} = h - (h @ U) @ C^{-1} @ V
            C = I_r + V @ U  [r, r]

        Cost: O(N·d·r + r³)  vs  O(N·d²) for full d×d solve.
        """
        A_w = proj.lora_A.weight   # [r, d]
        B_w = proj.lora_B.weight   # [d, r]
        s   = proj.scaling
        r   = proj.r
        dev = A_w.device

        # Work in float32 for numerical stability of the r×r solve.
        # Disable AMP autocast: matmuls under autocast return bf16 even with
        # explicit .float() inputs, which breaks linalg.solve's dtype check.
        h_f = h.float()
        A_f = A_w.float()
        B_f = B_w.float()

        with torch.autocast(device_type=dev.type, enabled=False):
            # C = I + (B_w.T * s) @ A_w.T   [r, r]
            #   B_w.T has shape [r, d], A_w.T has shape [d, r]
            C = torch.eye(r, device=dev, dtype=torch.float32) + (B_f.T * s) @ A_f.T

            # h @ U  = h @ A_w.T   [N, r]
            hU = h_f @ A_f.T

            # hU @ C^{-1}: solve C.T @ X.T = hU.T  →  X = solve(C.T, hU.T).T
            hU_Cinv = torch.linalg.solve(C.T, hU.T).T   # [N, r]

            # h - hU_Cinv @ V  = h - hU_Cinv @ (B_w.T * s)   [N, d]
            result = h_f - hU_Cinv @ (B_f.T * s)

        return result.to(h.dtype)

    def compute_write_injection(
        self,
        vocab_logits: torch.Tensor,
        layer: LayerId,
    ) -> torch.Tensor:
        """
        Map a vocabulary distribution to a layer-L injection vector δ_L.

        Pipeline:
            v  →  W_U.T @ v  →  subtract bias  →  (M.T)^{-1}  →  δ_L

        When δ_L is added to h_L, the lens approximately predicts v:
            lens(h_L + δ_L, layer=layer) ≈ unembed(W_U.T @ v)

        This inverts the full affine map f(h) = h @ M.T + bias, so that
        f(δ_L) ≈ W_U.T @ v  (i.e. f^{-1}(W_U.T @ v) = δ_L).

        Args:
            vocab_logits: [batch, V] or [V] — target vocabulary distribution
                          (e.g. one-hot, softmax output, or uniform bag-of-words).
            layer: Layer identifier (must be in self.layer_ids).

        Returns:
            [batch, d]  injection vector, in the same dtype as lm_head.weight.
        """
        proj = self.projections[self._resolve_module_key(layer)]
        W_U  = self._get_W_U()    # [V, d], frozen

        if vocab_logits.dim() == 1:
            vocab_logits = vocab_logits.unsqueeze(0)   # [1, V]

        # Project from vocab space → residual space:  [batch, V] @ [V, d] → [batch, d]
        h_target = vocab_logits.float() @ W_U.float()

        # Subtract bias so Woodbury inverts the full affine (not just linear) map.
        h_target = h_target - proj.bias.float()

        delta = self._woodbury_inverse_row(proj, h_target)   # [batch, d]
        return delta.to(W_U.dtype)

    def compute_orthogonality_penalty(
        self,
        activations: torch.Tensor,
        layer: LayerId,
    ) -> torch.Tensor:
        """
        Approach A penalty: ||h - h @ M.T @ M||^2.

        Encourages M.T to be orthogonal so that (M.T)^{-1} ≈ M (the plain
        transpose), making the write direction free to compute at inference.

        Derivation:
            M.T = I + A_w.T @ B_w.T * s
            M   = I + B_w * s @ A_w          (transpose of the rank-r update)

            h @ M.T @ M = (h @ M.T) @ M
                Mh  = h  + h  @ A_w.T @ B_w.T * s   (forward pass)
                MTMh= Mh + Mh @ B_w * s  @ A_w       (second pass with transposed map)

        Cost: two rank-r multiplies, O(N·d·r).  No inversion required.

        Args:
            activations: [batch, seq, d] or [batch, d].
            layer: Layer identifier.

        Returns:
            Scalar penalty (mean over all tokens).
        """
        proj = self.projections[self._resolve_module_key(layer)]
        A_w  = proj.lora_A.weight   # [r, d]
        B_w  = proj.lora_B.weight   # [d, r]
        s    = proj.scaling

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
            MhB  = Mh @ B_f * s       # [N, r]:   Mh @ (B_w * s)
            MTMh = Mh + MhB @ A_f     # [N, d]:   Mh @ M

        # Normalize by ||h||² so the penalty is a dimensionless fraction of
        # hidden-state energy that isn't preserved — stays in [0, 1] regardless
        # of activation scale or LoRA initialization.
        h_energy = h_f.pow(2).mean().clamp(min=1e-8)
        return (h_f - MTMh).pow(2).mean() / h_energy
