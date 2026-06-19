"""Activation-site planning for lens training."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Literal

import torch.nn as nn


ActivationSitePreset = Literal[
    "residual",
    "llama_expanded",
    "gpt2_expanded",
    "gpt2_attention",
]


@dataclass(frozen=True)
class ActivationSitePlan:
    """Ordered activation sites and how to collect them."""

    site_ids: List[str]
    hidden_state_sources: Dict[str, int] = field(default_factory=dict)
    custom_hooks: Dict[str, str] = field(default_factory=dict)
    custom_sources: Dict[str, str] = field(default_factory=dict)

    @property
    def site_count(self) -> int:
        return len(self.site_ids)


def build_activation_site_plan(
    model: nn.Module,
    *,
    num_layers: int,
    preset: ActivationSitePreset,
) -> ActivationSitePlan:
    """Build an ordered activation-site plan for ``model``."""
    if preset == "residual":
        return _build_residual_plan(num_layers)

    if preset == "llama_expanded":
        if not _is_llama_like(model):
            raise ValueError(
                "activation_site_preset='llama_expanded' requires a Llama-like model "
                "with model.layers, input_layernorm, post_attention_layernorm, "
                "self_attn.o_proj, and mlp.down_proj"
            )
        return _build_llama_expanded_plan(num_layers)

    if preset == "gpt2_expanded":
        if not _is_gpt2_like(model):
            raise ValueError(
                "activation_site_preset='gpt2_expanded' requires a GPT-2-like model "
                "with transformer.h, ln_1, attn.c_proj, ln_2, mlp.c_proj, and ln_f"
            )
        return _build_gpt2_expanded_plan(num_layers)

    if preset == "gpt2_attention":
        if not _is_gpt2_like(model):
            raise ValueError(
                "activation_site_preset='gpt2_attention' requires a GPT-2-like model "
                "with transformer.h, ln_1, attn.c_proj, ln_2, mlp.c_proj, and ln_f"
            )
        return _build_gpt2_attention_plan(num_layers)

    raise ValueError(f"Unknown activation site preset: {preset!r}")


def adapt_activation_site_plan_for_model(
    model: nn.Module,
    plan: ActivationSitePlan,
) -> ActivationSitePlan:
    """Rewrite custom hook paths to match the concrete wrapped model."""
    if not plan.custom_hooks:
        return plan

    available_modules = {name for name, _ in model.named_modules()}
    has_fsdp_wrapped_layers = any("._fsdp_wrapped_module" in name for name in available_modules)
    remapped_hooks = {
        hook_name: _resolve_module_path(
            path,
            available_modules,
            force_fsdp_layer_rewrite=has_fsdp_wrapped_layers,
        )
        for hook_name, path in plan.custom_hooks.items()
    }

    if remapped_hooks == plan.custom_hooks:
        return plan

    return ActivationSitePlan(
        site_ids=plan.site_ids,
        hidden_state_sources=plan.hidden_state_sources,
        custom_hooks=remapped_hooks,
        custom_sources=plan.custom_sources,
    )


def _build_residual_plan(num_layers: int) -> ActivationSitePlan:
    site_ids = [str(idx) for idx in range(num_layers)]
    hidden_state_sources = {str(idx): idx for idx in range(num_layers)}
    return ActivationSitePlan(site_ids=site_ids, hidden_state_sources=hidden_state_sources)


def _build_llama_expanded_plan(num_layers: int) -> ActivationSitePlan:
    site_ids: List[str] = ["embed_out"]
    hidden_state_sources: Dict[str, int] = {"embed_out": 0}
    custom_hooks: Dict[str, str] = {}
    custom_sources: Dict[str, str] = {}

    for idx in range(num_layers):
        prefix = f"L{idx:02d}"

        attn_in = f"{prefix}.attn_in"
        custom_hooks[attn_in] = f"model.layers.{idx}.input_layernorm"
        custom_sources[attn_in] = attn_in
        site_ids.append(attn_in)

        attn_out = f"{prefix}.attn_out"
        custom_hooks[attn_out] = f"model.layers.{idx}.self_attn.o_proj"
        custom_sources[attn_out] = attn_out
        site_ids.append(attn_out)

        resid_mid = f"{prefix}.resid_mid"
        resid_mid_key = f"{resid_mid}_input"
        custom_hooks[resid_mid_key] = f"model.layers.{idx}.post_attention_layernorm"
        custom_sources[resid_mid] = resid_mid_key
        site_ids.append(resid_mid)

        mlp_in = f"{prefix}.mlp_in"
        custom_hooks[mlp_in] = f"model.layers.{idx}.post_attention_layernorm"
        custom_sources[mlp_in] = mlp_in
        site_ids.append(mlp_in)

        mlp_out = f"{prefix}.mlp_out"
        custom_hooks[mlp_out] = f"model.layers.{idx}.mlp.down_proj"
        custom_sources[mlp_out] = mlp_out
        site_ids.append(mlp_out)

        resid_post = f"{prefix}.resid_post"
        hidden_state_sources[resid_post] = idx + 1
        site_ids.append(resid_post)

    final_norm_out = "final_norm_out"
    custom_hooks[final_norm_out] = "model.norm"
    custom_sources[final_norm_out] = final_norm_out
    site_ids.append(final_norm_out)

    return ActivationSitePlan(
        site_ids=site_ids,
        hidden_state_sources=hidden_state_sources,
        custom_hooks=custom_hooks,
        custom_sources=custom_sources,
    )


def _build_gpt2_expanded_plan(num_layers: int) -> ActivationSitePlan:
    site_ids: List[str] = ["embed_out"]
    hidden_state_sources: Dict[str, int] = {"embed_out": 0}
    custom_hooks: Dict[str, str] = {}
    custom_sources: Dict[str, str] = {}

    for idx in range(num_layers):
        prefix = f"L{idx:02d}"
        block = f"transformer.h.{idx}"

        attn_in = f"{prefix}.attn_in"
        custom_hooks[attn_in] = f"{block}.ln_1"
        custom_sources[attn_in] = attn_in
        site_ids.append(attn_in)

        attn_out = f"{prefix}.attn_out"
        custom_hooks[attn_out] = f"{block}.attn.c_proj"
        custom_sources[attn_out] = attn_out
        site_ids.append(attn_out)

        resid_mid = f"{prefix}.resid_mid"
        resid_mid_key = f"{resid_mid}_input"
        custom_hooks[resid_mid_key] = f"{block}.ln_2"
        custom_sources[resid_mid] = resid_mid_key
        site_ids.append(resid_mid)

        mlp_in = f"{prefix}.mlp_in"
        custom_hooks[mlp_in] = f"{block}.ln_2"
        custom_sources[mlp_in] = mlp_in
        site_ids.append(mlp_in)

        mlp_out = f"{prefix}.mlp_out"
        custom_hooks[mlp_out] = f"{block}.mlp.c_proj"
        custom_sources[mlp_out] = mlp_out
        site_ids.append(mlp_out)

        resid_post = f"{prefix}.resid_post"
        hidden_state_sources[resid_post] = idx + 1
        site_ids.append(resid_post)

    final_norm_out = "final_norm_out"
    custom_hooks[final_norm_out] = "transformer.ln_f"
    custom_sources[final_norm_out] = final_norm_out
    site_ids.append(final_norm_out)

    return ActivationSitePlan(
        site_ids=site_ids,
        hidden_state_sources=hidden_state_sources,
        custom_hooks=custom_hooks,
        custom_sources=custom_sources,
    )


def _build_gpt2_attention_plan(num_layers: int) -> ActivationSitePlan:
    site_ids: List[str] = []
    custom_hooks: Dict[str, str] = {}
    custom_sources: Dict[str, str] = {}

    for idx in range(num_layers):
        prefix = f"L{idx:02d}"
        block = f"transformer.h.{idx}"

        attn_in = f"{prefix}.attn_in"
        custom_hooks[attn_in] = f"{block}.ln_1"
        custom_sources[attn_in] = attn_in
        site_ids.append(attn_in)

        attn_out = f"{prefix}.attn_out"
        custom_hooks[attn_out] = f"{block}.attn.c_proj"
        custom_sources[attn_out] = attn_out
        site_ids.append(attn_out)

    return ActivationSitePlan(
        site_ids=site_ids,
        custom_hooks=custom_hooks,
        custom_sources=custom_sources,
    )


def _is_llama_like(model: nn.Module) -> bool:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None or len(layers) == 0:
        return False

    layer0 = layers[0]
    return (
        hasattr(base, "norm")
        and hasattr(layer0, "input_layernorm")
        and hasattr(layer0, "post_attention_layernorm")
        and hasattr(layer0, "self_attn")
        and hasattr(layer0.self_attn, "o_proj")
        and hasattr(layer0, "mlp")
        and hasattr(layer0.mlp, "down_proj")
    )


def _is_gpt2_like(model: nn.Module) -> bool:
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if blocks is None or len(blocks) == 0:
        return False

    block0 = blocks[0]
    return (
        hasattr(transformer, "ln_f")
        and hasattr(block0, "ln_1")
        and hasattr(block0, "attn")
        and hasattr(block0.attn, "c_proj")
        and hasattr(block0, "ln_2")
        and hasattr(block0, "mlp")
        and hasattr(block0.mlp, "c_proj")
    )


def _resolve_module_path(
    path: str,
    available_modules: set[str],
    *,
    force_fsdp_layer_rewrite: bool = False,
) -> str:
    """Resolve a logical module path against a wrapped model."""
    if path in available_modules:
        return path

    fsdp_match = re.match(r"^(model\.layers\.\d+)(\..+)$", path)
    if fsdp_match is not None:
        candidate = f"{fsdp_match.group(1)}._fsdp_wrapped_module{fsdp_match.group(2)}"
        if force_fsdp_layer_rewrite or candidate in available_modules:
            return candidate

    return path
