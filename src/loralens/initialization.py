# src/loralens/initialization.py
"""Optional data-driven initialization for LoRA lenses."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from loralens.lenses.lora_lens import LoRALens
from loralens.training.activation_sites import ActivationSitePlan, get_model_input_device
from loralens.training.distributed import DDPState, all_reduce_sum
from loralens.training.model_shard import ModelShardState


logger = logging.getLogger(__name__)


InitMode = Literal["default_lora", "mean_shift", "ridge_svd"]
LambdaScale = Literal["trace_xxt_over_d", "absolute"]
StatsDType = Literal["float32", "float64"]
SVDMetric = Literal["residual", "unembed"]
Normalization = Literal["none", "per_dim_std"]


@dataclass
class LoRAInitConfig:
    """Configuration for optional LoRA initialization."""

    mode: InitMode = "default_lora"
    calibration_tokens: int = 50_000
    ridge_lambda: float = 1e-3
    ridge_lambda_scale: LambdaScale = "trace_xxt_over_d"
    stats_dtype: StatsDType = "float32"
    jitter: float = 1e-6
    svd_metric: SVDMetric = "residual"
    normalization: Normalization = "none"


@dataclass
class _LayerStats:
    n: torch.Tensor
    sum_x: torch.Tensor
    sum_y: torch.Tensor
    sum_x2: torch.Tensor
    sum_y2: torch.Tensor
    xtx: torch.Tensor
    xty: torch.Tensor


def initialize_lora_lens(
    *,
    lens: LoRALens,
    model: torch.nn.Module,
    collector,
    activation_site_plan: ActivationSitePlan,
    dataloader_factory: Callable[[], DataLoader],
    ddp_state: DDPState,
    lens_device: torch.device,
    shard_state: Optional[ModelShardState],
    config: LoRAInitConfig,
    token_shift: int = 0,
) -> None:
    """Apply an optional data-driven initialization to a LoRA lens."""
    if config.mode == "default_lora":
        return
    if config.calibration_tokens <= 0:
        raise ValueError("lora_init_calibration_tokens must be positive")

    stats_dtype = _resolve_stats_dtype(config.stats_dtype)
    stats = _collect_calibration_stats(
        model=model,
        collector=collector,
        activation_site_plan=activation_site_plan,
        dataloader_factory=dataloader_factory,
        ddp_state=ddp_state,
        lens_device=lens_device,
        shard_state=shard_state,
        token_budget=config.calibration_tokens,
        stats_dtype=stats_dtype,
        token_shift=token_shift,
    )
    svd_metric = None
    if config.mode == "ridge_svd" and config.svd_metric == "unembed":
        svd_metric = _unembed_metric_factors(lens.unembed, dtype=stats_dtype)

    for layer_id in lens.layer_ids:
        if layer_id not in stats:
            raise KeyError(f"Missing calibration stats for LoRA layer {layer_id!r}")
        _initialize_projection(
            lens=lens,
            layer_id=layer_id,
            stats=stats[layer_id],
            mode=config.mode,
            ridge_lambda=config.ridge_lambda,
            ridge_lambda_scale=config.ridge_lambda_scale,
            jitter=config.jitter,
            svd_metric_name=config.svd_metric,
            svd_metric=svd_metric,
            normalization=config.normalization,
        )


def _resolve_stats_dtype(name: StatsDType) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported lora_init_stats_dtype: {name!r}")


def _new_layer_stats(hidden_size: int, dtype: torch.dtype) -> _LayerStats:
    return _LayerStats(
        n=torch.zeros((), dtype=dtype),
        sum_x=torch.zeros(hidden_size, dtype=dtype),
        sum_y=torch.zeros(hidden_size, dtype=dtype),
        sum_x2=torch.zeros(hidden_size, dtype=dtype),
        sum_y2=torch.zeros(hidden_size, dtype=dtype),
        xtx=torch.zeros(hidden_size, hidden_size, dtype=dtype),
        xty=torch.zeros(hidden_size, hidden_size, dtype=dtype),
    )


def _collect_calibration_stats(
    *,
    model: torch.nn.Module,
    collector,
    activation_site_plan: ActivationSitePlan,
    dataloader_factory: Callable[[], DataLoader],
    ddp_state: DDPState,
    lens_device: torch.device,
    shard_state: Optional[ModelShardState],
    token_budget: int,
    stats_dtype: torch.dtype,
    token_shift: int,
) -> dict[str, _LayerStats]:
    stats: dict[str, _LayerStats] = {}
    data_iter = iter(dataloader_factory())
    global_tokens = 0.0
    collector.attach()
    model_was_training = model.training
    model.eval()

    try:
        while global_tokens < token_budget:
            batch, data_iter = _next_batch(data_iter, dataloader_factory)
            input_ids = batch["input_ids"].to(lens_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(lens_device, non_blocking=True)

            local_tokens = attention_mask.sum().to(device=lens_device, dtype=torch.float32)
            global_batch_tokens = all_reduce_sum(local_tokens, ddp_state).item()
            global_tokens += global_batch_tokens

            if shard_state is not None and shard_state.enabled:
                model_input_device = get_model_input_device(model)
                input_ids_model = input_ids.to(model_input_device, non_blocking=True)
                attn_model = attention_mask.to(model_input_device, non_blocking=True)
            else:
                input_ids_model = input_ids
                attn_model = attention_mask

            with torch.no_grad():
                activations = collector.collect_with_grad(
                    input_ids=input_ids_model,
                    attention_mask=attn_model,
                )

            hidden_states = activations.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states during calibration")

            ordered = activation_site_plan.resolve_sites(hidden_states, activations.custom)
            site_map = dict(ordered)
            y_full = hidden_states[-1]

            if shard_state is not None and shard_state.enabled:
                y_full = y_full.to(lens_device, non_blocking=True)
                attention_mask = attention_mask.to(lens_device, non_blocking=True)

            if token_shift > 0:
                y = y_full[:, token_shift:, :]
                mask = attention_mask[:, token_shift:].bool()
                x_slice = slice(None, -token_shift)
            else:
                y = y_full
                mask = attention_mask.bool()
                x_slice = slice(None)

            for site_id in activation_site_plan.site_ids:
                x_full = site_map[site_id]
                if x_full.device != lens_device:
                    x_full = x_full.to(lens_device, non_blocking=True)
                x = x_full[:, x_slice, :]
                _accumulate_layer_stats(stats, site_id, x, y, mask, stats_dtype)

            del activations, hidden_states, ordered, site_map, y_full
            del input_ids, attention_mask
    finally:
        collector.detach()
        if model_was_training:
            model.train()

    for layer_stats in stats.values():
        _all_reduce_layer_stats(layer_stats, ddp_state)

    if ddp_state.is_main:
        logger.info(
            "LoRA calibration collected approximately %.0f global tokens",
            global_tokens,
        )
    return stats


def _next_batch(
    data_iter: Iterator,
    dataloader_factory: Callable[[], DataLoader],
) -> tuple[object, Iterator]:
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader_factory())
        return next(data_iter), data_iter


def _accumulate_layer_stats(
    stats: dict[str, _LayerStats],
    site_id: str,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    stats_dtype: torch.dtype,
) -> None:
    if x.shape != y.shape:
        raise ValueError(
            f"Calibration shape mismatch for {site_id!r}: X={tuple(x.shape)} Y={tuple(y.shape)}"
        )
    if mask.shape != x.shape[:2]:
        raise ValueError(
            f"Calibration mask shape mismatch for {site_id!r}: "
            f"mask={tuple(mask.shape)} X={tuple(x.shape)}"
        )

    hidden_size = x.size(-1)
    if site_id not in stats:
        stats[site_id] = _new_layer_stats(hidden_size, stats_dtype)

    x_flat = x[mask].to(dtype=stats_dtype)
    y_flat = y[mask].to(dtype=stats_dtype)
    if x_flat.numel() == 0:
        return

    layer_stats = stats[site_id]
    if layer_stats.sum_x.numel() != hidden_size:
        raise ValueError(f"Hidden-size mismatch for calibration site {site_id!r}")

    layer_stats.n += x_flat.size(0)
    layer_stats.sum_x += x_flat.sum(dim=0).cpu()
    layer_stats.sum_y += y_flat.sum(dim=0).cpu()
    layer_stats.sum_x2 += x_flat.square().sum(dim=0).cpu()
    layer_stats.sum_y2 += y_flat.square().sum(dim=0).cpu()
    layer_stats.xtx += (x_flat.T @ x_flat).cpu()
    layer_stats.xty += (x_flat.T @ y_flat).cpu()


def _all_reduce_layer_stats(layer_stats: _LayerStats, ddp_state: DDPState) -> None:
    if not ddp_state.enabled:
        return
    for value in (
        layer_stats.n,
        layer_stats.sum_x,
        layer_stats.sum_y,
        layer_stats.sum_x2,
        layer_stats.sum_y2,
        layer_stats.xtx,
        layer_stats.xty,
    ):
        device_value = value.to(ddp_state.device)
        dist.all_reduce(device_value, op=dist.ReduceOp.SUM)
        value.copy_(device_value.cpu())


def _initialize_projection(
    *,
    lens: LoRALens,
    layer_id: str,
    stats: _LayerStats,
    mode: InitMode,
    ridge_lambda: float,
    ridge_lambda_scale: LambdaScale,
    jitter: float,
    svd_metric_name: SVDMetric,
    svd_metric: Optional[tuple[torch.Tensor, torch.Tensor]],
    normalization: Normalization,
) -> None:
    projection = lens.projections[lens._module_keys[layer_id]]
    if stats.n.item() <= 1:
        raise ValueError(f"Not enough calibration tokens for layer {layer_id!r}")

    mean_x = stats.sum_x / stats.n
    mean_y = stats.sum_y / stats.n
    hidden_size = mean_x.numel()

    with torch.no_grad():
        bias = mean_y - mean_x
        projection.bias.copy_(bias.to(projection.bias.device, projection.bias.dtype))

    if mode == "mean_shift":
        return
    if mode != "ridge_svd":
        raise ValueError(f"Unknown LoRA init mode: {mode!r}")

    xtx = stats.xtx - stats.n * torch.outer(mean_x, mean_x)
    xty = stats.xty - stats.n * torch.outer(mean_x, mean_y)
    eye = torch.eye(hidden_size, dtype=stats.xtx.dtype)

    if normalization == "none":
        prior = eye
        left_scale = None
        right_scale = None
    elif normalization == "per_dim_std":
        var_x = (stats.sum_x2 / stats.n - mean_x.square()).clamp_min(1e-12)
        var_y = (stats.sum_y2 / stats.n - mean_y.square()).clamp_min(1e-12)
        std_x = var_x.sqrt()
        std_y = var_y.sqrt()
        xtx = xtx / torch.outer(std_x, std_x)
        xty = xty / torch.outer(std_x, std_y)
        prior = torch.diag(std_x / std_y)
        left_scale = std_x
        right_scale = std_y
    else:
        raise ValueError(f"Unknown LoRA init normalization: {normalization!r}")

    if ridge_lambda_scale == "trace_xxt_over_d":
        lambda_value = ridge_lambda * (torch.trace(xtx) / hidden_size).clamp_min(1e-12)
    elif ridge_lambda_scale == "absolute":
        lambda_value = torch.as_tensor(ridge_lambda, dtype=xtx.dtype)
    else:
        raise ValueError(f"Unknown ridge lambda scale: {ridge_lambda_scale!r}")

    lhs = xtx + lambda_value * eye
    rhs = xty + lambda_value * prior
    solution = _solve_with_jitter(lhs, rhs, jitter=jitter)
    if left_scale is not None and right_scale is not None:
        solution = solution * (right_scale.unsqueeze(0) / left_scale.unsqueeze(1))
    delta = solution - eye
    if svd_metric_name not in ("residual", "unembed"):
        raise ValueError(f"Unknown LoRA init SVD metric: {svd_metric_name!r}")
    _copy_truncated_delta_to_lora(projection, delta, metric=svd_metric)

    with torch.no_grad():
        bias = mean_y - mean_x @ solution
        projection.bias.copy_(bias.to(projection.bias.device, projection.bias.dtype))


def _solve_with_jitter(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    jitter: float,
) -> torch.Tensor:
    eye = torch.eye(lhs.size(0), dtype=lhs.dtype)
    current = 0.0
    last_error: Optional[RuntimeError] = None
    for attempt in range(6):
        try:
            if current == 0.0:
                return torch.linalg.solve(lhs, rhs)
            return torch.linalg.solve(lhs + current * eye, rhs)
        except RuntimeError as exc:
            last_error = exc
            current = jitter if attempt == 0 else current * 10.0
    raise RuntimeError("Ridge solve failed even with jitter") from last_error


def _unembed_metric_factors(
    unembed: torch.nn.Module,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return C and C^-1 for the right metric induced by the LM head."""
    if not hasattr(unembed, "lm_head") or not hasattr(unembed.lm_head, "weight"):
        raise ValueError("lora_init_svd_metric='unembed' requires unembed.lm_head.weight")

    weight = unembed.lm_head.weight.detach().to(device="cpu", dtype=dtype)
    gram = weight.T @ weight
    gram = gram / max(1, weight.size(0))
    gram = (gram + gram.T) * 0.5

    evals, evecs = torch.linalg.eigh(gram)
    floor = evals.max().clamp_min(1e-12) * 1e-6
    evals = evals.clamp_min(floor)
    sqrt_evals = evals.sqrt()

    c = (evecs * sqrt_evals.unsqueeze(0)) @ evecs.T
    c_inv = (evecs * (1.0 / sqrt_evals).unsqueeze(0)) @ evecs.T
    return c, c_inv


def _copy_truncated_delta_to_lora(
    projection,
    delta: torch.Tensor,
    metric: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> None:
    rank = projection.r
    alpha = projection.alpha
    if rank <= 0 or alpha == 0:
        raise ValueError("LoRA ridge_svd init requires positive rank and nonzero alpha")

    if metric is not None:
        c, c_inv = metric
        delta_for_svd = delta @ c
    else:
        c_inv = None
        delta_for_svd = delta

    u, s, vh = torch.linalg.svd(delta_for_svd, full_matrices=False)
    r_eff = min(rank, s.numel())
    u_r = u[:, :r_eff]
    s_root = s[:r_eff].clamp_min(0).sqrt()
    vh_r = vh[:r_eff, :]

    if c_inv is not None:
        delta = (u_r * s[:r_eff].unsqueeze(0)) @ vh_r @ c_inv
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r_eff = min(rank, s.numel())
        u_r = u[:, :r_eff]
        s_root = s[:r_eff].clamp_min(0).sqrt()
        vh_r = vh[:r_eff, :]

    # PyTorch Linear uses y = x @ weight.T. The LoRA update is
    # x @ (B.weight @ A.weight).T * alpha/r, so transpose the row-space SVD.
    scale = (rank / alpha) ** 0.5
    b_weight = vh_r.T * s_root.unsqueeze(0) * scale
    a_weight = s_root.unsqueeze(1) * u_r.T * scale

    if r_eff < rank:
        b_padded = torch.zeros(delta.size(0), rank, dtype=delta.dtype)
        a_padded = torch.zeros(rank, delta.size(1), dtype=delta.dtype)
        b_padded[:, :r_eff] = b_weight
        a_padded[:r_eff, :] = a_weight
        b_weight = b_padded
        a_weight = a_padded

    with torch.no_grad():
        projection.lora_B.weight.copy_(
            b_weight.to(projection.lora_B.weight.device, projection.lora_B.weight.dtype)
        )
        projection.lora_A.weight.copy_(
            a_weight.to(projection.lora_A.weight.device, projection.lora_A.weight.dtype)
        )
