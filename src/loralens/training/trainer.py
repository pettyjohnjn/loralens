# src/loralens/training/trainer.py
"""
LensTrainer - Orchestrates lens training with layer-wise backward
and memory-optimized KL computation.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
import time
from contextlib import nullcontext
from typing import Callable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from loralens.hooks import ActivationCollector
from loralens.losses import BaseLoss, SubsetKLLoss
from loralens.losses.shared_subset_kl import SharedSubsetKLLoss
from loralens.lenses import BaseLens

from .activation_sites import ActivationSitePlan
from .config import TrainConfig
from .distributed import DDPState, all_reduce_sum
from .amp import AMPContext
from .model_shard import ModelShardState, disabled_shard_state

logger = logging.getLogger(__name__)


def _get_model_input_device(model: nn.Module) -> torch.device:
    """
    Determine which device holds the model's embedding layer.

    When using ``device_map``, the embedding layer (where input_ids go)
    may live on a different GPU than the lens.
    """
    # HuggingFace models with device_map have this attribute
    if hasattr(model, "hf_device_map"):
        # The embedding module is typically the first entry
        for name, dev in model.hf_device_map.items():
            if "embed" in name.lower():
                return torch.device(dev) if isinstance(dev, (int, str)) else dev
        # Fallback: first entry in device_map
        first_dev = next(iter(model.hf_device_map.values()))
        return torch.device(first_dev) if isinstance(first_dev, (int, str)) else first_dev

    # Non-sharded: use first parameter's device
    return next(model.parameters()).device


def _masked_kl_logtarget(
    student_logits: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    attention_mask: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    KL(teacher || student) with teacher already as log-probs.

    Memory-efficient: keeps tensors in autocast dtype (bf16/fp16),
    only uses fp32 for scalar reductions.
    """
    # Stay in autocast dtype (bf16/fp16)
    s_logprobs = F.log_softmax(student_logits, dim=-1, dtype=student_logits.dtype)

    # KL per token in autocast dtype
    kl_token = (teacher_logprobs.exp() * (teacher_logprobs - s_logprobs)).sum(dim=-1)

    # Keep mask in same dtype to avoid fp32 tensors
    mask = attention_mask.to(dtype=kl_token.dtype)
    kl_token = kl_token * mask

    if reduction == "none":
        return kl_token
    elif reduction == "sum":
        # Only scalar goes to fp32
        return kl_token.sum(dtype=torch.float32)
    elif reduction == "mean":
        num = kl_token.sum(dtype=torch.float32)
        denom = mask.sum(dtype=torch.float32).clamp_min(1.0)
        return num / denom
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class LensTrainer:
    """Orchestrates lens training with memory-optimized layer-wise backward."""

    def __init__(
        self,
        model: nn.Module,
        lens: BaseLens,
        loss_fn: BaseLoss,
        collector: ActivationCollector,
        activation_site_plan: ActivationSitePlan,
        config: TrainConfig,
        ddp_state: DDPState,
        amp_ctx: AMPContext,
        shard_state: Optional[ModelShardState] = None,
        start_step: int = 0,
        start_total_tokens: int = 0,
    ) -> None:
        self.model = model
        self.lens = lens
        self.loss_fn = loss_fn
        self.collector = collector
        self.activation_site_plan = activation_site_plan
        self.config = config
        self.ddp_state = ddp_state
        self.amp_ctx = amp_ctx
        self.shard_state = shard_state or disabled_shard_state(ddp_state.device)
        self.start_step = start_step
        self.end_step = config.num_steps
        self.global_step = start_step
        self.total_tokens = start_total_tokens
        self._metrics_file = None
        self._metrics_writer = None

        # When model-parallel, the lens device may differ from ddp_state.device
        self._lens_device = (
            self.shard_state.lens_device
            if self.shard_state.enabled
            else ddp_state.device
        )

        # Determine if we use chunked KL (full vocab) vs subset KL
        self._use_subset_kl = isinstance(loss_fn, SubsetKLLoss)
        self._use_shared_subset_kl = isinstance(loss_fn, SharedSubsetKLLoss)
        self._use_chunked_kl = (
            config.loss_type == "kl"
            and config.kl_chunk_size is not None
            and config.kl_chunk_size > 0
            and not self._use_subset_kl
            and not self._use_shared_subset_kl
        )

    @staticmethod
    def _normalize_activation(value):
        """Unwrap hook payloads that come back as a singleton tuple/list."""
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise ValueError(f"Expected a single activation tensor, got {type(value)}")
            return value[0]
        return value

    def _collect_site_activations(
        self,
        hidden_states,
        custom_activations,
    ) -> list[tuple[str, torch.Tensor]]:
        """Resolve ordered activation sites from hidden states and custom hooks."""
        site_tensors = {}

        for site_id, hidden_idx in self.activation_site_plan.hidden_state_sources.items():
            site_tensors[site_id] = self._normalize_activation(hidden_states[hidden_idx])

        for site_id, custom_key in self.activation_site_plan.custom_sources.items():
            if custom_key not in custom_activations:
                raise KeyError(f"Missing custom activation {custom_key!r} for site {site_id!r}")
            site_tensors[site_id] = self._normalize_activation(custom_activations[custom_key])

        ordered_sites = []
        for site_id in self.activation_site_plan.site_ids:
            if site_id not in site_tensors:
                raise KeyError(f"Missing activation tensor for site {site_id!r}")
            ordered_sites.append((site_id, site_tensors[site_id]))

        return ordered_sites

    def train(
        self,
        dataloader_factory: Callable[[], DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        config = self.config
        ddp_state = self.ddp_state

        if ddp_state.is_main:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_config_csv()
            self._open_metrics_csv()

        lens = self.lens
        if ddp_state.enabled:
            lens_local_rank = self._lens_device.index or 0
            lens = DDP(
                lens,
                device_ids=[lens_local_rank],
                output_device=lens_local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        if ddp_state.is_main and self.start_step == 0 and self.config.save_initial_checkpoint:
            self._save_checkpoint(lens, 0)
        if self.start_step >= self.end_step:
            self._close_metrics_csv()
            return

        data_iter = iter(dataloader_factory())
        self.collector.attach()

        try:
            self._train_loop(lens, optimizer, scheduler, data_iter, dataloader_factory)
        finally:
            self.collector.detach()
            self._close_metrics_csv()

    def _train_loop(
        self,
        lens: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        data_iter: Iterator,
        dataloader_factory: Callable,
    ) -> None:
        config = self.config
        ddp_state = self.ddp_state
        amp_ctx = self.amp_ctx
        is_ddp = isinstance(lens, DDP)

        for step in range(self.start_step + 1, self.end_step + 1):
            self.global_step = step
            t0 = time.perf_counter()

            if ddp_state.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self._lens_device)

            optimizer.zero_grad(set_to_none=True)

            target_tokens = config.tokens_per_step
            accum_tokens = 0.0
            accum_loss = torch.zeros((), device=self._lens_device, dtype=torch.float32)
            microsteps = 0

            while True:
                microsteps += 1

                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader_factory())
                    batch = next(data_iter)

                # Non-blocking transfer
                input_ids = batch["input_ids"].to(self._lens_device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self._lens_device, non_blocking=True)

                local_tokens = attention_mask.sum().item()
                tokens_tensor = torch.tensor([local_tokens], device=self._lens_device)
                tokens_tensor = all_reduce_sum(tokens_tensor, ddp_state)
                batch_tokens = tokens_tensor.item()
                accum_tokens += batch_tokens

                should_sync = (
                    target_tokens is None
                    or accum_tokens >= target_tokens
                    or microsteps >= config.max_microsteps
                )

                micro_loss = self._forward_backward_layerwise(
                    lens=lens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    scale=batch_tokens / target_tokens if target_tokens else 1.0,
                    should_sync=should_sync,
                    is_ddp=is_ddp,
                )
                accum_loss = accum_loss + micro_loss


                del input_ids, attention_mask

                if should_sync:
                    break

            amp_ctx.unscale_(optimizer)
            if config.grad_clip_norm is not None:
                clip_grad_norm_(
                    [p for p in lens.parameters() if p.grad is not None],
                    config.grad_clip_norm,
                )
            amp_ctx.step(optimizer)

            if scheduler is not None:
                scheduler.step()

            if self._lens_device.type == "cuda":
                torch.cuda.synchronize(self._lens_device)
            dt = time.perf_counter() - t0

            self.total_tokens += int(accum_tokens)
            accum_loss = all_reduce_sum(accum_loss, ddp_state) / ddp_state.world_size

            completed_steps = step - self.start_step

            if ddp_state.is_main and completed_steps % config.log_every == 0:
                self._log_step(step, accum_loss.item(), accum_tokens / dt, dt, microsteps, optimizer)

            if ddp_state.is_main and (
                completed_steps % config.save_every == 0 or step == self.end_step
            ):
                self._save_checkpoint(lens, step)

    def _get_t_slices(self, T: int) -> List[Tuple[int, int]]:
        """Get T-dimension chunk slices."""
        chunk_size = self.config.kl_chunk_size or T
        if chunk_size <= 0 or chunk_size >= T:
            return [(0, T)]
        return [(t0, min(t0 + chunk_size, T)) for t0 in range(0, T, chunk_size)]

    def _forward_backward_layerwise(
        self,
        lens: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scale: float,
        should_sync: bool,
        is_ddp: bool,
    ) -> torch.Tensor:
        """Layer-wise forward/backward with T-chunking and model-parallel support."""
        amp_ctx = self.amp_ctx
        config = self.config
        shift = config.token_shift
        lens_device = self._lens_device

        # When model-parallel, input_ids go to the model's embedding device
        if self.shard_state.enabled:
            model_input_device = _get_model_input_device(self.model)
            input_ids_model = input_ids.to(model_input_device, non_blocking=True)
            attn_model = attention_mask.to(model_input_device, non_blocking=True)
        else:
            input_ids_model = input_ids
            attn_model = attention_mask


        with amp_ctx.autocast():
            activations = self.collector.collect_with_grad(
                input_ids=input_ids_model,
                attention_mask=attn_model,
            )

        if self.shard_state.enabled:
            del input_ids_model, attn_model

        teacher_logits_full = activations.logits
        hidden_states = activations.hidden_states
        custom_activations = activations.custom

        if teacher_logits_full is None:
            raise RuntimeError("Model did not return logits")
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states")

        ordered_site_acts = self._collect_site_activations(hidden_states, custom_activations)

        # --- Move outputs to lens device (model-parallel path) ---
        if self.shard_state.enabled:
            teacher_logits_full = teacher_logits_full.to(lens_device, non_blocking=True)
            # Ensure attention_mask / input_ids are on lens device for loss
            attention_mask = attention_mask.to(lens_device, non_blocking=True)
            input_ids = input_ids.to(lens_device, non_blocking=True)

        # Apply shift
        if shift > 0:
            attn_full = attention_mask[:, shift:]
            labels_full = input_ids[:, shift:]
            teacher_logits_full = teacher_logits_full[:, shift:, :]
        else:
            attn_full = attention_mask
            labels_full = input_ids


        del activations, hidden_states, custom_activations

        # Get T-dimension slices for chunking
        T_eff = attn_full.size(1)
        t_slices = self._get_t_slices(T_eff) if (
            self._use_chunked_kl or self._use_subset_kl
        ) else [(0, T_eff)]

        # In-place log-softmax (avoids large temp allocation)
        if self._use_chunked_kl:
            with torch.no_grad():
                for t0, t1 in t_slices:
                    tl = teacher_logits_full[:, t0:t1, :]
                    teacher_logits_full[:, t0:t1, :].copy_(
                        F.log_softmax(tl, dim=-1)
                    )
                    del tl

        subset_chunks = None
        if self._use_subset_kl:
            subset_chunks = []
            with torch.no_grad():
                for t0, t1 in t_slices:
                    subset_chunks.append(
                        self.loss_fn.prepare_teacher_subset(
                            teacher_logits_full[:, t0:t1, :].contiguous()
                        )
                    )

        raw_lens = lens.module if isinstance(lens, DDP) else lens
        layer_ids = raw_lens.layer_ids if raw_lens.layer_ids else [site_id for site_id, _ in ordered_site_acts]
        n_layers = len(layer_ids)
        inv_layers = 1.0 / n_layers
        site_map = dict(ordered_site_acts)

        total_loss = torch.zeros((), device=lens_device, dtype=torch.float32)

        # Determine microstep-level sync context
        micro_ctx = lens.no_sync() if (is_ddp and not should_sync) else nullcontext()

        with micro_ctx:
            for li, layer_id in enumerate(layer_ids):
                is_last_layer = (li == n_layers - 1)
                h_full_raw = site_map[str(layer_id)]

                # Move hidden states to lens device (model-parallel path)
                if self.shard_state.enabled and h_full_raw.device != lens_device:
                    h_full_raw = h_full_raw.to(lens_device, non_blocking=True)

                # DDP sync: only on last layer of sync microstep
                if is_ddp and should_sync:
                    layer_ctx = nullcontext() if is_last_layer else lens.no_sync()
                else:
                    layer_ctx = nullcontext()

                with layer_ctx:
                    with amp_ctx.autocast():
                        if self._use_chunked_kl:

                            layer_loss = self._compute_chunked_kl_loss(
                                lens=lens,
                                h_full_raw=h_full_raw,
                                teacher_logprobs_full=teacher_logits_full,  # Already log-probs!
                                attn_full=attn_full,
                                layer_id=layer_id,
                                shift=shift,
                                t_slices=t_slices,
                            )
                        elif self._use_shared_subset_kl:
                            # Shared subset KL - most memory efficient
                            layer_loss = self._compute_shared_subset_kl_loss(
                                lens=lens,
                                h_full_raw=h_full_raw,
                                teacher_logits_full=teacher_logits_full,
                                attn_full=attn_full,
                                layer_id=layer_id,
                                shift=shift,
                            )
                        elif self._use_subset_kl:
                            # Per-position subset KL path
                            layer_loss = self._compute_subset_kl_loss(
                                lens=lens,
                                h_full_raw=h_full_raw,
                                teacher_logits_full=teacher_logits_full,
                                attn_full=attn_full,
                                layer_id=layer_id,
                                shift=shift,
                                t_slices=t_slices,
                                subset_chunks=subset_chunks,
                            )
                        else:
                            # Standard path (CE or non-chunked KL)
                            layer_loss = self._compute_standard_loss(
                                lens=lens,
                                h_full_raw=h_full_raw,
                                teacher_logits_full=teacher_logits_full,
                                attn_full=attn_full,
                                labels_full=labels_full,
                                layer_id=layer_id,
                                shift=shift,
                            )

                        layer_loss = layer_loss * inv_layers * scale
                        total_loss = total_loss + layer_loss.detach()

                    # Backward immediately
                    scaled = amp_ctx.scale(layer_loss)
                    scaled.backward()


                del layer_loss, h_full_raw


        del ordered_site_acts, site_map, teacher_logits_full, attn_full, labels_full, subset_chunks

        return total_loss

    def _compute_chunked_kl_loss(
        self,
        lens: nn.Module,
        h_full_raw: torch.Tensor,
        teacher_logprobs_full: torch.Tensor,  # Already log-softmax!
        attn_full: torch.Tensor,
        layer_id,
        shift: int,
        t_slices: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Compute KL loss with T-dimension chunking."""
        device = h_full_raw.device
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        denom = torch.zeros((), device=device, dtype=torch.float32)

        # View, not copy
        if shift > 0:
            h_full = h_full_raw[:, :-shift, :]  # View!
        else:
            h_full = h_full_raw  # Reference!

        for t0, t1 in t_slices:
            attn = attn_full[:, t0:t1].contiguous()
            if attn.sum().item() == 0:
                continue


            h = h_full[:, t0:t1, :].contiguous()
            t_logprobs = teacher_logprobs_full[:, t0:t1, :].contiguous()

            # Forward through lens
            out = lens(h, layer=layer_id)
            if out.logits is None:
                raise RuntimeError("Lens did not return logits")

            # Compute chunk KL
            chunk_sum = _masked_kl_logtarget(
                student_logits=out.logits,
                teacher_logprobs=t_logprobs,
                attention_mask=attn,
                reduction="sum",
            )

            loss_sum = loss_sum + chunk_sum
            denom = denom + attn.sum(dtype=torch.float32)


            del out, chunk_sum, t_logprobs, h, attn


        del h_full

        return loss_sum / denom.clamp_min(1.0)

    def _compute_subset_kl_loss(
        self,
        lens: nn.Module,
        h_full_raw: torch.Tensor,
        teacher_logits_full: torch.Tensor,
        attn_full: torch.Tensor,
        layer_id,
        shift: int,
        t_slices: List[Tuple[int, int]],
        subset_chunks: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """
        Compute subset KL loss with T-chunking for memory efficiency.

        Even though subset KL only uses k tokens, we still need T-chunking
        because the lens.forward() computes full [B, chunk, V] internally
        before gathering the subset (for per-position varying indices).
        """
        device = h_full_raw.device
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        denom = torch.zeros((), device=device, dtype=torch.float32)

        # Lazy slicing - view, not copy
        if shift > 0:
            h_full = h_full_raw[:, :-shift, :]
        else:
            h_full = h_full_raw

        for chunk_idx, (t0, t1) in enumerate(t_slices):
            attn = attn_full[:, t0:t1].contiguous()
            if attn.sum().item() == 0:
                continue

            # Only copy the chunk
            h = h_full[:, t0:t1, :].contiguous()
            teacher_subset = (
                subset_chunks[chunk_idx] if subset_chunks is not None else None
            )
            teacher_chunk = None
            if teacher_subset is None:
                teacher_chunk = teacher_logits_full[:, t0:t1, :].contiguous()

            # Forward through SubsetKLLoss for this chunk
            chunk_loss = self.loss_fn.forward_with_lens(
                hidden_states=h,
                teacher_logits=teacher_chunk,
                lens=lens,
                layer=layer_id,
                attention_mask=attn,
                teacher_subset=teacher_subset,
            )

            # Accumulate (loss_fn returns mean, we want sum for proper averaging)
            chunk_count = attn.sum(dtype=torch.float32)
            loss_sum = loss_sum + chunk_loss * chunk_count
            denom = denom + chunk_count

            del h, teacher_chunk, attn, chunk_loss

        del h_full
        return loss_sum / denom.clamp_min(1.0)

    def _compute_shared_subset_kl_loss(
        self,
        lens: nn.Module,
        h_full_raw: torch.Tensor,
        teacher_logits_full: torch.Tensor,
        attn_full: torch.Tensor,
        layer_id,
        shift: int,
    ) -> torch.Tensor:
        """
        Compute shared subset KL loss with T-chunking.

        Uses SharedSubsetKLLoss which builds a shared candidate vocabulary
        per chunk, then computes student logits ONLY for those candidates.

        Memory: O(B * chunk_T * K) where K << V.
        """
        device = h_full_raw.device
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        denom = torch.zeros((), device=device, dtype=torch.float32)

        # Lazy slicing - view, not copy
        if shift > 0:
            h_full = h_full_raw[:, :-shift, :]
        else:
            h_full = h_full_raw

        # Get T-dimension slices
        T_eff = attn_full.size(1)
        chunk_size = self.config.kl_chunk_size or 128
        t_slices = [(t0, min(t0 + chunk_size, T_eff)) for t0 in range(0, T_eff, chunk_size)]

        # Get raw lens for forward_with_lens
        raw_lens = lens.module if isinstance(lens, DDP) else lens

        for t0, t1 in t_slices:
            attn = attn_full[:, t0:t1].contiguous()
            if attn.sum().item() == 0:
                continue

            # Only copy the chunk
            h = h_full[:, t0:t1, :].contiguous()
            teacher_chunk = teacher_logits_full[:, t0:t1, :].contiguous()

            # Forward through SharedSubsetKLLoss
            # This never materializes [B, chunk_T, V] student logits!
            chunk_loss = self.loss_fn.forward_with_lens(
                hidden_states=h,
                teacher_logits=teacher_chunk,
                lens=raw_lens,
                layer=layer_id,
                attention_mask=attn,
            )

            # Accumulate
            chunk_count = attn.sum(dtype=torch.float32)
            loss_sum = loss_sum + chunk_loss * chunk_count
            denom = denom + chunk_count

            del h, teacher_chunk, attn, chunk_loss

        del h_full
        return loss_sum / denom.clamp_min(1.0)

    def _compute_standard_loss(
        self,
        lens: nn.Module,
        h_full_raw: torch.Tensor,
        teacher_logits_full: torch.Tensor,
        attn_full: torch.Tensor,
        labels_full: torch.Tensor,
        layer_id,
        shift: int,
    ) -> torch.Tensor:
        """Compute standard loss (CE or non-chunked KL)."""
        if shift > 0:
            h = h_full_raw[:, :-shift, :].contiguous()
        else:
            h = h_full_raw.contiguous()

        lens_out = lens(h, layer=layer_id)

        loss = self.loss_fn(
            student_logits=lens_out.logits,
            teacher_logits=teacher_logits_full.contiguous(),
            attention_mask=attn_full.contiguous(),
            labels=labels_full.contiguous(),
        )

        del lens_out, h
        return loss

    def _write_run_config_csv(self) -> None:
        activation_sites_path = self.config.output_dir / "activation_sites.csv"
        self._write_activation_sites_csv(activation_sites_path)

        subset_mode = self.config.subset_kl_mode if self.config.loss_type == "subset_kl" else ""
        subset_k = self.config.subset_kl_k if self.config.loss_type == "subset_kl" else ""
        subset_k_tail = self.config.subset_kl_k_tail if self.config.loss_type == "subset_kl" else ""
        kl_chunk_size = self.config.kl_chunk_size if self.config.loss_type == "kl" else ""

        config_path = self.config.output_dir / "run_config.csv"
        with config_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name",
                "lens_type",
                "loss_type",
                "loss_impl",
                "lora_rank",
                "lora_init",
                "lora_init_calibration_tokens",
                "lora_init_ridge_lambda",
                "lora_init_ridge_lambda_scale",
                "lora_init_stats_dtype",
                "lora_init_jitter",
                "lora_init_svd_metric",
                "lora_init_normalization",
                "subset_kl_mode",
                "subset_kl_k",
                "subset_kl_k_tail",
                "kl_chunk_size",
                "activation_site_preset",
                "activation_site_count",
                "activation_sites_file",
                "batch_size",
                "num_steps",
                "warmup_steps",
                "lr_schedule",
                "min_lr_ratio",
                "lr_init",
                "tokens_per_step",
                "max_seq_len",
                "seed",
                "output_dir",
                "resume_checkpoint",
                "resume_start_step",
                "resume_target_step",
            ])
            writer.writerow([
                self.config.model_name,
                self.config.lens_type,
                self.config.loss_type,
                type(self.loss_fn).__name__,
                self.config.lora_rank,
                self.config.lora_init,
                self.config.lora_init_calibration_tokens,
                f"{self.config.lora_init_ridge_lambda:.10g}",
                self.config.lora_init_ridge_lambda_scale,
                self.config.lora_init_stats_dtype,
                f"{self.config.lora_init_jitter:.10g}",
                self.config.lora_init_svd_metric,
                self.config.lora_init_normalization,
                subset_mode,
                subset_k,
                subset_k_tail,
                kl_chunk_size,
                self.config.activation_site_preset,
                self.activation_site_plan.site_count,
                activation_sites_path.name,
                self.config.per_gpu_batch_size,
                self.config.num_steps,
                self.config.warmup_steps,
                self.config.lr_schedule,
                f"{self.config.min_lr_ratio:.10g}",
                f"{self.config.lr:.10g}",
                self.config.tokens_per_step,
                self.config.max_seq_len,
                self.config.seed,
                str(self.config.output_dir),
                "" if self.config.resume_checkpoint is None else str(self.config.resume_checkpoint),
                self.start_step,
                self.end_step,
            ])

    def _write_activation_sites_csv(self, path) -> None:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["site_id", "source_type", "source_ref"])
            for site_id in self.activation_site_plan.site_ids:
                if site_id in self.activation_site_plan.hidden_state_sources:
                    writer.writerow([
                        site_id,
                        "hidden_state",
                        self.activation_site_plan.hidden_state_sources[site_id],
                    ])
                    continue

                custom_key = self.activation_site_plan.custom_sources[site_id]
                writer.writerow([
                    site_id,
                    "custom_hook",
                    self.activation_site_plan.custom_hooks.get(custom_key, custom_key),
                ])

    def _open_metrics_csv(self) -> None:
        metrics_path = self.config.output_dir / "metrics.csv"
        append_mode = self.start_step > 0 and metrics_path.exists() and metrics_path.stat().st_size > 0
        mode = "a" if append_mode else "w"
        self._metrics_file = metrics_path.open(mode, newline="")
        self._metrics_writer = csv.writer(self._metrics_file)
        if not append_mode:
            self._metrics_writer.writerow([
                "step",
                "total_tokens",
                "loss",
                "tok_per_sec",
                "dt_ms",
                "microsteps",
                "lr",
                "peak_memory_gb",
            ])
        self._metrics_file.flush()

    def _close_metrics_csv(self) -> None:
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
            self._metrics_writer = None

    def _log_step(self, step, loss, tok_per_sec, dt, microsteps, optimizer):
        lr = optimizer.param_groups[0]["lr"]
        peak_mem_gb = ""
        msg = f"step {step}/{self.end_step} | loss={loss:.4f} | tok/s={tok_per_sec:.0f} | dt={dt*1000:.0f}ms | micro={microsteps} | lr={lr:.2e}"
        if self.config.log_memory and self.ddp_state.device.type == "cuda":
            if self.shard_state.is_sharded:
                # Report peak memory across all GPUs used by the model
                peak_mem_gb = sum(
                    torch.cuda.max_memory_allocated(i) for i in range(self.shard_state.num_gpus)
                ) / 1e9
                msg += f" | mem={peak_mem_gb:.1f}GB({self.shard_state.num_gpus}gpu)"
            else:
                peak_mem_gb = torch.cuda.max_memory_allocated(self.ddp_state.device) / 1e9
                msg += f" | mem={peak_mem_gb:.1f}GB"
        logger.info(msg)
        if self._metrics_writer is not None:
            self._metrics_writer.writerow([
                step,
                self.total_tokens,
                f"{loss:.8f}",
                f"{tok_per_sec:.4f}",
                f"{dt * 1000.0:.4f}",
                microsteps,
                f"{lr:.10g}",
                "" if peak_mem_gb == "" else f"{peak_mem_gb:.4f}",
            ])
            self._metrics_file.flush()

    def _save_checkpoint(self, lens: nn.Module, step: int) -> None:
        raw_lens = lens.module if isinstance(lens, DDP) else lens
        path = self.config.output_dir / f"lens_step_{step}.pt"
        checkpoint = {
            "step": step,
            "total_tokens": self.total_tokens,
            "config": self.config.to_dict(),
            "lens_state_dict": raw_lens.checkpoint_state_dict(),
        }

        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self.config.output_dir,
                prefix=f"{path.stem}.",
                suffix=".tmp",
            )
            os.close(fd)
            fd = None

            # The legacy serialization path is more robust on some shared filesystems.
            torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)

            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())

            os.replace(tmp_path, path)
        except Exception:
            if fd is not None:
                os.close(fd)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
            raise
        logger.info(f"Saved checkpoint to {path}")
