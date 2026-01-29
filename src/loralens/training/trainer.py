# src/loralens/training/trainer.py
"""
LensTrainer - Thin orchestration layer for lens training.

Memory optimizations implemented:
1. In-place teacher log-softmax conversion (avoids [B,T,V] allocation)
2. T-dimension chunking (processes sequence in chunks)
3. Lazy hidden state slicing (views instead of copies)
4. Aggressive del statements (helps GC)
5. Dtype preservation (keeps bf16/fp16, avoids fp32 upcasting)
6. Non-blocking data transfer
7. return_dict=False for model calls
"""

from __future__ import annotations

import logging
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

from .config import TrainConfig
from .distributed import DDPState, all_reduce_sum
from .amp import AMPContext


logger = logging.getLogger(__name__)


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
        config: TrainConfig,
        ddp_state: DDPState,
        amp_ctx: AMPContext,
    ) -> None:
        self.model = model
        self.lens = lens
        self.loss_fn = loss_fn
        self.collector = collector
        self.config = config
        self.ddp_state = ddp_state
        self.amp_ctx = amp_ctx
        self.global_step = 0
        self.total_tokens = 0
        
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
        
        lens = self.lens
        if ddp_state.enabled:
            lens = DDP(
                lens,
                device_ids=[ddp_state.local_rank],
                output_device=ddp_state.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        
        data_iter = iter(dataloader_factory())
        self.collector.attach()
        
        try:
            self._train_loop(lens, optimizer, scheduler, data_iter, dataloader_factory)
        finally:
            self.collector.detach()

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
        
        for step in range(1, config.num_steps + 1):
            self.global_step = step
            t0 = time.perf_counter()
            
            if ddp_state.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(ddp_state.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            target_tokens = config.tokens_per_step
            accum_tokens = 0.0
            accum_loss = torch.zeros((), device=ddp_state.device, dtype=torch.float32)
            microsteps = 0
            
            while True:
                microsteps += 1
                
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader_factory())
                    batch = next(data_iter)
                
                # OPTIMIZATION: Non-blocking transfer
                input_ids = batch["input_ids"].to(ddp_state.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(ddp_state.device, non_blocking=True)
                
                local_tokens = attention_mask.sum().item()
                tokens_tensor = torch.tensor([local_tokens], device=ddp_state.device)
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
                
                # OPTIMIZATION: Delete batch tensors early
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
            
            if ddp_state.device.type == "cuda":
                torch.cuda.synchronize(ddp_state.device)
            dt = time.perf_counter() - t0
            
            self.total_tokens += int(accum_tokens)
            accum_loss = all_reduce_sum(accum_loss, ddp_state) / ddp_state.world_size
            
            if ddp_state.is_main and step % config.log_every == 0:
                self._log_step(step, accum_loss.item(), accum_tokens / dt, dt, microsteps, optimizer)
            
            if ddp_state.is_main and step % config.save_every == 0:
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
        """
        Memory-optimized layer-wise forward/backward.
        
        Key optimizations:
        1. In-place log-softmax for teacher
        2. T-dimension chunking
        3. Lazy hidden state slicing
        4. Aggressive memory cleanup
        """
        amp_ctx = self.amp_ctx
        config = self.config
        shift = config.token_shift
        
        # OPTIMIZATION: return_dict=False for efficiency
        with torch.no_grad():
            with amp_ctx.autocast():
                model_out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=False,
                )
        
        # Unpack tuple output
        teacher_logits_full = model_out[0]  # [B, T, V]
        hidden_states = model_out[1] if len(model_out) > 1 else None
        
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states")
        
        per_layer_acts = hidden_states[1:]  # Drop embeddings
        
        # Apply shift
        if shift > 0:
            attn_full = attention_mask[:, shift:]
            labels_full = input_ids[:, shift:]
            teacher_logits_full = teacher_logits_full[:, shift:, :]
        else:
            attn_full = attention_mask
            labels_full = input_ids
        
        # OPTIMIZATION: Delete early
        del model_out, hidden_states
        
        # Get T-dimension slices for chunking
        T_eff = attn_full.size(1)
        t_slices = self._get_t_slices(T_eff) if self._use_chunked_kl else [(0, T_eff)]
        
        # OPTIMIZATION: In-place log-softmax for KL (chunked to avoid large temp)
        if self._use_chunked_kl:
            with torch.no_grad():
                for t0, t1 in t_slices:
                    tl = teacher_logits_full[:, t0:t1, :]
                    teacher_logits_full[:, t0:t1, :].copy_(
                        F.log_softmax(tl, dim=-1)
                    )
                    del tl
        
        raw_lens = lens.module if isinstance(lens, DDP) else lens
        layer_ids = raw_lens.layer_ids if raw_lens.layer_ids else list(range(len(per_layer_acts)))
        n_layers = len(layer_ids)
        inv_layers = 1.0 / n_layers
        
        total_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        
        # Determine microstep-level sync context
        micro_ctx = lens.no_sync() if (is_ddp and not should_sync) else nullcontext()
        
        with micro_ctx:
            for li, layer_id in enumerate(layer_ids):
                is_last_layer = (li == n_layers - 1)
                h_full_raw = per_layer_acts[li]
                
                # DDP sync: only on last layer of sync microstep
                if is_ddp and should_sync:
                    layer_ctx = nullcontext() if is_last_layer else lens.no_sync()
                else:
                    layer_ctx = nullcontext()
                
                with layer_ctx:
                    with amp_ctx.autocast():
                        if self._use_chunked_kl:
                            # OPTIMIZATION: Chunked KL with lazy slicing
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
                
                # OPTIMIZATION: Delete layer loss
                del layer_loss
        
        # OPTIMIZATION: Cleanup
        del per_layer_acts, teacher_logits_full, attn_full, labels_full
        
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
        
        # OPTIMIZATION: Lazy slicing - only view, not copy
        if shift > 0:
            h_full = h_full_raw[:, :-shift, :]  # View!
        else:
            h_full = h_full_raw  # Reference!
        
        for t0, t1 in t_slices:
            attn = attn_full[:, t0:t1].contiguous()
            if attn.sum().item() == 0:
                continue
            
            # OPTIMIZATION: Only copy the chunk
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
            
            # OPTIMIZATION: Delete chunk tensors immediately
            del out, chunk_sum, t_logprobs, h, attn
        
        # OPTIMIZATION: Delete full view
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
        
        # Get T-dimension slices
        T_eff = attn_full.size(1)
        chunk_size = self.config.kl_chunk_size or 128
        t_slices = [(t0, min(t0 + chunk_size, T_eff)) for t0 in range(0, T_eff, chunk_size)]
        
        for t0, t1 in t_slices:
            attn = attn_full[:, t0:t1].contiguous()
            if attn.sum().item() == 0:
                continue
            
            # Only copy the chunk
            h = h_full[:, t0:t1, :].contiguous()
            teacher_chunk = teacher_logits_full[:, t0:t1, :].contiguous()
            
            # Forward through SubsetKLLoss for this chunk
            chunk_loss = self.loss_fn.forward_with_lens(
                hidden_states=h,
                teacher_logits=teacher_chunk,
                lens=lens,
                layer=layer_id,
                attention_mask=attn,
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

    def _log_step(self, step, loss, tok_per_sec, dt, microsteps, optimizer):
        lr = optimizer.param_groups[0]["lr"]
        msg = f"step {step}/{self.config.num_steps} | loss={loss:.4f} | tok/s={tok_per_sec:.0f} | dt={dt*1000:.0f}ms | micro={microsteps} | lr={lr:.2e}"
        if self.config.log_memory and self.ddp_state.device.type == "cuda":
            mem_gb = torch.cuda.max_memory_allocated(self.ddp_state.device) / 1e9
            msg += f" | mem={mem_gb:.1f}GB"
        logger.info(msg)

    def _save_checkpoint(self, lens: nn.Module, step: int) -> None:
        raw_lens = lens.module if isinstance(lens, DDP) else lens
        path = self.config.output_dir / f"lens_step_{step}.pt"
        torch.save({
            "step": step,
            "total_tokens": self.total_tokens,
            "config": self.config.to_dict(),
            "lens_state_dict": raw_lens.state_dict(),
        }, path)
        logger.info(f"Saved checkpoint to {path}")
