# src/loralens/training/loop.py
from __future__ import annotations

import logging
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

from loralens.lenses import LogitLens, LoRALens, TunedLens

from .amp import autocast_ctx, make_scaler
from .data import build_dataloader, next_batch
from .ddp import DDPState, all_reduce_sum, init_ddp, shutdown_ddp
from .losses import masked_kl_logtarget
from .metrics import BitsPerByteState, estimate_batch_bytes, gather_peak_mem_lines
from .unembed import HFUnembed

logger = logging.getLogger(__name__)

DataSource = Literal["text", "pile"]
LensType = Literal["logit", "tuned", "lora"]


class LossChoice(str, Enum):
    CE = "ce"
    KL = "kl"


LossType = Union[LossChoice, str]


def _maybe_configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Train:
    model_name: str = "gpt2"
    model_revision: Optional[str] = None
    tokenizer_name: Optional[str] = None

    data_source: DataSource = "text"
    text_paths: List[Path] = field(default_factory=list)
    text_mode: Literal["lines", "whole", "jsonl"] = "lines"
    text_json_field: str = "text"
    pile_split: str = "train"

    max_seq_len: int = 1024
    stride: Optional[int] = None
    drop_remainder: bool = True
    document_separated: bool = True
    max_docs: Optional[int] = None

    lens_type: LensType = "tuned"
    lora_rank: int = 16

    per_gpu_batch_size: int = 1
    num_steps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_clip_norm: Optional[float] = 1.0

    token_shift: Optional[int] = None
    loss: LossType = LossChoice.KL

    tokens_per_step: Optional[int] = 262_144
    max_microsteps: int = 512

    # Chunk along T for KL. 0 disables chunking.
    kl_t_chunk: int = 128

    output: Path = Path("lens.pt")
    seed: int = 0
    log_every: int = 1
    save_every: int = 100

    report_bits_per_byte: bool = True
    log_mem_every: int = 1
    log_data_every: int = 10

    ddp: bool = True
    ddp_backend: str = "nccl"

    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    def execute(self) -> None:
        _maybe_configure_logging()

        if isinstance(self.loss, str):
            s = self.loss.lower()
            if s == "kl":
                self.loss = LossChoice.KL
            elif s == "ce":
                self.loss = LossChoice.CE
            else:
                raise ValueError(f"Unknown loss={self.loss!r}; expected 'kl' or 'ce'.")

        if self.token_shift is None:
            # KL compares same-position distributions; CE is next-token
            self.token_shift = 0 if self.loss == LossChoice.KL else 1

        ddp_state = init_ddp(backend=self.ddp_backend, enabled=self.ddp)
        _set_seed(self.seed + ddp_state.rank)

        tokenizer = self._load_tokenizer(ddp_state)
        model = self._load_model(tokenizer=tokenizer, device=ddp_state.device)

        def make_iter():
            dl = build_dataloader(
                tokenizer=tokenizer,
                ddp_state=ddp_state,
                seed=self.seed,
                data_source=self.data_source,
                text_paths=self.text_paths,
                text_mode=self.text_mode,
                text_json_field=self.text_json_field,
                pile_split=self.pile_split,
                max_seq_len=self.max_seq_len,
                stride=self.stride,
                drop_remainder=self.drop_remainder,
                document_separated=self.document_separated,
                max_docs=self.max_docs,
                per_gpu_batch_size=self.per_gpu_batch_size,
            )
            return iter(dl)

        data_iter = make_iter()

        lens_dtype = (
            torch.bfloat16
            if (self.amp and self.amp_dtype == "bf16")
            else torch.float16
            if (self.amp and self.amp_dtype == "fp16")
            else torch.float32
        )

        lens, layer_ids, hidden_size = self._build_lens(
            model=model, dtype=lens_dtype, device=ddp_state.device
        )

        if ddp_state.enabled:
            lens = DDP(
                lens,
                device_ids=[ddp_state.local_rank],
                output_device=ddp_state.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        optimizer = torch.optim.AdamW(lens.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = None
        if self.warmup_steps > 0:

            def lr_lambda(step: int) -> float:
                if step < self.warmup_steps:
                    return float(step + 1) / float(max(1, self.warmup_steps))
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scaler = make_scaler(self.amp, self.amp_dtype)

        try:
            self._train_loop(
                model=model,
                lens=lens,
                layer_ids=layer_ids,
                hidden_size=hidden_size,
                tokenizer=tokenizer,
                ddp_state=ddp_state,
                optimizer=optimizer,
                scheduler=scheduler,
                data_iter=data_iter,
                make_iter_fn=make_iter,
                scaler=scaler,
            )
        finally:
            shutdown_ddp(ddp_state)

    def _load_tokenizer(self, ddp_state: DDPState):
        tok_name = self.tokenizer_name or self.model_name
        tok = AutoTokenizer.from_pretrained(tok_name, revision=self.model_revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token is not None else "<pad>"
        return tok

    def _load_model(self, tokenizer, device: torch.device):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision)
        model.resize_token_embeddings(len(tokenizer))
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)
        model.eval()
        return model

    def _build_lens(self, model, dtype: torch.dtype, device: torch.device):
        cfg = model.config
        num_layers = int(cfg.n_layer) if hasattr(cfg, "n_layer") else int(cfg.num_hidden_layers)
        hidden_size = int(cfg.n_embd) if hasattr(cfg, "n_embd") else int(cfg.hidden_size)

        layer_ids = [str(i) for i in range(num_layers)]
        unembed = HFUnembed(model)

        if self.lens_type == "tuned":
            lens = TunedLens(layer_ids=layer_ids, hidden_size=hidden_size, unembed=unembed, init_zero=True)
        elif self.lens_type == "logit":
            lens = LogitLens(readout=unembed)
        elif self.lens_type == "lora":
            lens = LoRALens(
                layer_ids=layer_ids,
                hidden_size=hidden_size,
                readout=unembed,
                r=self.lora_rank,
                init_identity_base=True,
            )
        else:
            raise ValueError(f"Unknown lens_type={self.lens_type!r}")

        lens.to(device=device, dtype=dtype)
        lens.train()
        return lens, layer_ids, hidden_size

    def _train_loop(
        self,
        *,
        model,
        lens,
        layer_ids: List[str],
        hidden_size: int,
        tokenizer,
        ddp_state: DDPState,
        optimizer: torch.optim.Optimizer,
        scheduler,
        data_iter,
        make_iter_fn,
        scaler: torch.cuda.amp.GradScaler,
    ) -> None:
        out_path = self.output
        if ddp_state.is_main:
            out_path.parent.mkdir(parents=True, exist_ok=True)

        bpb = BitsPerByteState()
        total_tokens_seen = 0
        is_ddp = isinstance(lens, DDP)

        for step in range(1, self.num_steps + 1):
            if ddp_state.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(ddp_state.device)
                torch.cuda.synchronize(ddp_state.device)
            t_step0 = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            target_tokens = self.tokens_per_step
            accum_global_tokens = 0.0
            accum_loss_local = torch.zeros((), device=ddp_state.device, dtype=torch.float32)
            microsteps = 0

            while True:
                microsteps += 1
                batch, data_iter = next_batch(data_iter, make_iter_fn)

                input_ids_cpu = batch["input_ids"]
                attention_mask_cpu = batch["attention_mask"]

                if self.report_bits_per_byte:
                    local_tokens = float(attention_mask_cpu.sum().item())
                    local_bytes = float(max(estimate_batch_bytes(tokenizer, input_ids_cpu, attention_mask_cpu), 1))
                    bt = torch.tensor([local_tokens, local_bytes], device=ddp_state.device, dtype=torch.float32)
                    bt = all_reduce_sum(bt, ddp_state)
                    tokens_this_micro = float(bt[0].item())
                    bpb.update(tokens=float(bt[0].item()), nbytes=float(bt[1].item()))
                else:
                    local_tokens = float(attention_mask_cpu.sum().item())
                    t = torch.tensor([local_tokens], device=ddp_state.device, dtype=torch.float32)
                    t = all_reduce_sum(t, ddp_state)
                    tokens_this_micro = float(t[0].item())

                accum_global_tokens += tokens_this_micro

                input_ids = input_ids_cpu.to(ddp_state.device, non_blocking=True)
                attention_mask = attention_mask_cpu.to(ddp_state.device, non_blocking=True)

                # Teacher forward (frozen)
                with torch.no_grad():
                    with autocast_ctx(ddp_state.device, self.amp, self.amp_dtype):
                        model_out = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=False,
                        )

                # return_dict=False: (logits, hidden_states) for GPT-like CausalLM
                teacher_logits_full = model_out[0]  # [B, T, V] bf16/fp16
                hidden_states = model_out[1]
                per_layer_acts = hidden_states[1:]  # drop embeddings

                shift = max(int(self.token_shift or 0), 0)
                if shift > 0:
                    attn_full = attention_mask[:, shift:].contiguous()
                    acts = [h[:, :-shift, :].contiguous() for h in per_layer_acts]
                    teacher_logits_full = teacher_logits_full[:, shift:, :].contiguous()
                    labels_full = input_ids[:, shift:].contiguous()
                else:
                    attn_full = attention_mask
                    acts = [h.contiguous() for h in per_layer_acts]
                    labels_full = input_ids

                del model_out, hidden_states, per_layer_acts

                n_layers = len(acts)
                inv_layers = 1.0 / float(n_layers)
                scale = (tokens_this_micro / float(target_tokens)) if target_tokens is not None else 1.0

                micro_sync = (
                    (target_tokens is None)
                    or (accum_global_tokens >= float(target_tokens))
                    or (microsteps >= self.max_microsteps)
                )
                micro_ctx = lens.no_sync() if (is_ddp and not micro_sync) else nullcontext()

                T_eff = int(attn_full.size(1))
                t_chunk = int(self.kl_t_chunk) if (self.loss == LossChoice.KL) else 0
                if t_chunk <= 0 or t_chunk >= T_eff:
                    t_slices = [(0, T_eff)]
                else:
                    t_slices = [(t0, min(t0 + t_chunk, T_eff)) for t0 in range(0, T_eff, t_chunk)]

                with micro_ctx:
                    for li, (lid, h_full) in enumerate(zip(layer_ids, acts)):
                        if is_ddp and micro_sync:
                            layer_ctx = lens.no_sync() if (li != n_layers - 1) else nullcontext()
                        else:
                            layer_ctx = nullcontext()

                        with layer_ctx:
                            with autocast_ctx(ddp_state.device, self.amp, self.amp_dtype):
                                if self.loss == LossChoice.KL:
                                    # accumulate exact mean KL across T chunks for this layer
                                    loss_sum = torch.zeros((), device=ddp_state.device, dtype=torch.float32)
                                    denom = torch.zeros((), device=ddp_state.device, dtype=torch.float32)

                                    for t0, t1 in t_slices:
                                        attn = attn_full[:, t0:t1].contiguous()
                                        # cheap early skip on all-pad chunks
                                        if attn.sum().item() == 0:
                                            continue

                                        h = h_full[:, t0:t1, :].contiguous()

                                        out = lens(
                                            h,
                                            layer=lid,
                                            labels=None,
                                            attention_mask=None,
                                            return_logits=True,
                                            return_loss=False,
                                        )
                                        if out.logits is None:
                                            raise RuntimeError("Lens.forward did not return logits.")

                                        # Teacher logprobs for this chunk in autocast dtype (bf16/fp16)
                                        # No explicit fp32 materialization of [B,t,V]
                                        with torch.no_grad():
                                            t_logits = teacher_logits_full[:, t0:t1, :].contiguous()
                                            t_logprobs = torch.log_softmax(t_logits, dim=-1).contiguous()

                                        chunk_sum = masked_kl_logtarget(
                                            student_logits=out.logits,
                                            teacher_logprobs=t_logprobs,
                                            attention_mask=attn,
                                            reduction="sum",
                                        )
                                        loss_sum = loss_sum + chunk_sum
                                        denom = denom + attn.sum(dtype=torch.float32)

                                        del out, chunk_sum, t_logprobs, t_logits, h, attn

                                    layer_loss = loss_sum / denom.clamp_min(1.0)

                                else:
                                    out = lens(
                                        h_full,
                                        layer=lid,
                                        labels=labels_full,
                                        attention_mask=attn_full,
                                        return_logits=False,
                                        return_loss=True,
                                    )
                                    if out.loss is None:
                                        raise RuntimeError("Lens.forward did not return loss.")
                                    layer_loss = out.loss

                                layer_loss = layer_loss * inv_layers * scale
                                accum_loss_local = accum_loss_local + layer_loss.detach().float()

                            if scaler.is_enabled():
                                scaler.scale(layer_loss).backward()
                            else:
                                layer_loss.backward()

                        del layer_loss

                del acts, teacher_logits_full, attn_full, labels_full

                if target_tokens is None:
                    break
                if accum_global_tokens >= float(target_tokens):
                    break
                if microsteps >= self.max_microsteps:
                    break

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                if self.grad_clip_norm is not None:
                    clip_grad_norm_([p for p in lens.parameters() if p.grad is not None], self.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if self.grad_clip_norm is not None:
                    clip_grad_norm_([p for p in lens.parameters() if p.grad is not None], self.grad_clip_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if ddp_state.device.type == "cuda":
                torch.cuda.synchronize(ddp_state.device)
            dt = time.perf_counter() - t_step0

            total_tokens_seen += int(accum_global_tokens)

            loss_sum = all_reduce_sum(accum_loss_local, ddp_state)
            mean_loss_global = loss_sum / float(ddp_state.world_size)

            n2b = bpb.nats_to_bpb_factor() if self.report_bits_per_byte else None
            reported = (mean_loss_global * float(n2b)) if (n2b is not None) else mean_loss_global

            tok_per_s = float(accum_global_tokens) / max(dt, 1e-9)
            mem_lines = gather_peak_mem_lines(ddp_state) if (step % max(self.log_mem_every, 1) == 0) else ""

            if ddp_state.is_main and self.log_every and (step == 1 or step % self.log_every == 0 or step == self.num_steps):
                lr = optimizer.param_groups[0]["lr"]
                if n2b is not None:
                    logger.info(
                        f"[Train] step {step}/{self.num_steps} "
                        f"loss={reported.item():.4f} (bits/byte) mean_nats={mean_loss_global.item():.4f} "
                        f"tok/s={tok_per_s:.0f} dt={dt*1000:.1f}ms "
                        f"tokens={total_tokens_seen} micro={microsteps} toks_step={int(accum_global_tokens)} "
                        f"lr={lr:.2e}{mem_lines}"
                    )
                else:
                    logger.info(
                        f"[Train] step {step}/{self.num_steps} "
                        f"loss={mean_loss_global.item():.4f} (nats/token) "
                        f"tok/s={tok_per_s:.0f} dt={dt*1000:.1f}ms "
                        f"tokens={total_tokens_seen} micro={microsteps} toks_step={int(accum_global_tokens)} "
                        f"lr={lr:.2e}{mem_lines}"
                    )

            if ddp_state.is_main and (step % self.save_every == 0 or step == self.num_steps):
                self._save_lens(
                    lens=lens,
                    layer_ids=layer_ids,
                    hidden_size=hidden_size,
                    tokenizer=tokenizer,
                    path=out_path,
                    global_step=step,
                    total_tokens=total_tokens_seen,
                    nats_to_bpb=n2b,
                )

    def _save_lens(
        self,
        *,
        lens,
        layer_ids,
        hidden_size: int,
        tokenizer,
        path: Path,
        global_step: int,
        total_tokens: int,
        nats_to_bpb: Optional[float],
    ) -> None:
        module = lens.module if isinstance(lens, DDP) else lens
        obj = {
            "train_config": {
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "lens_type": self.lens_type,
                "max_seq_len": self.max_seq_len,
                "stride": self.stride,
                "token_shift": self.token_shift,
                "loss": self.loss.value if isinstance(self.loss, LossChoice) else str(self.loss),
                "global_step": global_step,
                "total_tokens": total_tokens,
                "report_bits_per_byte": self.report_bits_per_byte,
                "nats_to_bpb_estimate": nats_to_bpb,
                "amp": self.amp,
                "amp_dtype": self.amp_dtype,
                "tokens_per_step": self.tokens_per_step,
                "kl_t_chunk": self.kl_t_chunk,
            },
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
            "layer_ids": layer_ids,
            "hidden_size": hidden_size,
            "state_dict": module.state_dict(),
        }
        torch.save(obj, path)
        logger.info(f"Saved lens checkpoint to {path}")