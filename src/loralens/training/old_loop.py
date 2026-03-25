# src/loralens/training/loop.py
#
# Changes:
# - Gradient accumulation to target a global tokens_per_step (tuned-lens-style large effective batch)
# - DDP no_sync() on non-sync microsteps (reduces comms cost)
# - Layer-wise backward for KL (memory optimization) + log_target KL (avoids softmax probs tensor)
# - Data/packing diagnostics: padding fraction, per-sample lengths, tokens/sample, tokens/microstep
# - DDP fix for layer-wise backward: find_unused_parameters=True
# - Logs iteration speed (tok/s), dt, and peak GPU memory each logged step
#
# Run with torchrun, e.g.:
#   cd /.../loralens/src
#   torchrun --standalone --nproc_per_node=4 -m loralens.training.train_lens

from __future__ import annotations

import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from loralens.data.loader import (
    ChunkConfig,
    ChunkedTextDataset,
    TextSource,
    TextSourceConfig,
    fixed_length_collate_fn,
)
from loralens.data.hf_adapter import StreamingTextSource
from loralens.data.hf_pile import PileSource, PileSourceConfig
from loralens.lenses import LogitLens, LoRALens, TunedLens

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


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    m = mask.to(x.dtype)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


def _masked_kl_logtarget(
    *,
    student_logits: torch.Tensor,      # [B,T,V]
    teacher_logprobs: torch.Tensor,    # [B,T,V] (no grad)
    attention_mask: Optional[torch.Tensor],  # [B,T]
) -> torch.Tensor:
    """
    KL(p_teacher || p_student) averaged over non-masked tokens.

    Uses log_target=True to avoid materializing teacher probabilities tensor.
    """
    log_q = F.log_softmax(student_logits, dim=-1)  # [B,T,V]
    kl_elem = F.kl_div(log_q, teacher_logprobs, log_target=True, reduction="none")  # [B,T,V]
    kl_per_tok = kl_elem.sum(dim=-1)  # [B,T]
    return _masked_mean(kl_per_tok, attention_mask)

class HFUnembed(nn.Module):
    """
    Approximation of tuned_lens.nn.unembed.Unembed for HF causal LMs.
    For GPT-2-like models this applies ln_f before lm_head.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Prefer GPT-2 style: model.transformer.ln_f + model.lm_head
        self.ln_f = None
        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            self.ln_f = model.transformer.ln_f

        if hasattr(model, "lm_head"):
            self.lm_head = model.lm_head
        elif hasattr(model, "get_output_embeddings"):
            self.lm_head = model.get_output_embeddings()
        else:
            raise ValueError("Could not locate LM head (lm_head / output_embeddings).")

        # attempt to expose vocab_size
        if hasattr(self.lm_head, "out_features"):
            self.vocab_size = int(self.lm_head.out_features)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.ln_f is not None:
            h = self.ln_f(h)
        return self.lm_head(h)

class _ShardedIterable(Iterable[str]):
    """
    Shard an arbitrary iterable of documents across DDP ranks by index mod world_size.
    Works for streaming sources without requiring a DistributedSampler.
    """

    def __init__(self, base: Iterable[str], *, rank: int, world_size: int, start_offset: int = 0):
        self.base = base
        self.rank = rank
        self.world_size = world_size
        self.start_offset = start_offset

    def __iter__(self) -> Iterator[str]:
        for i, doc in enumerate(self.base):
            if ((i + self.start_offset) % self.world_size) == self.rank:
                yield doc


def _estimate_batch_bytes(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> int:
    # Approximate bytes by decoding non-padding tokens back to text and counting UTF-8 bytes.
    # This is only for bits/byte reporting; it is intentionally approximate.
    bsz = input_ids.size(0)
    total = 0
    for i in range(bsz):
        valid = attention_mask[i].to(torch.bool)
        ids = input_ids[i][valid].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        total += len(text.encode("utf-8"))
    return total


@dataclass
class Train:
    # Model / tokenizer
    model_name: str = "gpt2"
    model_revision: Optional[str] = None
    tokenizer_name: Optional[str] = None

    # Data
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

    # Lens
    lens_type: LensType = "tuned"
    lora_rank: int = 16

    # Optimization
    per_gpu_batch_size: int = 1
    num_steps: int = 1000            # optimizer steps
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_clip_norm: Optional[float] = 1.0

    token_shift: Optional[int] = None
    loss: LossType = LossChoice.KL  # "kl" or "ce"

    # Gradient accumulation target (GLOBAL tokens per optimizer step).
    # tuned-lens commonly uses ~2**18 = 262144 global tokens/step.
    tokens_per_step: Optional[int] = 262_144
    max_microsteps: int = 512  # safety cap if tokens_per_step isn't reached due to padding/short chunks

    # Output / logging
    output: Path = Path("lens.pt")
    seed: int = 0
    log_every: int = 1
    save_every: int = 100
    checkpoint_dir: Optional[Path] = None
    checkpoint_freq: Optional[int] = None

    # Comparable reporting
    report_bits_per_byte: bool = True

    # DDP
    ddp: bool = True
    ddp_backend: str = "nccl"

    # Mixed precision + perf/mem logging
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"  # A100: bf16 recommended
    log_mem_every: int = 1

    # Data checks
    log_data_every: int = 10  # print packing diagnostics every N optimizer steps

    @dataclass
    class _DDPState:
        enabled: bool
        rank: int
        world_size: int
        local_rank: int
        device: torch.device

        @property
        def is_main(self) -> bool:
            return self.rank == 0

    def _maybe_init_ddp(self) -> _DDPState:
        want = self.ddp and ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)
        if not want:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return self._DDPState(enabled=False, rank=0, world_size=1, local_rank=0, device=device)

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        dist.init_process_group(backend=self.ddp_backend, rank=rank, world_size=world_size)
        dist.barrier()
        return self._DDPState(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank, device=device)

    def _autocast_ctx(self, device: torch.device):
        if not self.amp or device.type != "cuda":
            return nullcontext()
        if self.amp_dtype == "bf16":
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if self.amp_dtype == "fp16":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        raise ValueError(f"Unknown amp_dtype={self.amp_dtype!r}")

    def _make_scaler(self):
        if not self.amp or self.amp_dtype != "fp16":
            return torch.cuda.amp.GradScaler(enabled=False)
        return torch.cuda.amp.GradScaler(enabled=True)

    def _all_reduce_sum(self, x: torch.Tensor, ddp_state: _DDPState) -> torch.Tensor:
        if not ddp_state.enabled:
            return x
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    def _gather_vec(self, vec: torch.Tensor, ddp_state: _DDPState) -> List[torch.Tensor]:
        if not ddp_state.enabled:
            return [vec]
        outs = [torch.zeros_like(vec) for _ in range(ddp_state.world_size)]
        dist.all_gather(outs, vec)
        return outs

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

        # tuned-lens defaults: KL shift 0, CE shift 1
        if self.token_shift is None:
            self.token_shift = 0 if self.loss == LossChoice.KL else 1

        ddp_state = self._maybe_init_ddp()
        _set_seed(self.seed + ddp_state.rank)

        if ddp_state.is_main:
            logger.info(f"DDP: enabled={ddp_state.enabled} world_size={ddp_state.world_size}")
            logger.info(f"Device: {ddp_state.device}")
            logger.info(f"Loss: {self.loss.value} token_shift={self.token_shift}")
            logger.info(f"AMP: {self.amp} amp_dtype={self.amp_dtype}")
            if self.tokens_per_step is not None:
                logger.info(f"tokens_per_step (global): {self.tokens_per_step}")

        tokenizer = self._load_tokenizer(ddp_state)
        model = self._load_model(tokenizer=tokenizer, device=ddp_state.device, ddp_state=ddp_state)

        dataloader = self._build_dataloader(tokenizer, ddp_state=ddp_state)
        data_iter = iter(dataloader)

        lens_dtype = torch.bfloat16 if (self.amp and self.amp_dtype == "bf16") else (
            torch.float16 if (self.amp and self.amp_dtype == "fp16") else torch.float32
        )

        lens, layer_ids, hidden_size = self._build_lens(
            model=model,
            dtype=lens_dtype,
            device=ddp_state.device,
            ddp_state=ddp_state,
        )

        if ddp_state.enabled:
            # Required when doing layer-wise backward (many params unused per backward)
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

        scaler = self._make_scaler()

        self._train_loop(
            model=model,
            lens=lens,
            layer_ids=layer_ids,
            tokenizer=tokenizer,
            ddp_state=ddp_state,
            optimizer=optimizer,
            scheduler=scheduler,
            data_iter=data_iter,
            scaler=scaler,
            hidden_size=hidden_size,
        )

        if ddp_state.enabled:
            dist.barrier()
            dist.destroy_process_group()

    # ---------------- Model/tokenizer ----------------

    def _load_tokenizer(self, ddp_state: _DDPState):
        tok_name = self.tokenizer_name or self.model_name
        if ddp_state.is_main:
            logger.info(f"Loading tokenizer: {tok_name}")
        tokenizer = AutoTokenizer.from_pretrained(tok_name, revision=self.model_revision)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<pad>"

        if (
            hasattr(tokenizer, "model_max_length")
            and tokenizer.model_max_length is not None
            and tokenizer.model_max_length < self.max_seq_len
        ):
            tokenizer.model_max_length = self.max_seq_len

        return tokenizer

    def _load_model(self, tokenizer, device: torch.device, ddp_state: _DDPState):
        if ddp_state.is_main:
            logger.info(f"Loading model: {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision)
        model.resize_token_embeddings(len(tokenizer))

        for p in model.parameters():
            p.requires_grad = False

        model.to(device)
        model.eval()
        return model

    # ---------------- Data ----------------

    def _build_text_source(self, ddp_state: _DDPState):
        if self.data_source == "text":
            if not self.text_paths:
                raise ValueError("data_source='text' but no text_paths were provided.")
            cfg = TextSourceConfig(paths=self.text_paths, mode=self.text_mode, json_field=self.text_json_field)
            base = TextSource(cfg)
        elif self.data_source == "pile":
            pile_cfg = PileSourceConfig(split=self.pile_split)
            pile_source = PileSource(pile_cfg)
            base = StreamingTextSource(iterable=pile_source)
        else:
            raise ValueError(f"Unknown data_source={self.data_source!r}")

        if ddp_state.enabled and ddp_state.world_size > 1:
            base = _ShardedIterable(base, rank=ddp_state.rank, world_size=ddp_state.world_size, start_offset=self.seed)

        return base

    def _build_dataloader(self, tokenizer, ddp_state: _DDPState):
        text_source = self._build_text_source(ddp_state=ddp_state)
        stride = self.stride if self.stride is not None else self.max_seq_len

        chunk_cfg = ChunkConfig(
            seq_len=self.max_seq_len,
            stride=stride,
            drop_remainder=self.drop_remainder,
            add_special_tokens=True,
            document_separated=self.document_separated,
        )

        dataset = ChunkedTextDataset(
            text_source=text_source,
            tokenizer=tokenizer,
            chunk_cfg=chunk_cfg,
            max_docs=self.max_docs,
        )

        return DataLoader(
            dataset,
            batch_size=self.per_gpu_batch_size,
            collate_fn=fixed_length_collate_fn,
        )

    # ---------------- Lens ----------------

    def _get_readout_and_config(self, model):
        unembed = HFUnembed(model)

        cfg = model.config
        if hasattr(cfg, "n_layer"):
            num_layers = int(cfg.n_layer)
        elif hasattr(cfg, "num_hidden_layers"):
            num_layers = int(cfg.num_hidden_layers)
        else:
            raise ValueError("Model config lacks n_layer / num_hidden_layers.")

        if hasattr(cfg, "n_embd"):
            hidden_size = int(cfg.n_embd)
        elif hasattr(cfg, "hidden_size"):
            hidden_size = int(cfg.hidden_size)
        else:
            raise ValueError("Model config lacks n_embd / hidden_size.")

        vocab_size = getattr(unembed, "vocab_size", None)
        return unembed, num_layers, hidden_size, vocab_size

    def _build_lens(self, model, dtype: torch.dtype, device: torch.device, ddp_state):
        unembed, num_layers, hidden_size, vocab_size = self._get_readout_and_config(model)

        layer_ids = [str(i) for i in range(num_layers)]

        if ddp_state.is_main:
            logger.info(f"Building {self.lens_type!r} lens for {num_layers} layers (hidden_size={hidden_size}).")

        if self.lens_type == "tuned":
            # tuned-lens repo does not include final layer translator
            layer_ids_for_train = [str(i) for i in range(num_layers - 1)]  # or num_layers-1 for exact match
            lens = TunedLens(
                layer_ids=layer_ids_for_train,
                hidden_size=hidden_size,
                unembed=unembed,
                vocab_size=vocab_size,
                init_zero=True,
            )
            layer_ids = layer_ids_for_train
        elif self.lens_type == "logit":
            lens = LogitLens(readout=unembed)  # if your LogitLens expects readout-like module
        elif self.lens_type == "lora":
            lens = LoRALens(...)

        lens.to(device=device, dtype=dtype)
        lens.train()
        return lens, layer_ids, hidden_size

    # ---------------- Training ----------------

    def _next_batch(self, data_iter, tokenizer, ddp_state: _DDPState):
        try:
            batch = next(data_iter)
            return batch, data_iter
        except StopIteration:
            # If dataset is finite, restart; if streaming, this typically won't happen.
            dataloader = self._build_dataloader(tokenizer, ddp_state=ddp_state)
            data_iter = iter(dataloader)
            batch = next(data_iter)
            return batch, data_iter

    def _train_loop(
        self,
        *,
        model,
        lens,
        layer_ids: List[str],
        tokenizer,
        ddp_state: _DDPState,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        data_iter,
        scaler: torch.cuda.amp.GradScaler,
        hidden_size: int,
    ) -> None:
        out_path = self.output
        if ddp_state.is_main:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if self.checkpoint_dir is not None:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        total_tokens_seen = 0
        total_tokens_for_ratio = 0.0
        total_bytes_for_ratio = 0.0

        is_ddp = isinstance(lens, DDP)

        for step in range(1, self.num_steps + 1):
            if ddp_state.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(ddp_state.device)
                torch.cuda.synchronize(ddp_state.device)
            t0 = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            # Accumulate until we hit tokens_per_step (global) or max_microsteps
            target_tokens = self.tokens_per_step
            accum_global_tokens = 0.0
            accum_local_loss = torch.zeros((), device=ddp_state.device, dtype=torch.float32)
            microsteps = 0

            # For data diagnostics: track first microstep stats
            diag = {}

            while True:
                microsteps += 1
                batch, data_iter = self._next_batch(data_iter, tokenizer, ddp_state)

                input_ids = batch["input_ids"].to(ddp_state.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(ddp_state.device, non_blocking=True)

                # Compute local and global token count for this microbatch (masked tokens)
                tok_local = attention_mask.sum().to(torch.float32)  # scalar
                tok_global = self._all_reduce_sum(tok_local, ddp_state)
                tokens_this_micro = float(tok_global.item())
                accum_global_tokens += tokens_this_micro

                # Data diagnostics (only once per optimizer step)
                if ddp_state.is_main and (step % max(self.log_data_every, 1) == 0) and (microsteps == 1):
                    # local per-sample lengths on rank 0
                    lengths = attention_mask.sum(dim=1).detach().cpu()
                    diag = {
                        "min_len": int(lengths.min().item()) if lengths.numel() else 0,
                        "max_len": int(lengths.max().item()) if lengths.numel() else 0,
                        "mean_len": float(lengths.float().mean().item()) if lengths.numel() else 0.0,
                        "pad_frac": float((1.0 - attention_mask.float().mean()).item()),
                        "bsz": int(attention_mask.size(0)),
                        "seq": int(attention_mask.size(1)),
                    }

                # bits/byte accounting (global)
                if self.report_bits_per_byte:
                    with torch.no_grad():
                        bytes_this = _estimate_batch_bytes(tokenizer, input_ids, attention_mask)
                    bt = torch.tensor([float(tok_local.item()), float(max(bytes_this, 1))], device=ddp_state.device)
                    bt = self._all_reduce_sum(bt, ddp_state)
                    total_tokens_for_ratio += float(bt[0].item())
                    total_bytes_for_ratio += float(bt[1].item())

                # Teacher forward (frozen)
                with torch.no_grad():
                    with self._autocast_ctx(ddp_state.device):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                        )

                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("Model did not return hidden_states.")

                per_layer_acts = hidden_states[1:]  # drop embedding
                shift = max(int(self.token_shift or 0), 0)

                if shift > 0:
                    labels = input_ids[:, shift:].contiguous()
                    attn = attention_mask[:, shift:].contiguous()
                    acts = [h[:, :-shift, :].contiguous() for h in per_layer_acts]
                    teacher_logits = outputs.logits[:, shift:, :].contiguous()
                else:
                    labels = input_ids
                    attn = attention_mask
                    acts = [h.contiguous() for h in per_layer_acts]
                    teacher_logits = outputs.logits

                # Teacher logprobs once (no grad)
                with torch.no_grad():
                    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)

                n_layers = len(acts)
                inv_layers = 1.0 / float(n_layers)

                # DDP no_sync for non-final microsteps (avoid allreduce each microbatch)
                micro_sync = (target_tokens is None) or (accum_global_tokens >= float(target_tokens)) or (microsteps >= self.max_microsteps)
                micro_ctx = nullcontext() if (not is_ddp or micro_sync) else lens.no_sync()  # type: ignore[attr-defined]

                with micro_ctx:
                    # Layer-wise backward (memory optimization) + per-layer DDP no_sync for all but last layer (within the microstep)
                    for li, (layer_id, h) in enumerate(zip(layer_ids, acts)):
                        # If we're already in micro no_sync(), no need to manage layer no_sync().
                        if is_ddp and micro_sync:
                            layer_sync_ctx = nullcontext() if (li == n_layers - 1) else lens.no_sync()  # type: ignore[attr-defined]
                        else:
                            layer_sync_ctx = nullcontext()

                        with layer_sync_ctx:
                            with self._autocast_ctx(ddp_state.device):
                                if self.loss == LossChoice.CE:
                                    out = lens(
                                        h,
                                        layer=layer_id,
                                        labels=labels,
                                        attention_mask=attn,
                                        return_logits=False,
                                        return_loss=True,
                                    )
                                    if out.loss is None:
                                        raise RuntimeError("Lens.forward did not return a loss.")
                                    layer_loss = out.loss
                                elif self.loss == LossChoice.KL:
                                    out = lens(
                                        h,
                                        layer=layer_id,
                                        labels=None,
                                        attention_mask=None,
                                        return_logits=True,
                                        return_loss=False,
                                    )
                                    if out.logits is None:
                                        raise RuntimeError("Lens.forward did not return logits.")
                                    layer_loss = _masked_kl_logtarget(
                                        student_logits=out.logits,
                                        teacher_logprobs=teacher_logprobs,
                                        attention_mask=attn,
                                    )
                                else:
                                    raise ValueError(f"Unknown loss type: {self.loss}")

                                # Keep objective as mean over layers, then scale for grad accumulation
                                layer_loss = layer_loss * inv_layers

                                # Scale by accumulation: target is mean loss per microbatch; we average by number of microsteps
                                # so effective objective stays stable when changing tokens_per_step.
                                # (We use microsteps later once known; approximate with 1/max_microsteps is worse.)
                                # Here: scale by 1, and after loop we divide grads? Not possible.
                                # Instead: use 1/microsteps at the time of backward by deferring scaling:
                                # We scale each microstep by 1 (then we will scale by 1/microsteps at step end by adjusting LR).
                                # Better: scale by tokens ratio: (tokens_this_micro / accum_target_tokens).
                                # We'll do token-proportional scaling so the step approximates per-token mean.
                                if target_tokens is not None:
                                    scale = float(tokens_this_micro) / float(target_tokens)
                                else:
                                    scale = 1.0
                                layer_loss = layer_loss * scale

                                accum_local_loss = accum_local_loss + layer_loss.detach().float()

                            if scaler.is_enabled():
                                scaler.scale(layer_loss).backward()
                            else:
                                layer_loss.backward()

                        del out, layer_loss

                del teacher_logprobs, teacher_logits, acts, per_layer_acts, hidden_states, outputs

                # Decide whether to stop accumulating
                if target_tokens is None:
                    break
                if accum_global_tokens >= float(target_tokens):
                    break
                if microsteps >= self.max_microsteps:
                    break

            # Finish optimizer step
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
            dt = time.perf_counter() - t0

            total_tokens_seen += int(accum_global_tokens)

            # All-reduce loss for logging (average over ranks)
            loss_sum = self._all_reduce_sum(accum_local_loss, ddp_state)
            mean_loss_global = loss_sum / float(ddp_state.world_size)

            tok_per_s = float(accum_global_tokens) / max(dt, 1e-9)

            # tuned-lens comparable bits/byte factor
            reported = mean_loss_global
            nats_to_bpb = None
            if self.report_bits_per_byte and total_bytes_for_ratio > 0:
                nats_to_bpb = (total_tokens_for_ratio / total_bytes_for_ratio) / math.log(2.0)
                reported = mean_loss_global * float(nats_to_bpb)

            # Peak GPU memory (gathered)
            mem_lines = ""
            if ddp_state.device.type == "cuda" and (step % max(self.log_mem_every, 1) == 0):
                peak_alloc = float(torch.cuda.max_memory_allocated(ddp_state.device))
                peak_reserved = float(torch.cuda.max_memory_reserved(ddp_state.device))
                mem_vec = torch.tensor([peak_alloc, peak_reserved], device=ddp_state.device, dtype=torch.float64)
                gathered = self._gather_vec(mem_vec, ddp_state)
                if ddp_state.is_main:
                    parts = []
                    for r, m in enumerate(gathered):
                        a = float(m[0].item()) / (1024**2)
                        rs = float(m[1].item()) / (1024**2)
                        parts.append(f"gpu{r}:alloc={a:.0f}MiB,resv={rs:.0f}MiB")
                    mem_lines = " | " + " ".join(parts)

            # Logging (rank 0 only)
            if ddp_state.is_main and self.log_every and (step == 1 or step % self.log_every == 0 or step == self.num_steps):
                lr = optimizer.param_groups[0]["lr"]

                diag_str = ""
                if diag:
                    # tokens/sample (rank0 local) for quick sanity. global tokens/microstep already reflected in accum_global_tokens.
                    diag_str = (
                        f" | pack:bs={diag['bsz']} seq={diag['seq']} "
                        f"min={diag['min_len']} max={diag['max_len']} mean={diag['mean_len']:.1f} "
                        f"pad={diag['pad_frac']*100:.1f}%"
                    )

                if nats_to_bpb is not None:
                    logger.info(
                        f"[Train] step {step}/{self.num_steps} "
                        f"loss={reported.item():.4f} (bits/byte) "
                        f"mean_nats={mean_loss_global.item():.4f} "
                        f"tok/s={tok_per_s:.0f} "
                        f"dt={dt*1000:.1f}ms "
                        f"tokens={total_tokens_seen} "
                        f"micro={microsteps} "
                        f"toks_step={int(accum_global_tokens)} "
                        f"lr={lr:.2e}"
                        f"{mem_lines}"
                        f"{diag_str}"
                    )
                else:
                    logger.info(
                        f"[Train] step {step}/{self.num_steps} "
                        f"loss={mean_loss_global.item():.4f} (nats/token) "
                        f"tok/s={tok_per_s:.0f} "
                        f"dt={dt*1000:.1f}ms "
                        f"tokens={total_tokens_seen} "
                        f"micro={microsteps} "
                        f"toks_step={int(accum_global_tokens)} "
                        f"lr={lr:.2e}"
                        f"{mem_lines}"
                        f"{diag_str}"
                    )

            # Save (rank 0 only)
            if ddp_state.is_main and (step % self.save_every == 0 or step == self.num_steps):
                self._save_lens(
                    lens=lens,
                    tokenizer=tokenizer,
                    layer_ids=layer_ids,
                    hidden_size=hidden_size,
                    path=out_path,
                    global_step=step,
                    total_tokens=total_tokens_seen,
                    nats_to_bpb=(float(nats_to_bpb) if nats_to_bpb is not None else None),
                )

            if (
                ddp_state.is_main
                and self.checkpoint_dir is not None
                and self.checkpoint_freq is not None
                and step % self.checkpoint_freq == 0
            ):
                ckpt_path = self.checkpoint_dir / f"lens_step_{step}.pt"
                self._save_lens(
                    lens=lens,
                    tokenizer=tokenizer,
                    layer_ids=layer_ids,
                    hidden_size=hidden_size,
                    path=ckpt_path,
                    global_step=step,
                    total_tokens=total_tokens_seen,
                    nats_to_bpb=(float(nats_to_bpb) if nats_to_bpb is not None else None),
                )

    # ---------------- Saving ----------------

    def _save_lens(
        self,
        *,
        lens,
        tokenizer,
        layer_ids: List[str],
        hidden_size: int,
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
                "lora_rank": self.lora_rank,
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
                "layerwise_backward": True,
                "tokens_per_step": self.tokens_per_step,
            },
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
            "layer_ids": layer_ids,
            "hidden_size": hidden_size,
            "state_dict": module.state_dict(),
        }
        torch.save(obj, path)
        logger.info(f"Saved lens checkpoint to {path}")