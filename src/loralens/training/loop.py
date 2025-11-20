# src/loralens/training/loop.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from loralens.data.loader import (
    TextSourceConfig,
    TextSource,
    ChunkConfig,
    ChunkedTextDataset,
    fixed_length_collate_fn,
)
from loralens.data.hf_pile import PileSource, PileSourceConfig
from loralens.data.hf_adapter import StreamingTextSource
from loralens.hooks.hook_manager import HookManager
from loralens.hooks.activation_hook import ActivationHook
from loralens.lenses.logit_lens import LogitLens
from loralens.lenses.tuned_lens import TunedLens
from loralens.lenses.lora_lens import LoRALens


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _layer_index_from_gpt2_name(name: str) -> Optional[int]:
    """
    Extract the block index from GPT-2 style module names like:
        'transformer.h.0.mlp.c_proj'
    Returns None if parsing fails.
    """
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "h" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL(student || teacher) with optional temperature.
    Shapes: [batch, seq, vocab] for both tensors.
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    log_p = F.log_softmax(student_logits, dim=-1)
    q = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


@dataclass
class Train:
    """
    Minimal single-GPU training loop for a lens on top of a HF CausalLM.

    - Can read from local text files via TextSource / ChunkedTextDataset.
    - Or stream The Pile via HuggingFace datasets when use_pile=True.
    - Uses distillation loss: KL between lens logits and final model logits.
    """

    # Model / tokenizer
    model_name: str = "sshleifer/tiny-gpt2"

    # Data source selection
    use_pile: bool = False
    pile_split: str = "train"
    pile_shuffle_buffer_size: int = 50_000

    # Local text data
    train_paths: List[str] = field(default_factory=list)
    text_mode: Literal["lines", "whole", "jsonl"] = "lines"
    json_field: str = "text"

    # Chunking
    seq_len: int = 128
    stride: Optional[int] = None
    drop_remainder: bool = True
    add_special_tokens: bool = True
    max_docs: Optional[int] = None

    batch_size: int = 8
    num_workers: int = 0

    # Training
    num_steps: int = 1_000
    lr: float = 1e-4
    weight_decay: float = 0.0
    distill_temperature: float = 1.0

    # Lens
    lens_type: Literal["lora", "tuned", "logit"] = "lora"
    lora_rank: int = 8
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    lora_freeze_base: bool = True

    # Device / logging / saving
    device: str = field(default_factory=_default_device)  # "cuda", "cpu", or "auto"
    log_every: int = 10
    save_path: Optional[str] = None  # path to torch.save(lens.state_dict())

    def execute(self) -> None:
        # Resolve device
        if self.device == "auto":
            device_str = _default_device()
        else:
            device_str = self.device
        device = torch.device(device_str)
        print(f"[Train] Using device: {device}")

        # Load model + tokenizer
        print(f"[Train] Loading model '{self.model_name}'")
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_name
        )

        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Infer config bits we need (GPT-2 style for now)
        n_layer = getattr(model.config, "n_layer", None)
        hidden_size = getattr(
            model.config, "n_embd", getattr(model.config, "hidden_size", None)
        )
        if n_layer is None or hidden_size is None:
            raise ValueError(
                "Training loop currently assumes a GPT-2-like config with "
                "config.n_layer and config.n_embd or config.hidden_size."
            )

        # Grab readout head; for GPT-2 this is lm_head
        readout = getattr(model, "lm_head", None)
        if readout is None or not isinstance(readout, nn.Module):
            raise ValueError(
                "Could not find a usable readout head on the model. "
                "For now, this training loop expects `model.lm_head`."
            )

        layer_ids = list(range(n_layer))

        # Construct lens
        if self.lens_type == "lora":
            lens = LoRALens(
                layer_ids=layer_ids,
                hidden_size=hidden_size,
                readout=readout,
                r=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                freeze_base=self.lora_freeze_base,
            )
        elif self.lens_type == "tuned":
            lens = TunedLens(
                layer_ids=layer_ids,
                hidden_size=hidden_size,
                readout=readout,
            )
        elif self.lens_type == "logit":
            lens = LogitLens(readout=readout)
        else:
            raise ValueError(f"Unknown lens_type: {self.lens_type!r}")

        lens.to(device)
        lens.train()

        # Optimizer over lens parameters only
        optimizer = torch.optim.AdamW(
            (p for p in lens.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # -----------------------------------------------------
        # Dataset selection: local text OR Pile
        # -----------------------------------------------------
        if self.use_pile:
            print("[Train] Streaming The Pile via HuggingFace datasets")
            pile_cfg = PileSourceConfig(
                split=self.pile_split,
                streaming=True,
                shuffle_buffer_size=self.pile_shuffle_buffer_size,
                text_field="text",
            )
            raw_iter = PileSource(pile_cfg)  # yields raw text strings
            text_source = StreamingTextSource(raw_iter)
        else:
            if not self.train_paths:
                raise ValueError(
                    "Train.train_paths is empty and use_pile=False. "
                    "Provide at least one text file OR set --use_pile."
                )

            text_cfg = TextSourceConfig(
                paths=self.train_paths,
                mode=self.text_mode,
                json_field=self.json_field,
            )
            text_source = TextSource(text_cfg)

        chunk_cfg = ChunkConfig(
            seq_len=2048,
            stride=2048,
            drop_remainder=True,
            add_special_tokens=True,
            document_separated=False
        )
        dataset = ChunkedTextDataset(
            text_source=text_source,
            tokenizer=tokenizer,
            chunk_cfg=chunk_cfg,
            max_docs=self.max_docs,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=fixed_length_collate_fn,
        )

        # Hook setup: capture GPT-2 MLP projections transformer.h.{i}.mlp.c_proj
        manager = HookManager(model)
        layer_activations: Dict[int, torch.Tensor] = {}

        def predicate(name: str, module: nn.Module) -> bool:
            return name.endswith(".mlp.c_proj")

        def hook_factory(name: str, module: nn.Module) -> ActivationHook:
            layer_idx = _layer_index_from_gpt2_name(name)
            if layer_idx is None:
                raise ValueError(f"Could not parse layer index from module name {name!r}")

            def on_activation(
                x: torch.Tensor, module_name: str, mod: nn.Module
            ) -> None:
                # We store the raw activations; they are produced under no_grad
                # and will be used as inputs to the lens (which has gradients).
                layer_activations[layer_idx] = x.detach()

            return ActivationHook(
                name=f"mlp_c_proj_{layer_idx}",
                on_activation=on_activation,
            )

        manager.add_hooks_by_predicate(predicate, hook_factory)
        print(f"[Train] Registered {len(manager.list_hooks())} hooks on MLP projections")

        try:
            self._train_loop(
                model=model,
                lens=lens,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                layer_activations=layer_activations,
            )
        finally:
            manager.remove_all()

        if self.save_path is not None:
            print(f"[Train] Saving lens state_dict to: {self.save_path}")
            torch.save(lens.state_dict(), self.save_path)

    def _train_loop(
        self,
        *,
        model: nn.Module,
        lens: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        layer_activations: Dict[int, torch.Tensor],
    ) -> None:
        step = 0
        data_iter = iter(dataloader)

        while step < self.num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart the iterator if we exhaust the underlying iterable dataset
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Clear activations for this step
            layer_activations.clear()

            # Teacher forward pass, frozen model, no gradients
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                )
                teacher_logits = outputs.logits  # [batch, seq, vocab]
                teacher_logits = teacher_logits.detach()

            if not layer_activations:
                raise RuntimeError(
                    "No activations were captured by hooks during forward. "
                    "Check the hook predicate / factory."
                )

            # Lens forward + distillation loss
            optimizer.zero_grad(set_to_none=True)

            # Average KL over all hooked layers
            total_loss = 0.0
            n_layers_seen = 0

            for layer_id in sorted(layer_activations.keys()):
                activations = layer_activations[layer_id].to(device)  # [B, S, H]

                # For LogitLens, we ignore the layer id
                if isinstance(lens, LogitLens):
                    student_logits = lens.compute_logits(activations)
                else:
                    student_logits = lens.compute_logits(
                        activations, layer=layer_id
                    )

                loss_layer = _kl_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=self.distill_temperature,
                )
                total_loss = total_loss + loss_layer
                n_layers_seen += 1

            total_loss = total_loss / float(n_layers_seen)

            total_loss.backward()
            optimizer.step()

            if (step + 1) % self.log_every == 0:
                print(
                    f"[Train] Step {step + 1}/{self.num_steps} "
                    f"loss={total_loss.item():.4f} "
                    f"(layers={n_layers_seen})"
                )

            step += 1