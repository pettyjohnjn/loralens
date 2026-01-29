# src/loralens/cli/main.py
"""
Command-line interface for LoRA Lens training.

Usage:
    loralens train --model_name gpt2 --lens_type lora --loss_type subset_kl
    loralens train --help
"""

import argparse
import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from loralens.hooks import ActivationCollector
from loralens.losses import create_loss
from loralens.lenses import create_lens
from loralens.training import (
    LensTrainer,
    TrainConfig,
    HFUnembed,
    get_model_config,
    DDPState,
    init_ddp,
    shutdown_ddp,
    AMPContext,
)
from loralens.data import (
    TextSource,
    TextSourceConfig,
    ChunkConfig,
    ChunkedTextDataset,
    fixed_length_collate_fn,
)


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def set_seed(seed: int, ddp_state: DDPState) -> None:
    """Set random seeds."""
    random.seed(seed + ddp_state.rank)
    torch.manual_seed(seed + ddp_state.rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + ddp_state.rank)


def build_dataloader_factory(config: TrainConfig, tokenizer, ddp_state: DDPState):
    """Create a function that returns fresh DataLoaders."""
    def factory() -> DataLoader:
        # Build text source
        if config.data_source == "text":
            text_cfg = TextSourceConfig(
                paths=config.text_paths,
                mode=config.text_mode,
                json_field=config.text_json_field,
            )
            text_source = TextSource(text_cfg)
        else:
            # Pile source
            from loralens.data import PileSource, PileSourceConfig
            pile_cfg = PileSourceConfig(split=config.pile_split)
            pile_source = PileSource(pile_cfg)
            from loralens.data import StreamingTextSource
            text_source = StreamingTextSource(iterable=pile_source)
        
        # Chunk config
        chunk_cfg = ChunkConfig(
            seq_len=config.max_seq_len,
            stride=config.stride,
            drop_remainder=config.drop_remainder,
            document_separated=config.document_separated,
        )
        
        # Dataset
        dataset = ChunkedTextDataset(
            text_source=text_source,
            tokenizer=tokenizer,
            chunk_cfg=chunk_cfg,
            max_docs=config.max_docs,
        )
        
        return DataLoader(
            dataset,
            batch_size=config.per_gpu_batch_size,
            collate_fn=fixed_length_collate_fn,
            pin_memory=ddp_state.device.type == "cuda",
        )
    
    return factory


def train(args: argparse.Namespace) -> None:
    """Run training."""
    # Build config from args
    config = TrainConfig(
        model_name=args.model_name,
        data_source=args.data_source,
        lens_type=args.lens_type,
        lora_rank=args.lora_rank,
        loss_type=args.loss_type,
        kl_chunk_size=args.kl_chunk_size,
        subset_kl_k=args.subset_kl_k,
        shared_subset_top_m=args.shared_subset_top_m,
        shared_subset_max_K=args.shared_subset_max_K,
        max_seq_len=args.max_seq_len,
        per_gpu_batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        tokens_per_step=args.tokens_per_step,
        amp_enabled=args.amp,
        amp_dtype=args.amp_dtype,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
    
    # Initialize distributed
    ddp_state = init_ddp(enabled=args.ddp)
    set_seed(config.seed, ddp_state)
    
    if ddp_state.is_main:
        logger.info(f"Config: {config}")
        logger.info(f"DDP: {ddp_state}")
    
    try:
        # Load model and tokenizer
        if ddp_state.is_main:
            logger.info(f"Loading model: {config.model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Freeze model
        for p in model.parameters():
            p.requires_grad = False
        model.to(ddp_state.device)
        model.eval()
        
        # Get model config
        model_cfg = get_model_config(model)
        
        # Create components
        unembed = HFUnembed(model)
        
        # Create loss function
        loss_kwargs = {"reduction": "mean"}
        if config.loss_type == "kl":
            loss_kwargs["chunk_size"] = config.kl_chunk_size
        elif config.loss_type == "subset_kl":
            loss_kwargs["k"] = config.subset_kl_k
        elif config.loss_type == "shared_subset_kl":
            loss_kwargs["top_m"] = config.shared_subset_top_m
            loss_kwargs["max_K"] = config.shared_subset_max_K
        
        loss_fn = create_loss(config.loss_type, **loss_kwargs)
        
        # Create lens
        layer_ids = list(range(model_cfg["num_layers"]))
        lens_kwargs = {}
        if config.lens_type in ("tuned", "lora"):
            lens_kwargs["layer_ids"] = layer_ids
            lens_kwargs["hidden_size"] = model_cfg["hidden_size"]
            lens_kwargs["unembed"] = unembed
        if config.lens_type == "lora":
            lens_kwargs["r"] = config.lora_rank
        if config.lens_type == "logit":
            lens_kwargs["unembed"] = unembed
        
        lens = create_lens(config.lens_type, **lens_kwargs)
        
        # Move lens to device with appropriate dtype
        amp_ctx = AMPContext(
            enabled=config.amp_enabled,
            dtype=config.amp_dtype,
            device=ddp_state.device,
        )
        lens.to(device=ddp_state.device, dtype=amp_ctx.get_lens_dtype())
        lens.train()
        
        if ddp_state.is_main:
            logger.info(f"Lens: {lens}")
            logger.info(f"Loss: {loss_fn}")
        
        # Create collector
        collector = ActivationCollector(model)
        
        # Create trainer
        trainer = LensTrainer(
            model=model,
            lens=lens,
            loss_fn=loss_fn,
            collector=collector,
            config=config,
            ddp_state=ddp_state,
            amp_ctx=amp_ctx,
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            lens.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Create scheduler
        scheduler = None
        if config.warmup_steps > 0:
            def lr_lambda(step):
                if step < config.warmup_steps:
                    return (step + 1) / config.warmup_steps
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Create dataloader factory
        dataloader_factory = build_dataloader_factory(config, tokenizer, ddp_state)
        
        # Train
        trainer.train(dataloader_factory, optimizer, scheduler)
        
    finally:
        shutdown_ddp(ddp_state)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LoRA Lens - Scalable lens-based interpretability",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a lens")
    
    # Model
    train_parser.add_argument("--model_name", type=str, default="gpt2")
    train_parser.add_argument("--data_source", choices=["text", "pile"], default="pile")
    
    # Lens
    train_parser.add_argument("--lens_type", choices=["logit", "tuned", "lora"], default="lora")
    train_parser.add_argument("--lora_rank", type=int, default=16)
    
    # Loss
    train_parser.add_argument("--loss_type", choices=["kl", "subset_kl", "shared_subset_kl", "ce"], default="kl")
    train_parser.add_argument("--kl_chunk_size", type=int, default=128)
    train_parser.add_argument("--subset_kl_k", type=int, default=128, 
                              help="Number of top-k tokens for subset KL")
    train_parser.add_argument("--shared_subset_top_m", type=int, default=16,
                              help="Per-position candidates for shared subset KL")
    train_parser.add_argument("--shared_subset_max_K", type=int, default=512,
                              help="Max shared candidate set size")
    
    # Training
    train_parser.add_argument("--max_seq_len", type=int, default=1024)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--num_steps", type=int, default=1000)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--warmup_steps", type=int, default=0)
    train_parser.add_argument("--tokens_per_step", type=int, default=262144)
    
    # AMP
    train_parser.add_argument("--amp", action="store_true", default=True)
    train_parser.add_argument("--no_amp", action="store_false", dest="amp")
    train_parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    
    # DDP
    train_parser.add_argument("--ddp", action="store_true", default=True)
    train_parser.add_argument("--no_ddp", action="store_false", dest="ddp")
    
    # Logging
    train_parser.add_argument("--log_every", type=int, default=10)
    train_parser.add_argument("--save_every", type=int, default=500)
    train_parser.add_argument("--output_dir", type=str, default="./checkpoints")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--log_level", default="INFO")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    if args.command == "train":
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
