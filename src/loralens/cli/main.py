# src/loralens/cli/main.py
"""
Command-line interface for LoRA Lens training.

Usage:
    loralens train --model_name gpt2 --lens_type lora --loss_type subset_kl
    loralens train --help
"""

import argparse
import gc
import logging
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from loralens.hooks import ActivationCollector
from loralens.initialization import LoRAInitConfig, initialize_lora_lens
from loralens.lenses import LoRALens, create_lens
from loralens.losses import create_loss
from loralens.training import (
    LensTrainer,
    TrainConfig,
    HFUnembed,
    get_model_config,
    adapt_activation_site_plan_for_model,
    build_activation_site_plan,
    DDPState,
    init_ddp,
    shutdown_ddp,
    AMPContext,
    load_sharded_model,
    load_sharded_model_multinode,
    disabled_shard_state,
)
from loralens.data import (
    TextSource,
    TextSourceConfig,
    ChunkConfig,
    ChunkedTextDataset,
    fixed_length_collate_fn,
)


logger = logging.getLogger(__name__)


def _checkpoint_step_key(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return -1


def _resolve_resume_checkpoint_path(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint path does not exist: {path}")

    candidates = sorted(
        (p for p in path.glob("lens_step_*.pt") if p.is_file()),
        key=lambda p: (_checkpoint_step_key(p), p.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching lens_step_*.pt found in {path}")
    return candidates[-1]


def _load_resume_checkpoint(checkpoint_path: Path, lens: torch.nn.Module) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    lens.load_checkpoint_state_dict(checkpoint["lens_state_dict"])
    return checkpoint


def _validate_resume_checkpoint(config: TrainConfig, checkpoint: dict[str, Any], path: Path) -> None:
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ValueError(f"Checkpoint {path} is missing saved config metadata")

    expected_pairs = (
        ("model_name", config.model_name),
        ("lens_type", config.lens_type),
        ("loss_type", config.loss_type),
        ("lora_rank", config.lora_rank),
        ("lora_alpha", config.lora_alpha),
        ("subset_kl_mode", config.subset_kl_mode),
        ("subset_kl_k", config.subset_kl_k),
        ("subset_kl_k_tail", config.subset_kl_k_tail),
        ("kl_chunk_size", config.kl_chunk_size),
        ("activation_site_preset", config.activation_site_preset),
        ("token_shift", config.token_shift),
        ("max_seq_len", config.max_seq_len),
    )

    mismatches = []
    for key, expected in expected_pairs:
        observed = saved.get(key)
        if observed != expected:
            mismatches.append(f"{key}: checkpoint={observed!r}, requested={expected!r}")

    if mismatches:
        raise ValueError(
            f"Checkpoint {path} does not match the requested run configuration:\n"
            + "\n".join(mismatches)
        )


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )
    # Suppress noisy HTTP request logs from HuggingFace/datasets libraries
    for noisy_logger in (
        "httpx", "httpcore", "urllib3", "filelock",
        "datasets", "huggingface_hub",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


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
            pin_memory=torch.cuda.is_available(),
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
        lora_alpha=args.lora_alpha,
        lora_init=args.lora_init,
        lora_init_calibration_tokens=args.lora_init_calibration_tokens,
        lora_init_ridge_lambda=args.lora_init_ridge_lambda,
        lora_init_ridge_lambda_scale=args.lora_init_ridge_lambda_scale,
        lora_init_stats_dtype=args.lora_init_stats_dtype,
        lora_init_jitter=args.lora_init_jitter,
        lora_init_svd_metric=args.lora_init_svd_metric,
        lora_init_normalization=args.lora_init_normalization,
        activation_site_preset=args.activation_site_preset,
        write_loss_type=args.write_loss_type,
        write_loss_weight=args.write_loss_weight,
        loss_type=args.loss_type,
        kl_chunk_size=args.kl_chunk_size,
        subset_kl_k=args.subset_kl_k,
        subset_kl_mode=args.subset_kl_mode,
        subset_kl_k_tail=args.subset_kl_k_tail,
        subset_kl_tail_proposal=args.subset_kl_tail_proposal,
        subset_kl_tail_proposal_alpha=args.subset_kl_tail_proposal_alpha,
        subset_kl_tail_proposal_tau=args.subset_kl_tail_proposal_tau,
        token_shift=args.token_shift,
        max_seq_len=args.max_seq_len,
        per_gpu_batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        min_lr_ratio=args.min_lr_ratio,
        tokens_per_step=args.tokens_per_step,
        amp_enabled=args.amp,
        amp_dtype=args.amp_dtype,
        ddp_enabled=args.ddp,
        log_every=args.log_every,
        save_every=args.save_every,
        save_initial_checkpoint=args.save_initial_checkpoint,
        output_dir=Path(args.output_dir) if args.output_dir else Path("./checkpoints"),
        resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        seed=args.seed,
        run_tag=args.run_tag if args.run_tag else None,
        # Model parallel
        model_parallel=args.model_parallel,
        model_dtype=args.model_dtype,
        per_gpu_max_memory=getattr(args, "per_gpu_max_memory", None),
        cpu_offload_gb=getattr(args, "cpu_offload_gb", 0),
        attn_implementation=getattr(args, "attn_implementation", None),
        # Multi-node model parallel
        multinode_model_parallel=getattr(args, "multinode_model_parallel", False),
        fsdp_offload_to_cpu=getattr(args, "fsdp_offload_to_cpu", False),
    )

    # Derive canonical output path when --output_dir was not explicitly set
    if not args.output_dir:
        base = Path(args.base_output_dir)
        config.output_dir = base / config.canonical_run_path()
        logger.info("Output dir (auto): %s", config.output_dir)

    # Validate: can't use both single-node and multi-node MP
    if config.model_parallel and config.multinode_model_parallel:
        raise ValueError(
            "Cannot use both --model_parallel (single-node device_map) and "
            "--multinode_model_parallel (FSDP). Pick one."
        )

    # When single-node model parallel is on, DDP should be off
    if config.model_parallel and config.ddp_enabled:
        logger.warning(
            "Model parallel enabled - disabling DDP. "
            "Use a single process (no torchrun) for model-parallel training."
        )
        config.ddp_enabled = False

    # Multi-node MP requires DDP (torchrun sets up process groups)
    if config.multinode_model_parallel and not config.ddp_enabled:
        logger.warning("multinode_model_parallel requires DDP - enabling it.")
        config.ddp_enabled = True

    # Initialize distributed
    ddp_state = init_ddp(enabled=config.ddp_enabled)
    set_seed(config.seed, ddp_state)

    if ddp_state.is_main:
        logger.info(f"Config: {config}")
        logger.info(f"DDP: {ddp_state}")

    # Keep references explicit so we can tear them down on exit.
    tokenizer = None
    model = None
    unembed = None
    loss_fn = None
    lens = None
    collector = None
    trainer = None
    optimizer = None
    scheduler = None
    dataloader_factory = None
    shard_state = None
    activation_site_plan = None
    start_step = 0
    resumed_total_tokens = 0

    try:
        # Load model and tokenizer
        if ddp_state.is_main:
            logger.info(f"Loading model: {config.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        model_dtype = dtype_map.get(config.model_dtype, torch.bfloat16)

        # --- Model loading: three paths ---
        if config.multinode_model_parallel:
            # Path 1: Multi-node FSDP sharding
            # load_sharded_model_multinode loads to CPU first, then wraps
            # with FSDP. We must create unembed BEFORE FSDP wrapping
            # since FSDP shards the parameters.
            #
            # Load to CPU, create unembed before FSDP wrapping shards params

            from transformers import AutoModelForCausalLM as _AutoModel
            _cpu_model_kwargs = {
                "torch_dtype": model_dtype,
                "device_map": "cpu",
                "low_cpu_mem_usage": True,
            }
            if config.model_revision is not None:
                _cpu_model_kwargs["revision"] = config.model_revision
            if config.attn_implementation is not None:
                _cpu_model_kwargs["attn_implementation"] = config.attn_implementation

            _cpu_model = _AutoModel.from_pretrained(config.model_name, **_cpu_model_kwargs)
            for p in _cpu_model.parameters():
                p.requires_grad = False
            _cpu_model.eval()

            # Get config and create unembed BEFORE FSDP wrapping
            model_cfg = get_model_config(_cpu_model)
            activation_site_plan = build_activation_site_plan(
                _cpu_model,
                num_layers=model_cfg["num_layers"],
                preset=config.activation_site_preset,
            )
            unembed = HFUnembed(_cpu_model)
            unembed = unembed.clone_to_device(ddp_state.device)
            logger.info(f"Cloned unembed to {ddp_state.device} (pre-FSDP)")

            # Now wrap with FSDP
            model, _mn_shard = load_sharded_model_multinode(
                config.model_name,
                local_rank=ddp_state.local_rank,
                model_revision=config.model_revision,
                dtype=model_dtype,
                cpu_offload=config.fsdp_offload_to_cpu,
                attn_implementation=config.attn_implementation,
                _preloaded_model=_cpu_model,  # Pass already-loaded model
            )
            del _cpu_model
            activation_site_plan = adapt_activation_site_plan_for_model(
                model,
                activation_site_plan,
            )

            # Trainer uses standard DDP path (FSDP handles sharding transparently)
            shard_state = disabled_shard_state(ddp_state.device)
            lens_device = ddp_state.device
            if ddp_state.is_main:
                logger.info(f"FSDP sharding: {_mn_shard}")

        elif config.model_parallel:
            # Path 2: Single-node device_map sharding
            model, shard_state = load_sharded_model(
                config.model_name,
                model_revision=config.model_revision,
                dtype=model_dtype,
                per_gpu_max_memory=config.per_gpu_max_memory,
                cpu_offload_gb=config.cpu_offload_gb,
                attn_implementation=config.attn_implementation,
            )
            logger.info(f"Model sharding: {shard_state}")
            lens_device = shard_state.lens_device

        else:
            # Path 3: Standard single-GPU (with optional DDP replication)
            model_kwargs = {
                "torch_dtype": model_dtype,
                "low_cpu_mem_usage": True,
            }
            if config.model_revision is not None:
                model_kwargs["revision"] = config.model_revision
            if config.attn_implementation is not None:
                model_kwargs["attn_implementation"] = config.attn_implementation

            model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
            for p in model.parameters():
                p.requires_grad = False
            model.to(ddp_state.device)
            model.eval()
            shard_state = disabled_shard_state(ddp_state.device)
            lens_device = ddp_state.device

        # Get model config (skip for multinode, already done above)
        if not config.multinode_model_parallel:
            model_cfg = get_model_config(model)
            activation_site_plan = build_activation_site_plan(
                model,
                num_layers=model_cfg["num_layers"],
                preset=config.activation_site_preset,
            )

        # Create unembed (skip for multinode, already done above)
        if not config.multinode_model_parallel:
            unembed = HFUnembed(model)
            if config.model_parallel:
                unembed = unembed.clone_to_device(lens_device)
                logger.info(f"Cloned unembed to {lens_device}")

        if (
            ddp_state.is_main
            and config.lens_type == "tuned"
            and activation_site_plan.site_count > model_cfg["num_layers"]
        ):
            logger.warning(
                "Using tuned lenses with activation_site_preset=%s creates one full-rank "
                "translator per site (%s total sites). For Llama 3 8B this is likely too "
                "large for routine runs; use residual-only or switch to LoRA for expanded sites.",
                config.activation_site_preset,
                activation_site_plan.site_count,
            )

        # Create loss function
        loss_kwargs = {"reduction": "mean"}
        if config.loss_type == "kl":
            loss_kwargs["chunk_size"] = (
                config.kl_chunk_size if config.kl_chunk_size > 0 else None
            )
        elif config.loss_type == "subset_kl":
            loss_kwargs["k"] = config.subset_kl_k
            loss_kwargs["mode"] = config.subset_kl_mode
            loss_kwargs["k_tail"] = config.subset_kl_k_tail
            loss_kwargs["tail_proposal"] = config.subset_kl_tail_proposal
            loss_kwargs["tail_proposal_alpha"] = config.subset_kl_tail_proposal_alpha
            loss_kwargs["tail_proposal_tau"] = config.subset_kl_tail_proposal_tau

        loss_fn = create_loss(config.loss_type, **loss_kwargs)

        # Create lens
        layer_ids = activation_site_plan.site_ids
        lens_kwargs = {}
        if config.lens_type in ("tuned", "lora", "bidir_lora"):
            lens_kwargs["layer_ids"] = layer_ids
            lens_kwargs["hidden_size"] = model_cfg["hidden_size"]
            lens_kwargs["unembed"] = unembed
        if config.lens_type in ("lora", "bidir_lora"):
            lens_kwargs["r"] = config.lora_rank
            lens_kwargs["alpha"] = config.lora_alpha
        if config.lens_type == "logit":
            lens_kwargs["unembed"] = unembed

        lens = create_lens(config.lens_type, **lens_kwargs)

        # Move lens to lens device with appropriate dtype
        amp_ctx = AMPContext(
            enabled=config.amp_enabled,
            dtype=config.amp_dtype,
            device=lens_device,
        )
        lens.to(device=lens_device, dtype=amp_ctx.get_lens_dtype())
        lens.train()

        if config.resume_checkpoint is not None:
            config.resume_checkpoint = _resolve_resume_checkpoint_path(config.resume_checkpoint)
            if ddp_state.is_main:
                logger.info(f"Loading checkpoint: {config.resume_checkpoint}")
            checkpoint = _load_resume_checkpoint(config.resume_checkpoint, lens)
            _validate_resume_checkpoint(config, checkpoint, config.resume_checkpoint)
            start_step = int(checkpoint.get("step", 0))
            resumed_total_tokens = int(checkpoint.get("total_tokens", 0))
            if ddp_state.is_main:
                logger.info(
                    "Resuming from step %s with total_tokens=%s",
                    start_step,
                    resumed_total_tokens,
                )

        if ddp_state.is_main:
            logger.info(f"Lens: {lens}")
            logger.info(f"Loss: {loss_fn}")
            logger.info(
                "Activation sites: preset=%s count=%s",
                config.activation_site_preset,
                activation_site_plan.site_count,
            )

        # Create dataloader factory before optional calibration initialization.
        dataloader_factory = build_dataloader_factory(config, tokenizer, ddp_state)

        # Create collector
        collector = ActivationCollector(model, custom_hooks=activation_site_plan.custom_hooks)

        if config.lora_init != "default_lora":
            if config.lens_type not in ("lora", "bidir_lora") or not isinstance(lens, LoRALens):
                raise ValueError("--lora_init is only supported with --lens_type lora")
            if config.resume_checkpoint is not None:
                if ddp_state.is_main:
                    logger.warning(
                        "Skipping lora_init=%s because resume_checkpoint is set.",
                        config.lora_init,
                    )
            else:
                if ddp_state.is_main:
                    logger.info("Applying LoRA initialization: %s", config.lora_init)
                initialize_lora_lens(
                    lens=lens,
                    model=model,
                    collector=collector,
                    activation_site_plan=activation_site_plan,
                    dataloader_factory=dataloader_factory,
                    ddp_state=ddp_state,
                    lens_device=lens_device,
                    shard_state=shard_state,
                    config=LoRAInitConfig(
                        mode=config.lora_init,
                        calibration_tokens=config.lora_init_calibration_tokens,
                        ridge_lambda=config.lora_init_ridge_lambda,
                        ridge_lambda_scale=config.lora_init_ridge_lambda_scale,
                        stats_dtype=config.lora_init_stats_dtype,
                        jitter=config.lora_init_jitter,
                        svd_metric=config.lora_init_svd_metric,
                        normalization=config.lora_init_normalization,
                    ),
                    token_shift=config.token_shift,
                )

        # Create optimizer after optional initialization so optimizer state starts clean.
        optimizer = torch.optim.AdamW(
            lens.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Create scheduler
        scheduler = None
        if config.warmup_steps > 0 or config.lr_schedule != "constant":
            def lr_lambda(step):
                if step < config.warmup_steps:
                    return (step + 1) / config.warmup_steps
                if config.lr_schedule == "constant":
                    return 1.0

                decay_steps = max(1, config.num_steps - config.warmup_steps)
                progress = min(1.0, max(0.0, (step - config.warmup_steps + 1) / decay_steps))
                if config.lr_schedule == "linear":
                    decay = 1.0 - progress
                elif config.lr_schedule == "cosine":
                    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                else:
                    raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")
                return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * decay

            if start_step > 0:
                for group in optimizer.param_groups:
                    group.setdefault("initial_lr", config.lr)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda,
                    last_epoch=start_step - 1,
                )
            else:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Create trainer
        trainer = LensTrainer(
            model=model,
            lens=lens,
            loss_fn=loss_fn,
            collector=collector,
            activation_site_plan=activation_site_plan,
            config=config,
            ddp_state=ddp_state,
            amp_ctx=amp_ctx,
            shard_state=shard_state,
            start_step=start_step,
            start_total_tokens=resumed_total_tokens,
        )

        # Train
        trainer.train(dataloader_factory, optimizer, scheduler)

    finally:
        try:
            if collector is not None:
                collector.detach()
        except Exception:
            logger.exception("Collector cleanup failed")

        try:
            shutdown_ddp(ddp_state)
        except Exception:
            logger.exception("DDP shutdown failed")

        del trainer, optimizer, scheduler, dataloader_factory
        del collector, lens, loss_fn, unembed, model, tokenizer, shard_state

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                logger.exception("CUDA cleanup failed")


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
    train_parser.add_argument(
        "--lens_type",
        choices=["logit", "tuned", "lora", "bidir_lora"],
        default="lora",
        help=(
            "lora: standard read-only LoRA lens. "
            "bidir_lora: bidirectional LoRA lens — same architecture, adds write-direction "
            "training via --write_loss_type."
        ),
    )
    train_parser.add_argument("--lora_rank", type=int, default=16)
    train_parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha. Defaults to lora_rank when omitted.",
    )
    train_parser.add_argument(
        "--write_loss_type",
        choices=["none", "ortho", "suffix", "both"],
        default="none",
        help=(
            "Bidirectional write loss (only active with --lens_type bidir_lora). "
            "none: read-only training (identical to lora). "
            "ortho: add ||h - h@M.T@M||^2 penalty to encourage M to be orthogonal "
            "(transpose ≈ inverse, cheap). "
            "suffix: inject write encoding at a random resid_post site and run the model "
            "suffix to compute CE loss (directly trains write effectiveness, ~2x cost). "
            "both: apply both ortho and suffix losses simultaneously."
        ),
    )
    train_parser.add_argument(
        "--write_loss_weight",
        type=float,
        default=0.1,
        help="Weight applied to the write loss term(s) relative to the read loss.",
    )
    train_parser.add_argument(
        "--lora_init",
        choices=["default_lora", "mean_shift", "ridge_svd"],
        default="default_lora",
        help=(
            "LoRA translator initialization strategy. "
            "default_lora: standard Kaiming init (no data needed). "
            "mean_shift: shift B so B@A@h=0 at init, then train normally. "
            "ridge_svd: fit ridge regression to calibration data, truncate to LoRA rank — "
            "typically gives a much better starting loss."
        ),
    )
    train_parser.add_argument(
        "--lora_init_calibration_tokens",
        type=int,
        default=50000,
        help="Global non-padding token budget for data-driven LoRA initialization.",
    )
    train_parser.add_argument(
        "--lora_init_ridge_lambda",
        type=float,
        default=1e-3,
        help="Ridge coefficient for lora_init=ridge_svd.",
    )
    train_parser.add_argument(
        "--lora_init_ridge_lambda_scale",
        choices=["trace_xxt_over_d", "absolute"],
        default="trace_xxt_over_d",
        help=(
            "How to interpret lora_init_ridge_lambda. "
            "trace_xxt_over_d: multiply lambda by the mean squared activation magnitude "
            "(recommended — makes regularization scale-invariant across layers and models). "
            "absolute: use lambda as-is."
        ),
    )
    train_parser.add_argument(
        "--lora_init_stats_dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Accumulator dtype for LoRA calibration statistics.",
    )
    train_parser.add_argument(
        "--lora_init_jitter",
        type=float,
        default=1e-6,
        help="Initial diagonal jitter added if the ridge solve is ill-conditioned.",
    )
    train_parser.add_argument(
        "--lora_init_svd_metric",
        choices=["residual", "unembed"],
        default="residual",
        help="Metric used when truncating ridge Delta into LoRA factors.",
    )
    train_parser.add_argument(
        "--lora_init_normalization",
        choices=["none", "per_dim_std"],
        default="none",
        help="Optional feature normalization for ridge calibration before converting back to raw coordinates.",
    )
    train_parser.add_argument(
        "--activation_site_preset",
        choices=["residual", "llama_expanded", "gpt2_expanded", "gpt2_attention"],
        default="residual",
        help=(
            "Which hidden states to attach lenses to. "
            "residual: one lens per post-attention + post-MLP residual site per layer (default, works on any model). "
            "gpt2_expanded: also add pre-attention and pre-MLP sites (GPT-2 only). "
            "llama_expanded: same but for LLaMA/Mistral architectures. "
            "gpt2_attention: residual sites plus attention output (GPT-2 only)."
        ),
    )

    # Loss
    train_parser.add_argument("--loss_type", choices=["kl", "subset_kl", "ce"], default="kl")
    train_parser.add_argument("--kl_chunk_size", type=int, default=128)
    train_parser.add_argument("--subset_kl_k", type=int, default=128,
                              help="Number of top-k tokens for subset KL")
    train_parser.add_argument(
        "--subset_kl_mode",
        choices=["topk", "mc", "k2", "k3"],
        default="topk",
        help=(
            "Subset KL estimator. "
            "topk: renormalize KL over top-k tokens only — fast, recommended default. "
            "mc: exact KL on top-k head + importance-weighted MC estimate on the tail. "
            "k2: top-k head KL + Schulman K2 squared-error tail penalty. "
            "k3: top-k head KL + Schulman K3 tail estimator. "
            "Use --subset_kl_k_tail > 0 to enable the tail term for mc/k2/k3."
        ),
    )
    train_parser.add_argument("--subset_kl_k_tail", type=int, default=0,
                              help="Number of tail samples per token for mc/k2/k3 modes (0 = head only)")
    train_parser.add_argument(
        "--subset_kl_tail_proposal",
        choices=["target", "teacher", "mixed", "tempered"],
        default="target",
        help="Proposal distribution for tail sampling in mc/k2/k3 modes. target uses the teacher distribution.",
    )
    train_parser.add_argument("--subset_kl_tail_proposal_alpha", type=float, default=0.8,
                              help="Target mixture weight for mixed tail proposal")
    train_parser.add_argument("--subset_kl_tail_proposal_tau", type=float, default=0.7,
                              help="Tempering exponent for mixed/tempered tail proposal")
    train_parser.add_argument(
        "--token_shift",
        type=int,
        default=None,
        help=(
            "Token offset between teacher logits and target labels. "
            "0: teacher and student see the same position (KL distillation). "
            "1: predict position i+1's token from position i's hidden state (CE). "
            "Defaults to 0 for kl/subset_kl and 1 for ce."
        ),
    )

    # Training
    train_parser.add_argument("--max_seq_len", type=int, default=1024)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--num_steps", type=int, default=1000)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--warmup_steps", type=int, default=0)
    train_parser.add_argument("--lr_schedule", choices=["constant", "linear", "cosine"], default="constant")
    train_parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    train_parser.add_argument("--tokens_per_step", type=int, default=262144)

    # AMP
    train_parser.add_argument("--amp", action="store_true", default=True)
    train_parser.add_argument("--no_amp", action="store_false", dest="amp")
    train_parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")

    # DDP
    train_parser.add_argument("--ddp", action="store_true", default=True)
    train_parser.add_argument("--no_ddp", action="store_false", dest="ddp")

    # Model parallelism (for large teacher models)
    train_parser.add_argument("--model_parallel", action="store_true", default=False,
                              help="Shard teacher model across GPUs via device_map")
    train_parser.add_argument("--model_dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                              help="Precision for model loading (default: bf16)")
    train_parser.add_argument("--per_gpu_max_memory", type=str, default=None,
                              help="Per-GPU memory limit, e.g. '38GiB'")
    train_parser.add_argument("--cpu_offload_gb", type=int, default=0,
                              help="GB of CPU RAM for weight overflow (0=disabled)")
    train_parser.add_argument("--attn_implementation", type=str, default=None,
                              help="Attention impl: flash_attention_2, sdpa, eager")

    # Multi-node model parallelism (PyTorch FSDP)
    train_parser.add_argument("--multinode_model_parallel", action="store_true", default=False,
                              help="Shard teacher across nodes via PyTorch FSDP. "
                                   "Use with torchrun --nnodes=N --nproc_per_node=4.")
    train_parser.add_argument("--fsdp_offload_to_cpu", action="store_true", default=False,
                              help="Offload FSDP param shards to CPU between forward calls")

    # Logging
    train_parser.add_argument("--log_every", type=int, default=10)
    train_parser.add_argument("--save_every", type=int, default=500)
    train_parser.add_argument(
        "--save_initial_checkpoint",
        action="store_true",
        default=False,
        help="Save lens_step_0.pt after optional initialization and before optimizer steps.",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Explicit output directory for checkpoints and logs. "
            "When omitted, a canonical path is derived from the run's attributes and "
            "placed under --base_output_dir: "
            "{model}/{lens}/{loss}/{sites}/{init}/{tag}"
        ),
    )
    train_parser.add_argument(
        "--base_output_dir",
        type=str,
        default="./checkpoints",
        help="Root directory for auto-derived canonical output paths (default: ./checkpoints).",
    )
    train_parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help=(
            "Short label appended to the canonical output path (last component). "
            "Useful for distinguishing ablations with identical structural config. "
            "Defaults to seed{seed} when omitted."
        ),
    )
    train_parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Checkpoint file to resume from, or a checkpoint directory containing lens_step_*.pt",
    )
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
