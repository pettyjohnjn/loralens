#!/usr/bin/env python3
"""
Write injection evaluation: compare three methods for injecting a target token
into a GPT-2 residual stream and measuring downstream prediction quality.

Methods:
  no_inject  — baseline: model predicts without any injection
  unembed    — inject W_U[target] directly (raw unembedding column)
  ortho      — inject via BidirLoRALens Woodbury inverse (ortho-trained)
  suffix     — inject via BidirLoRALens Woodbury inverse (suffix-trained)

For each resid_post site (L00-L11), injects at the last token position of each
test sequence and measures how well the model then predicts a random target token
at that position.

Usage (on a GPU node):
  python scripts/eval_write_injection.py \
    --ortho_ckpt  src/checkpoints/.../bidir-r64-ortho/.../lens_step_250.pt \
    --suffix_ckpt src/checkpoints/.../bidir-r64-suffix/.../lens_step_500.pt \
    --out         /eagle/.../evaluation/write_injection_eval.json \
    --n_examples  512
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# loralens imports (PYTHONPATH must include loralens-refactored/src)
from loralens.lenses import create_lens
from loralens.lenses.bidir_lora_lens import BidirLoRALens
from loralens.training import HFUnembed, get_model_config


# ── helpers ───────────────────────────────────────────────────────────────────

def load_site_ids(ckpt_path: Path) -> List[str]:
    csv_path = ckpt_path.parent / "activation_sites.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"activation_sites.csv not found next to {ckpt_path}")
    with csv_path.open() as f:
        return [row["site_id"] for row in csv.DictReader(f)]


def load_bidir_lens(ckpt_path: Path, model, device) -> BidirLoRALens:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]
    site_ids = load_site_ids(ckpt_path)
    model_cfg = get_model_config(model)
    lens = create_lens(
        "bidir_lora",
        layer_ids=site_ids,
        hidden_size=model_cfg["hidden_size"],
        unembed=HFUnembed(model),
        r=int(cfg["lora_rank"]),
        alpha=float(cfg.get("lora_alpha", cfg["lora_rank"])),
    )
    lens.load_checkpoint_state_dict(ckpt["lens_state_dict"])
    lens.to(device)
    lens.eval()
    return lens


def run_with_injection(
    model,
    input_ids: torch.Tensor,          # [1, T]
    layer_idx: int,
    delta: Optional[torch.Tensor],    # [1, d] or None
    inject_pos: int,
) -> torch.Tensor:
    """Forward pass with optional additive delta injected at inject_pos."""
    if delta is None:
        with torch.no_grad():
            return model(input_ids=input_ids).logits[0, inject_pos]

    done = [False]
    def hook(_module, _inp, output):
        if done[0]:
            return output
        done[0] = True
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, inject_pos, :] = h[:, inject_pos, :] + delta.to(h.dtype)
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0, inject_pos]
    finally:
        handle.remove()
    return logits


def rank_of(logits: torch.Tensor, target_id: int) -> int:
    """1-based rank of target_id in descending logits."""
    return int((logits > logits[target_id]).sum().item()) + 1


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    ortho_lens: Optional[BidirLoRALens],
    suffix_lens: Optional[BidirLoRALens],
    n_examples: int,
    device: torch.device,
    seed: int = 42,
) -> Dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Test prompts: short sentences where we inject at the last token position
    PROMPTS = [
        "The quick brown fox jumps over the lazy",
        "In the beginning God created the heavens and the",
        "To be or not to be that is the",
        "Four score and seven years ago our fathers brought forth on this continent a new",
        "It was the best of times it was the worst of",
        "All animals are equal but some animals are more equal than",
        "Call me Ishmael Some years ago never mind how long precisely having little money",
        "It is a truth universally acknowledged that a single man in possession of a good fortune",
        "The sky above the port was the color of television tuned to a dead",
        "In a hole in the ground there lived a",
    ] * max(1, n_examples // 10)

    vocab_size = model.config.vocab_size
    resid_sites = [f"L{i:02d}.resid_post" for i in range(model.config.num_hidden_layers)]

    # Per-layer accumulators: method → list of (rank, logprob)
    results = {layer: {m: [] for m in ("no_inject", "unembed", "ortho", "suffix")}
               for layer in resid_sites}

    for ex_idx in range(n_examples):
        prompt = rng.choice(PROMPTS)
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]  # [1, T]
        T = input_ids.shape[1]
        if T < 2:
            continue
        inject_pos = T - 1  # inject at last token, measure prediction there

        # Random target token (avoids conflating "easy context" with injection quality)
        target_id = rng.randint(0, vocab_size - 1)
        target_onehot = F.one_hot(torch.tensor([target_id], device=device),
                                   num_classes=vocab_size).float()

        for layer_id in resid_sites:
            layer_idx = int(layer_id[1:3])

            # ── no injection ──
            logits = run_with_injection(model, input_ids, layer_idx, None, inject_pos)
            r = rank_of(logits, target_id)
            lp = float(F.log_softmax(logits.float(), dim=-1)[target_id])
            results[layer_id]["no_inject"].append((r, lp))

            # ── unembed baseline: inject W_U[target_id] directly ──
            W_U = (ortho_lens or suffix_lens)._get_W_U()
            delta_unembed = W_U[target_id].unsqueeze(0).to(device)  # [1, d]
            logits = run_with_injection(model, input_ids, layer_idx, delta_unembed, inject_pos)
            r = rank_of(logits, target_id)
            lp = float(F.log_softmax(logits.float(), dim=-1)[target_id])
            results[layer_id]["unembed"].append((r, lp))

            # ── ortho lens ──
            if ortho_lens is not None:
                with torch.no_grad():
                    delta_ortho = ortho_lens.compute_write_injection(target_onehot, layer_id)
                logits = run_with_injection(model, input_ids, layer_idx, delta_ortho, inject_pos)
                r = rank_of(logits, target_id)
                lp = float(F.log_softmax(logits.float(), dim=-1)[target_id])
                results[layer_id]["ortho"].append((r, lp))

            # ── suffix lens ──
            if suffix_lens is not None:
                with torch.no_grad():
                    delta_suffix = suffix_lens.compute_write_injection(target_onehot, layer_id)
                logits = run_with_injection(model, input_ids, layer_idx, delta_suffix, inject_pos)
                r = rank_of(logits, target_id)
                lp = float(F.log_softmax(logits.float(), dim=-1)[target_id])
                results[layer_id]["suffix"].append((r, lp))

        if (ex_idx + 1) % 50 == 0:
            print(f"[eval] {ex_idx + 1}/{n_examples} examples done", flush=True)

    # Aggregate
    agg = {}
    for layer_id, methods in results.items():
        agg[layer_id] = {}
        for method, pairs in methods.items():
            if not pairs:
                continue
            ranks = [r for r, _ in pairs]
            lps   = [lp for _, lp in pairs]
            agg[layer_id][method] = {
                "top1":   sum(r == 1 for r in ranks) / len(ranks),
                "top5":   sum(r <= 5 for r in ranks) / len(ranks),
                "top10":  sum(r <= 10 for r in ranks) / len(ranks),
                "top100": sum(r <= 100 for r in ranks) / len(ranks),
                "mrr":    sum(1.0 / r for r in ranks) / len(ranks),
                "mean_logprob": sum(lps) / len(lps),
                "n": len(ranks),
            }
    return agg


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ortho_ckpt",  type=Path, default=None)
    parser.add_argument("--suffix_ckpt", type=Path, default=None)
    parser.add_argument("--out",         type=Path, required=True)
    parser.add_argument("--n_examples",  type=int, default=512)
    parser.add_argument("--model_name",  default="gpt2")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    if args.ortho_ckpt is None and args.suffix_ckpt is None:
        raise ValueError("At least one of --ortho_ckpt or --suffix_ckpt must be given")

    device = torch.device(args.device)

    print(f"[eval] loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ortho_lens  = load_bidir_lens(args.ortho_ckpt,  model, device) if args.ortho_ckpt  else None
    suffix_lens = load_bidir_lens(args.suffix_ckpt, model, device) if args.suffix_ckpt else None

    print(f"[eval] running {args.n_examples} examples across all resid_post sites")
    agg = evaluate(model, tokenizer, ortho_lens, suffix_lens, args.n_examples, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(agg, f, indent=2)

    # Print summary table
    methods = ["no_inject", "unembed", "ortho", "suffix"]
    present = [m for m in methods if any(m in agg[l] for l in agg)]
    print(f"\n{'Layer':<20} " + " ".join(f"{m:>12}" for m in present) + "  (MRR)")
    print("-" * (20 + 14 * len(present)))
    for layer in sorted(agg):
        row = f"{layer:<20}"
        for m in present:
            mrr = agg[layer].get(m, {}).get("mrr", float("nan"))
            row += f" {mrr:>12.4f}"
        print(row)

    print(f"\n[done] results saved to {args.out}")


if __name__ == "__main__":
    main()
