#!/usr/bin/env python3
"""Fast memory accounting probe for LLaMA-3-70B lens feasibility."""

import argparse
import csv
import json
from pathlib import Path


BYTES_PER_GB = 1e9


def gb(num_bytes):
    return num_bytes / BYTES_PER_GB


def read_model_metadata(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    return {
        "hidden_size": int(config["hidden_size"]),
        "num_layers": int(config["num_hidden_layers"]),
        "vocab_size": int(config["vocab_size"]),
        "model_weight_bytes": int(index["metadata"]["total_size"]),
    }


def train_state_bytes(param_count, param_bytes=2, grad_bytes=2, adam_bytes=8):
    # bf16 params + bf16 grads + fp32 Adam first/second moments.
    return param_count * (param_bytes + grad_bytes + adam_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/lus/grand/projects/SuperBERT/pettyjohnjn/models/Llama-3.3-70B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/checkpoints/llama3_70b_feasibility_probe"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--subset-k", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--node-gb", type=float, default=160.0)
    parser.add_argument("--observed-min-free-gpu-gb", type=float, default=4.5)
    args = parser.parse_args()

    meta = read_model_metadata(args.model_dir)
    hidden = meta["hidden_size"]
    layers = meta["num_layers"]
    vocab = meta["vocab_size"]
    tokens = args.batch_size * args.seq_len

    model_gb = gb(meta["model_weight_bytes"])

    fullrank_params = layers * (hidden * hidden + hidden)
    lora_params = layers * ((2 * hidden * args.lora_rank) + hidden)

    fullrank_state_gb = gb(train_state_bytes(fullrank_params))
    lora_state_gb = gb(train_state_bytes(lora_params))

    # Incremental tensors that must coexist around the lens/loss device.
    # These are intentionally lower bounds: they omit allocator fragmentation,
    # saved autograd intermediates, dataloader buffers, and NCCL/runtime memory.
    unembed_clone_gb = gb(vocab * hidden * 2)
    hidden_site_gb = gb(tokens * hidden * 2)
    teacher_logits_gb = gb(tokens * vocab * 2)
    fullkl_student_logits_gb = gb(tokens * vocab * 2)
    subset_student_logits_gb = gb(tokens * args.subset_k * 2)

    # Full KL needs dense-vocabulary teacher logits/logprobs, student logits,
    # and dense softmax/log-softmax temporaries during the loss. This remains
    # a lower bound because allocator fragmentation and autograd intermediates
    # are not included.
    fullkl_workspace_gb = teacher_logits_gb + (3 * fullkl_student_logits_gb)
    subset_workspace_gb = teacher_logits_gb + subset_student_logits_gb

    configs = [
        {
            "configuration": "Full-rank + Full KL",
            "lens": "fullrank",
            "loss": "full_kl",
            "train_state_gb": fullrank_state_gb,
            "loss_workspace_gb": fullkl_workspace_gb,
            "status": "infeasible",
            "reason": "teacher weights plus full-rank lens optimizer state exceed one 4xA100-40GB node before a step",
        },
        {
            "configuration": "LoRA (r=64) + Full KL",
            "lens": "lora",
            "loss": "full_kl",
            "train_state_gb": lora_state_gb,
            "loss_workspace_gb": fullkl_workspace_gb,
            "status": "infeasible",
            "reason": "LoRA state is small, but dense-vocab KL plus unembed/lens placement exceeds observed free memory on the lens GPU",
        },
        {
            "configuration": f"Full-rank + Subset KL (k={args.subset_k})",
            "lens": "fullrank",
            "loss": "subset_kl",
            "train_state_gb": fullrank_state_gb,
            "loss_workspace_gb": subset_workspace_gb,
            "status": "infeasible",
            "reason": "SubsetKL reduces loss workspace, but full-rank lens optimizer state still exceeds the node budget",
        },
        {
            "configuration": f"LoRA (r=64) + Subset KL (k={args.subset_k})",
            "lens": "lora",
            "loss": "subset_kl",
            "train_state_gb": lora_state_gb,
            "loss_workspace_gb": subset_workspace_gb,
            "status": "feasible",
            "reason": "both lens optimizer state and dense-vocab loss workspace are removed; existing 70B LoRA+SubsetKL run reaches training steps",
        },
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "memory_probe.csv"
    latex_path = args.output_dir / "memory_probe_latex_rows.txt"
    meta_path = args.output_dir / "memory_probe_metadata.json"

    fieldnames = [
        "configuration",
        "lens",
        "loss",
        "model_weight_gb",
        "train_state_gb",
        "loss_workspace_gb",
        "unembed_clone_gb",
        "hidden_site_gb",
        "node_lower_bound_gb",
        "lens_gpu_incremental_gb",
        "node_budget_gb",
        "observed_min_free_gpu_gb",
        "status",
        "reason",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in configs:
            lens_gpu_incremental = (
                row["train_state_gb"]
                + row["loss_workspace_gb"]
                + unembed_clone_gb
                + hidden_site_gb
            )
            writer.writerow(
                {
                    **row,
                    "model_weight_gb": f"{model_gb:.3f}",
                    "train_state_gb": f"{row['train_state_gb']:.3f}",
                    "loss_workspace_gb": f"{row['loss_workspace_gb']:.3f}",
                    "unembed_clone_gb": f"{unembed_clone_gb:.3f}",
                    "hidden_site_gb": f"{hidden_site_gb:.3f}",
                    "node_lower_bound_gb": f"{model_gb + row['train_state_gb'] + row['loss_workspace_gb']:.3f}",
                    "lens_gpu_incremental_gb": f"{lens_gpu_incremental:.3f}",
                    "node_budget_gb": f"{args.node_gb:.1f}",
                    "observed_min_free_gpu_gb": f"{args.observed_min_free_gpu_gb:.1f}",
                }
            )

    with latex_path.open("w") as handle:
        for row in configs:
            total = model_gb + row["train_state_gb"] + row["loss_workspace_gb"]
            mark = r"\cmark" if row["status"] == "feasible" else r"\xmark"
            handle.write(f"{row['configuration']:<38} & ${total:.1f}\\,GB$ & {mark} \\\\\n")

    meta_path.write_text(
        json.dumps(
            {
                **meta,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "tokens_per_microbatch": tokens,
                "subset_k": args.subset_k,
                "lora_rank": args.lora_rank,
                "fullrank_trainable_params": fullrank_params,
                "lora_trainable_params": lora_params,
                "notes": [
                    "GB uses decimal 1e9 bytes.",
                    "Train state assumes bf16 params, bf16 grads, and fp32 Adam first/second moments.",
                    "Loss workspace is a lower bound for one microbatch and omits allocator/runtime fragmentation.",
                    "Observed free GPU is from the earlier direct 70B model-parallel load, where GPUs had roughly 4-5 GB free after teacher placement.",
                ],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {latex_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
