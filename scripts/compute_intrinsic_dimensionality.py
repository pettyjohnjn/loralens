#!/usr/bin/env python3
"""Compute singular-value summaries for full-rank tuned lens translators.

The LoRA Lens paper uses this to measure the intrinsic dimensionality of the
learned residual translator matrices in tuned KL baselines. Checkpoints are
expected to contain loralens trainer dictionaries with a ``lens_state_dict``
entry, but the extractor also accepts a raw state dict for convenience.
"""

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


DEFAULT_GPT2_DIR = None
DEFAULT_LLAMA3_8B_DIR = None
DEFAULT_OUTPUT_DIR = None


class ModelSpec:
    def __init__(self, name, checkpoint_dir, checkpoint_step):
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_step = checkpoint_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SVD-based intrinsic-dimensionality data for tuned lenses."
    )
    parser.add_argument("--gpt2-dir", type=Path, default=DEFAULT_GPT2_DIR)
    parser.add_argument("--llama3-8b-dir", type=Path, default=DEFAULT_LLAMA3_8B_DIR)
    parser.add_argument(
        "--gpt2-step",
        default="latest",
        help="Checkpoint step: latest, all, an integer, or a .pt path.",
    )
    parser.add_argument(
        "--llama3-8b-step",
        default="latest",
        help="Checkpoint step: latest, all, an integer, or a .pt path.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for each matrix SVD. Checkpoints are still loaded on CPU.",
    )
    parser.add_argument(
        "--rel-thresholds",
        default="1e-3",
        help="Comma-separated effective-rank thresholds relative to top singular value.",
    )
    parser.add_argument(
        "--variance-targets",
        default="0.9,0.95,0.99,0.999",
        help="Comma-separated cumulative squared-singular-value targets.",
    )
    parser.add_argument(
        "--rank-probes",
        default="1,4,8,16,32,64,96,128,256,384",
        help="Comma-separated ranks for cumulative-variance probe columns.",
    )
    parser.add_argument(
        "--include-bias-norm",
        action="store_true",
        help="Also summarize translator bias norms when bias tensors are present.",
    )
    return parser.parse_args()


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def strip_prefix(value, prefix):
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def checkpoint_step(path):
    match = re.fullmatch(r"lens_step_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Not a lens checkpoint name: {path}")
    return int(match.group(1))


def resolve_checkpoints(checkpoint_dir, step):
    if step == "all":
        checkpoints = sorted(
            checkpoint_dir.glob("lens_step_*.pt"),
            key=checkpoint_step,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No lens_step_*.pt checkpoints in {checkpoint_dir}")
        return checkpoints
    if step == "latest":
        checkpoints = sorted(
            checkpoint_dir.glob("lens_step_*.pt"),
            key=checkpoint_step,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No lens_step_*.pt checkpoints in {checkpoint_dir}")
        return [checkpoints[-1]]
    if step.endswith(".pt"):
        path = Path(step)
        return [path if path.is_absolute() else checkpoint_dir / path]
    if not step.isdigit():
        raise ValueError(f"Step must be 'latest', 'all', an integer, or a .pt path: {step}")
    return [checkpoint_dir / f"lens_step_{step}.pt"]


def find_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("lens_state_dict", "state_dict", "model_state_dict"):
            value = obj.get(key)
            if is_tensor_state_dict(value):
                return value
        if is_tensor_state_dict(obj):
            return obj
    raise ValueError(
        "Checkpoint does not contain a recognizable tensor state dict. "
        "Expected 'lens_state_dict' or a raw state dict."
    )


def is_tensor_state_dict(value):
    return isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values())


def layer_from_translator_key(key):
    key = strip_prefix(key, "module.")
    match = re.search(r"(?:^|\.)translators\.([^.]+)\.weight$", key)
    if match is not None:
        return match.group(1)
    return key.rsplit(".", 1)[0]


def layer_sort_key(layer):
    numbers = re.findall(r"\d+", layer)
    return (int(numbers[-1]) if numbers else 10**9, layer)


def extract_translators(
    state_dict,
):
    weights = []  # type: List[Tuple[str, str, torch.Tensor]]
    biases = {}  # type: Dict[str, torch.Tensor]

    for key, tensor in state_dict.items():
        clean_key = strip_prefix(key, "module.")
        if re.search(r"(?:^|\.)translators\.[^.]+\.weight$", clean_key):
            if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
                raise ValueError(f"Translator weight is not square: {key} {tuple(tensor.shape)}")
            weights.append((layer_from_translator_key(clean_key), clean_key, tensor))
        elif re.search(r"(?:^|\.)translators\.[^.]+\.bias$", clean_key):
            layer = clean_key.rsplit(".", 1)[0].split(".")[-1]
            biases[layer] = tensor

    weights.sort(key=lambda item: layer_sort_key(item[0]))
    if not weights:
        matching = sorted(k for k in state_dict if "translator" in k.lower())[:30]
        raise ValueError(
            "No tuned-lens translator weights found. "
            f"Example translator-like keys: {matching}"
        )
    return weights, biases


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return torch.device(requested)


def safe_ratio(numerator, denominator):
    if denominator <= 0.0 or math.isnan(denominator):
        return torch.zeros_like(numerator)
    return numerator / denominator


def first_rank_reaching(cumulative, target):
    hits = torch.nonzero(cumulative >= target, as_tuple=False)
    if hits.numel() == 0:
        return int(cumulative.numel())
    return int(hits[0].item()) + 1


def checkpoint_metadata(checkpoint, checkpoint_path):
    step = checkpoint.get("step") if isinstance(checkpoint, dict) else None
    if step is None:
        try:
            step = checkpoint_step(checkpoint_path)
        except ValueError:
            step = ""
    total_tokens = checkpoint.get("total_tokens") if isinstance(checkpoint, dict) else None
    return step, total_tokens


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_model(
    spec,
    output_dir,
    device,
    rel_thresholds,
    variance_targets,
    rank_probes,
    include_bias_norm,
):
    start = time.perf_counter()
    model_dir = output_dir / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = resolve_checkpoints(spec.checkpoint_dir, spec.checkpoint_step)
    singular_rows = []  # type: List[Dict[str, Any]]
    summary_rows = []  # type: List[Dict[str, Any]]
    singular_tensors = {}  # type: Dict[str, Dict[str, torch.Tensor]]
    checkpoint_entries = []
    num_layers = None

    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        print(f"[load] {spec.name}: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_step_value, total_tokens = checkpoint_metadata(checkpoint, checkpoint_path)
        checkpoint_key = f"step_{checkpoint_step_value}"
        state_dict = find_state_dict(checkpoint)
        weights, biases = extract_translators(state_dict)
        num_layers = len(weights)
        singular_tensors[checkpoint_key] = {}
        checkpoint_entries.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": checkpoint_step_value,
                "total_tokens": total_tokens,
            }
        )

        for layer, key, weight_cpu in weights:
            layer_start = time.perf_counter()
            matrix = weight_cpu.detach().to(device=device, dtype=torch.float32)
            singular_values = torch.linalg.svdvals(matrix).detach().cpu()
            del matrix
            if device.type == "cuda":
                torch.cuda.empty_cache()

            singular_tensors[checkpoint_key][layer] = singular_values
            top = float(singular_values[0].item()) if singular_values.numel() else 0.0
            squares = singular_values.square()
            total_energy = float(squares.sum().item())
            cumulative_energy = torch.cumsum(squares, dim=0)
            cumulative_variance = safe_ratio(cumulative_energy, total_energy)
            normalized = safe_ratio(singular_values, top)

            summary = {
                "model": spec.name,
                "checkpoint_step": checkpoint_step_value,
                "total_tokens": total_tokens,
                "layer": layer,
                "state_key": key,
                "hidden_size": int(weight_cpu.shape[0]),
                "num_singular_values": int(singular_values.numel()),
                "top_singular_value": top,
                "frobenius_norm": math.sqrt(total_energy),
                "frobenius_energy": total_energy,
                "nuclear_norm": float(singular_values.sum().item()),
                "stable_rank": total_energy / (top * top) if top > 0.0 else 0.0,
                "svd_seconds": time.perf_counter() - layer_start,
            }

            for threshold in rel_thresholds:
                col = f"effective_rank_rel_{threshold:g}"
                summary[col] = int((normalized > threshold).sum().item())
            for target in variance_targets:
                col = f"rank_for_variance_{target:g}"
                summary[col] = first_rank_reaching(cumulative_variance, target)
            for rank in rank_probes:
                idx = min(rank, singular_values.numel()) - 1
                energy_fraction = (
                    float(cumulative_variance[idx].item()) if idx >= 0 else 0.0
                )
                residual_fraction = max(0.0, 1.0 - energy_fraction)
                summary[f"frobenius_energy_fraction_at_rank_{rank}"] = energy_fraction
                summary[f"frobenius_residual_fraction_at_rank_{rank}"] = residual_fraction
                summary[f"relative_frobenius_error_at_rank_{rank}"] = math.sqrt(
                    residual_fraction
                )
                summary[f"variance_at_rank_{rank}"] = energy_fraction
            if include_bias_norm:
                bias = biases.get(layer)
                summary["bias_norm"] = (
                    float(bias.detach().float().norm().item()) if bias is not None else ""
                )

            summary_rows.append(summary)

            for idx, value in enumerate(singular_values.tolist(), start=1):
                singular_rows.append(
                    {
                        "model": spec.name,
                        "checkpoint_step": checkpoint_step_value,
                        "total_tokens": total_tokens,
                        "layer": layer,
                        "rank_index": idx,
                        "singular_value": value,
                        "squared_singular_value": value * value,
                        "normalized_singular_value": float(normalized[idx - 1].item()),
                        "cumulative_frobenius_energy_fraction": float(
                            cumulative_variance[idx - 1].item()
                        ),
                        "cumulative_variance": float(cumulative_variance[idx - 1].item()),
                    }
                )

            rank64 = summary.get(
                "frobenius_energy_fraction_at_rank_64",
                summary.get("variance_at_rank_64", float("nan")),
            )
            print(
                f"[svd] {spec.name} step={checkpoint_step_value} {layer}: "
                f"d={weight_cpu.shape[0]} top={top:.6g} rank@1e-3="
                f"{int((normalized > 1e-3).sum().item())} "
                f"frob@64={rank64:.6f}",
                flush=True,
            )

        del checkpoint, state_dict

    summary_fieldnames = list(summary_rows[0].keys())
    write_csv(model_dir / "layer_summary.csv", summary_rows, summary_fieldnames)
    write_csv(
        model_dir / "singular_values.csv",
        singular_rows,
        [
            "model",
            "checkpoint_step",
            "total_tokens",
            "layer",
            "rank_index",
            "singular_value",
            "squared_singular_value",
            "normalized_singular_value",
            "cumulative_frobenius_energy_fraction",
            "cumulative_variance",
        ],
    )
    torch.save(
        {
            "model": spec.name,
            "checkpoint_paths": [entry["checkpoint_path"] for entry in checkpoint_entries],
            "singular_values": singular_tensors,
        },
        model_dir / "singular_values.pt",
    )

    elapsed = time.perf_counter() - start
    metadata = {
        "model": spec.name,
        "checkpoint_dir": str(spec.checkpoint_dir),
        "checkpoint_selector": spec.checkpoint_step,
        "checkpoints": checkpoint_entries,
        "num_checkpoints": len(checkpoint_entries),
        "num_layers": num_layers,
        "device": str(device),
        "matrix_kind": "tuned_lens_residual_translator_weight",
        "energy_definition": "frobenius_energy=sum(singular_value**2)",
        "rel_thresholds": rel_thresholds,
        "variance_targets": variance_targets,
        "rank_probes": rank_probes,
        "elapsed_seconds": elapsed,
    }
    with (model_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return metadata


def main() -> None:
    args = parse_args()
    rel_thresholds = parse_float_list(args.rel_thresholds)
    variance_targets = parse_float_list(args.variance_targets)
    rank_probes = parse_int_list(args.rank_probes)
    device = choose_device(args.device)

    specs = [
        ModelSpec("gpt2", args.gpt2_dir, args.gpt2_step),
        ModelSpec("llama3_8b", args.llama3_8b_dir, args.llama3_8b_step),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "created_by": Path(__file__).name,
        "output_dir": str(args.output_dir),
        "device": str(device),
        "models": [],
    }
    for spec in specs:
        run_metadata["models"].append(
            analyze_model(
                spec=spec,
                output_dir=args.output_dir,
                device=device,
                rel_thresholds=rel_thresholds,
                variance_targets=variance_targets,
                rank_probes=rank_probes,
                include_bias_norm=args.include_bias_norm,
            )
        )

    with (args.output_dir / "run_metadata.json").open("w") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[done] wrote intrinsic-dimensionality data to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
