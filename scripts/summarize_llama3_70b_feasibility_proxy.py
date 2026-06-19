#!/usr/bin/env python3
"""Summarize proxy memory measurements for the 70B feasibility table."""

import argparse
import csv
from pathlib import Path


RUN_LABELS = {
    "fullrank_fullkl": "Full-rank + Full KL",
    "lora_r64_fullkl": "LoRA (r = 64) + Full KL",
    "fullrank_subsetkl_k256": "Full-rank + Subset KL (k = 256)",
    "lora_r64_subsetkl_k256": "LoRA (r = 64) + Subset KL (k = 256)",
}


def peak_memory_gb(run_dir: Path):
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return None
    with metrics.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row["peak_memory_gb"])
        for row in rows
        if row.get("peak_memory_gb") not in (None, "")
    ]
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-out",
        type=Path,
        default=Path("src/checkpoints/llama3_70b_feasibility_proxy"),
    )
    parser.add_argument("--proxy-params-b", type=float, default=8.0)
    parser.add_argument("--target-params-b", type=float, default=70.0)
    parser.add_argument("--single-node-gb", type=float, default=160.0)
    args = parser.parse_args()

    scale = args.target_params_b / args.proxy_params_b

    print("Configuration,Proxy peak GB,70B scaled GB,Feasible on 160GB")
    for run_name, label in RUN_LABELS.items():
        peak = peak_memory_gb(args.base_out / run_name)
        if peak is None:
            print(f"{label},missing,missing,unknown")
            continue
        scaled = peak * scale
        feasible = "yes" if scaled <= args.single_node_gb else "no"
        print(f"{label},{peak:.2f},{scaled:.1f},{feasible}")

    print()
    print("LaTeX rows:")
    for run_name, label in RUN_LABELS.items():
        peak = peak_memory_gb(args.base_out / run_name)
        if peak is None:
            mem = r"\ph"
            feasible = r"\ph"
        else:
            scaled = peak * scale
            mem = rf"{scaled:.0f}\,GB"
            feasible = r"\cmark" if scaled <= args.single_node_gb else r"\xmark"
        print(f"{label:<45} & ${mem}$ & {feasible} \\\\")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
