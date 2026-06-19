#!/usr/bin/env python3
"""Summarize direct 70B feasibility runs."""

import argparse
import csv
import re
from pathlib import Path


RUN_LABELS = {
    "fullrank_fullkl": "Full-rank + Full KL",
    "lora_r64_fullkl": "LoRA (r = 64) + Full KL",
    "fullrank_subsetkl_k256": "Full-rank + Subset KL (k = 256)",
    "lora_r64_subsetkl_k256": "LoRA (r = 64) + Subset KL (k = 256)",
}


def peak_memory_gb(run_dir):
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


def log_text(log_root, run_name):
    candidates = sorted(log_root.glob(f"{run_name}*.log"))
    if not candidates:
        return None
    return candidates[-1].read_text(errors="replace")


def peak_rss_gb(log_root, run_name):
    text = log_text(log_root, run_name)
    if text is None:
        return None
    matches = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if matches:
        return max(int(value) for value in matches) / (1024**2)
    matches = re.findall(r"\[rss\]\s+max_rss_gb=([0-9.]+)", text)
    if matches:
        return max(float(value) for value in matches)
    return None


def log_status(log_root, run_name):
    text = log_text(log_root, run_name)
    if text is None:
        return "no log"
    low = text.lower()
    if "cuda out of memory" in low or "outofmemoryerror" in low:
        return "OOM"
    if "traceback" in low or "error" in low or "failed" in low:
        return "failed"
    if "[timing] end_utc=" in text:
        return "completed"
    return "started"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-out",
        type=Path,
        default=Path("src/checkpoints/llama3_70b_feasibility_direct_debug"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/llama3_70b_feasibility_direct_debug"),
    )
    parser.add_argument("--single-node-gb", type=float, default=160.0)
    args = parser.parse_args()

    print("Configuration,Status,Peak GPU GB,Peak RSS GB,Feasible on 160GB")
    for run_name, label in RUN_LABELS.items():
        peak = peak_memory_gb(args.base_out / run_name)
        rss = peak_rss_gb(args.log_root, run_name)
        status = log_status(args.log_root, run_name)
        measured = rss if rss is not None else peak
        if measured is None:
            print(f"{label},{status},missing,missing,no")
            continue
        feasible = "yes" if measured <= args.single_node_gb and status == "completed" else "no"
        peak_s = "missing" if peak is None else f"{peak:.2f}"
        rss_s = "missing" if rss is None else f"{rss:.2f}"
        print(f"{label},{status},{peak_s},{rss_s},{feasible}")

    print()
    print("LaTeX rows:")
    for run_name, label in RUN_LABELS.items():
        peak = peak_memory_gb(args.base_out / run_name)
        rss = peak_rss_gb(args.log_root, run_name)
        status = log_status(args.log_root, run_name)
        measured = rss if rss is not None else peak
        if measured is None:
            mem = r"\mathrm{OOM/fail}"
            feasible = r"\xmark"
        else:
            mem = rf"{measured:.1f}\,GB"
            feasible = r"\cmark" if measured <= args.single_node_gb and status == "completed" else r"\xmark"
        print(f"{label:<45} & ${mem}$ & {feasible} \\\\")


if __name__ == "__main__":
    main()
