#!/usr/bin/env python3
"""Plot layerwise validation KL for GPT-2 LoRA initialization comparisons."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "evaluation"
OUT_DIR = EVAL / "plots"
OUT_PNG = OUT_DIR / "gpt2_layerwise_init_comparison.png"
OUT_CSV = OUT_DIR / "gpt2_layerwise_init_comparison.csv"


@dataclass(frozen=True)
class Series:
    label: str
    path: Path
    lens_key: str
    color: str
    linestyle: str = "-"


SERIES = [
    Series(
        label="Full tuned lens baseline",
        path=EVAL
        / "gpt2_debug_quality_top10_500"
        / "tuned_kl_baseline"
        / "aggregate_metrics.json",
        lens_key="tuned",
        color="#111111",
        linestyle="--",
    ),
    Series(
        label="LoRA default init, full KL",
        path=EVAL
        / "gpt2_lora_init_fullkl_r64_debug"
        / "lora_fullkl_r64_default_lora__lens_step_250"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#666666",
    ),
    Series(
        label="LoRA ridge init, full KL",
        path=EVAL
        / "gpt2_lora_init_fullkl_r64_debug"
        / "lora_fullkl_r64_ridge_svd__lens_step_250"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#1f77b4",
    ),
    Series(
        label="LoRA default init, MC 256+256",
        path=EVAL
        / "gpt2_lora_init_mc_256x256_r64_debug"
        / "lora_mc256x256_r64_default_lora__lens_step_250"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#ff7f0e",
    ),
    Series(
        label="LoRA ridge init, MC 256+256",
        path=EVAL
        / "gpt2_lora_init_mc_256x256_r64_debug"
        / "lora_mc256x256_r64_ridge_svd__lens_step_250"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#2ca02c",
    ),
]


def layer_index(key: str) -> int:
    match = re.search(r"(\d+)$", key)
    if match is None:
        raise ValueError(f"Could not parse layer index from {key!r}")
    return int(match.group(1))


def load_layerwise_kl(series: Series) -> list[tuple[int, float]]:
    with series.path.open() as f:
        data = json.load(f)
    values = data[series.lens_key]["kl"]
    return sorted(
        ((layer_index(key), float(value)) for key, value in values.items()),
        key=lambda item: item[0],
    )


def main() -> None:
    curves = [(series, load_layerwise_kl(series)) for series in SERIES]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.2))

    markers = ["o", "s", "^", "D", "P"]
    for marker, (series, points) in zip(markers, curves):
        layers = [layer for layer, _ in points]
        kls = [kl for _, kl in points]
        ax.plot(
            layers,
            kls,
            label=f"{series.label} (mean={sum(kls) / len(kls):.3f})",
            color=series.color,
            linestyle=series.linestyle,
            linewidth=2.2,
            marker=marker,
            markersize=5.5,
        )

    ax.set_title("GPT-2 Lens Quality by Layer")
    ax.set_xlabel("Residual layer")
    ax.set_ylabel("Validation KL")
    ax.set_yscale("log")
    ax.set_xticks([layer for layer, _ in curves[0][1]])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.02,
        "Log y-axis. LoRA runs: r=64, step 250. MC estimator: 256 head + 256 tail. Lower is better.",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "layer", "validation_kl"])
        for series, points in curves:
            for layer, kl in points:
                writer.writerow([series.label, layer, f"{kl:.9f}"])

    print(OUT_PNG)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
