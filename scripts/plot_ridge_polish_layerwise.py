#!/usr/bin/env python3
"""Plot layerwise KL for ridge-initialized LoRA as full-KL polish proceeds."""

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
OUT_PNG = OUT_DIR / "gpt2_ridge_polish_layerwise.png"
OUT_CSV = OUT_DIR / "gpt2_ridge_polish_layerwise.csv"


@dataclass(frozen=True)
class Series:
    label: str
    path: Path
    lens_key: str
    color: str
    linestyle: str = "-"
    linewidth: float = 2.0


RIDGE_ROOT = EVAL / "gpt2_lora_ridge_only_r64_debug"

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
        linewidth=2.5,
    ),
    Series(
        label="Ridge init, 0 steps",
        path=RIDGE_ROOT
        / "lora_fullkl_r64_ridge_svd_step0__lens_step_0"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#9ecae1",
    ),
    Series(
        label="Ridge + full KL, 50 steps",
        path=RIDGE_ROOT
        / "lora_fullkl_r64_ridge_svd__lens_step_50"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#6baed6",
    ),
    Series(
        label="Ridge + full KL, 100 steps",
        path=RIDGE_ROOT
        / "lora_fullkl_r64_ridge_svd__lens_step_100"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#4292c6",
    ),
    Series(
        label="Ridge + full KL, 150 steps",
        path=RIDGE_ROOT
        / "lora_fullkl_r64_ridge_svd__lens_step_150"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#2171b5",
    ),
    Series(
        label="Ridge + full KL, 200 steps",
        path=RIDGE_ROOT
        / "lora_fullkl_r64_ridge_svd__lens_step_200"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#08519c",
    ),
    Series(
        label="Ridge + full KL, 250 steps",
        path=RIDGE_ROOT
        / "lora_fullkl_r64_ridge_svd__lens_step_250"
        / "aggregate_metrics.json",
        lens_key="lora",
        color="#08306b",
        linewidth=2.5,
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
    return sorted(
        (
            (layer_index(layer_key), float(value))
            for layer_key, value in data[series.lens_key]["kl"].items()
        ),
        key=lambda item: item[0],
    )


def plot_panel(ax, curves, *, exclude_layer0: bool) -> None:
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    for marker, (series, points) in zip(markers, curves):
        if exclude_layer0:
            points = [(layer, kl) for layer, kl in points if layer != 0]
        layers = [layer for layer, _ in points]
        kls = [kl for _, kl in points]
        ax.plot(
            layers,
            kls,
            label=f"{series.label} (mean={sum(kls) / len(kls):.3f})",
            color=series.color,
            linestyle=series.linestyle,
            linewidth=series.linewidth,
            marker=marker,
            markersize=5,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Residual layer")
    ax.set_ylabel("Validation KL")
    ax.set_xticks([layer for layer, _ in curves[0][1] if not exclude_layer0 or layer != 0])
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    curves = [(series, load_layerwise_kl(series)) for series in SERIES]

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.0), sharey=False)

    plot_panel(axes[0], curves, exclude_layer0=False)
    axes[0].set_title("All layers")
    plot_panel(axes[1], curves, exclude_layer0=True)
    axes[1].set_title("Layers 1-11 only")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("GPT-2 r=64 LoRA Ridge Init: Layerwise KL During Full-KL Polish")
    fig.text(
        0.01,
        0.02,
        "Log y-axis. The right panel removes layer 0 to show whether remaining layers are close enough to the full tuned baseline.",
        color="#555555",
        fontsize=9,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 0.80, 0.96))
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
