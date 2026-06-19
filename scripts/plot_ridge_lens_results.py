#!/usr/bin/env python3
"""Plot GPT-2 LoRA ridge initialization eval results."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "evaluation"
OUT = EVAL / "ridge_lens_summary.png"


def mean_kl(path: Path, lens: str = "lora") -> float:
    with path.open() as f:
        data = json.load(f)
    values = [float(v) for v in data[lens]["kl"].values()]
    return sum(values) / len(values)


def collect_curve(root: Path, pattern: str, label: str) -> tuple[str, list[int], list[float]]:
    points: list[tuple[int, float]] = []
    regex = re.compile(pattern)
    for path in root.glob("*/aggregate_metrics.json"):
        match = regex.search(path.parent.name)
        if match is None:
            continue
        points.append((int(match.group(1)), mean_kl(path)))
    points.sort()
    return label, [p[0] for p in points], [p[1] for p in points]


def main() -> None:
    main_root = EVAL / "gpt2_lora_ridge_only_r64_debug"
    metric_root = EVAL / "gpt2_lora_ridge_metric_ablation_r64_debug"
    lamnorm_root = EVAL / "gpt2_lora_ridge_lambda_norm_ablation_r64_debug"

    curves = [
        collect_curve(
            main_root,
            r"lora_fullkl_r64_default_lora__lens_step_(\d+)$",
            "Default LoRA init",
        ),
        collect_curve(
            main_root,
            r"lora_fullkl_r64_ridge_svd__lens_step_(\d+)$",
            "Ridge/SVD init + KL train",
        ),
        collect_curve(
            metric_root,
            r"lora_fullkl_r64_ridge_residual_metric__lens_step_(\d+)$",
            "Ridge lens + short KL polish",
        ),
        collect_curve(
            lamnorm_root,
            r"lora_fullkl_r64_ridge_lam1em3_normper_dim_std__lens_step_(\d+)$",
            "Best lambda/norm polish",
        ),
    ]

    ridge_only_path = (
        main_root
        / "lora_fullkl_r64_ridge_svd_step0__lens_step_0"
        / "aggregate_metrics.json"
    )
    ridge_only_kl = mean_kl(ridge_only_path)
    logit_kl = mean_kl(ridge_only_path, lens="logit")

    # Add the separately evaluated ridge-only point to the long ridge-training curve.
    merged_curves = []
    for label, xs, ys in curves:
        if label == "Ridge/SVD init + KL train":
            xs = [0] + xs
            ys = [ridge_only_kl] + ys
        merged_curves.append((label, xs, ys))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    styles = {
        "Default LoRA init": dict(color="#555555", marker="o", linewidth=2.2),
        "Ridge/SVD init + KL train": dict(color="#1f77b4", marker="o", linewidth=2.6),
        "Ridge lens + short KL polish": dict(color="#2ca02c", marker="s", linewidth=2.2),
        "Best lambda/norm polish": dict(color="#ff7f0e", marker="D", linewidth=2.2),
    }

    for label, xs, ys in merged_curves:
        ax.plot(xs, ys, label=label, **styles[label])

    ax.axhline(
        logit_kl,
        color="#999999",
        linestyle=":",
        linewidth=2,
        label=f"Logit Lens ({logit_kl:.2f})",
    )

    ax.scatter([0], [ridge_only_kl], color="#1f77b4", s=65, zorder=5)
    ax.annotate(
        f"Ridge-only\n{ridge_only_kl:.2f}",
        xy=(0, ridge_only_kl),
        xytext=(14, 16),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1),
        color="#1f77b4",
    )

    default_250 = dict(zip(merged_curves[0][1], merged_curves[0][2]))[250]
    ridge_100 = dict(zip(merged_curves[1][1], merged_curves[1][2]))[100]
    ax.annotate(
        "Ridge step 100\nmatches default step 250",
        xy=(100, ridge_100),
        xytext=(118, 0.92),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1),
        color="#1f77b4",
    )
    ax.annotate(
        f"default 250: {default_250:.2f}",
        xy=(250, default_250),
        xytext=(-92, 26),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1),
        color="#555555",
    )

    ax.set_title("GPT-2 r=64 LoRA Lens: Ridge Init Turns Training Into Short KL Polish")
    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Validation mean KL across residual layers")
    ax.set_xlim(-5, 260)
    ax.set_ylim(0.55, 3.55)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.01,
        0.02,
        "Eval: 16.4k Pile test tokens. Lower is better.",
        transform=ax.transAxes,
        color="#555555",
        fontsize=9,
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
