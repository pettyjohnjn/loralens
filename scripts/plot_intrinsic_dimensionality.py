#!/usr/bin/env python3
"""Plot appendix SVD diagnostics for full-rank tuned lenses."""

import argparse
import csv
from pathlib import Path


DEFAULT_ANALYSIS_DIR = Path(
    "/eagle/projects/ModCon/jpettyjohn/loralens-refactored/analysis/"
    "intrinsic_dimensionality"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create intrinsic-dimensionality appendix plots."
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--output-stem",
        default="rank64_frobenius_energy_trajectory",
        help="Output filename stem. PDF and PNG are both written.",
    )
    return parser.parse_args()


def load_rank64_summary(path):
    by_step = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["checkpoint_step"])
            by_step.setdefault(
                step,
                {
                    "tokens": int(float(row["total_tokens"])),
                    "values": [],
                },
            )
            by_step[step]["values"].append(
                float(row["frobenius_energy_fraction_at_rank_64"])
            )

    rows = []
    for step in sorted(by_step):
        values = by_step[step]["values"]
        rows.append(
            {
                "step": step,
                "tokens": by_step[step]["tokens"],
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def plot_panel(ax, rows, title, color):
    xs = [row["step"] for row in rows]
    means = [100.0 * row["mean"] for row in rows]
    lows = [100.0 * row["min"] for row in rows]
    highs = [100.0 * row["max"] for row in rows]

    ax.fill_between(xs, lows, highs, color=color, alpha=0.18, linewidth=0)
    ax.plot(xs, means, color=color, linewidth=2.4, marker="o", markersize=4.5)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Rank-64 Frobenius energy (%)")
    ax.set_ylim(25, 80)
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.8)
    ax.grid(True, axis="x", color="#eeeeee", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpt2_rows = load_rank64_summary(args.analysis_dir / "gpt2" / "layer_summary.csv")
    llama_rows = load_rank64_summary(
        args.analysis_dir / "llama3_8b" / "layer_summary.csv"
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), constrained_layout=True)
    plot_panel(axes[0], gpt2_rows, "GPT-2 Small", "#2f6f9f")
    plot_panel(axes[1], llama_rows, "LLaMA-3-8B", "#9a4d2f")

    out_pdf = args.analysis_dir / f"{args.output_stem}.pdf"
    out_png = args.analysis_dir / f"{args.output_stem}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[plot] wrote {out_pdf}")
    print(f"[plot] wrote {out_png}")


if __name__ == "__main__":
    main()
