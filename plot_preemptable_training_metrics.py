#!/usr/bin/env python3

import colorsys
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(
    "/eagle/projects/ModCon/jpettyjohn/loralens-refactored/src/checkpoints/gpt2_preemptable_sweep_1000"
)
OUT = ROOT / "training_plots"
FAMILY_BASE = {
    "tuned_kl": "#111111",
    "lora_kl": "#7c3aed",
    "tuned_topk": "#1d4ed8",
    "lora_topk": "#60a5fa",
    "tuned_hajek": "#047857",
    "lora_hajek": "#34d399",
    "other": "#6b7280",
}
FAMILY_LABELS = {
    "tuned_kl": "Tuned Full KL",
    "lora_kl": "LoRA Full KL",
    "tuned_topk": "Tuned Top-k",
    "lora_topk": "LoRA Top-k",
    "tuned_hajek": "Tuned Hajek",
    "lora_hajek": "LoRA Hajek",
    "other": "Other",
}


@dataclass
class Run:
    name: str
    config: dict[str, str]
    metrics: list[dict[str, float]]
    family: str
    objective_label: str
    lens_label: str
    rank: Optional[int]
    k: Optional[int]
    tail: Optional[int]
    color: str = FAMILY_BASE["other"]

    @property
    def final_step(self) -> int:
        return int(self.metrics[-1]["step"])

    @property
    def target_step(self) -> int:
        return int(self.config.get("resume_target_step") or self.config.get("num_steps") or 0)

    @property
    def completion_ratio(self) -> float:
        target = self.target_step or 1
        return self.final_step / target

    @property
    def final_loss(self) -> float:
        return self.metrics[-1]["loss"]

    @property
    def best_loss(self) -> float:
        return min(row["loss"] for row in self.metrics)

    @property
    def avg_tok_per_sec_last50(self) -> float:
        tail = self.metrics[-50:] if len(self.metrics) >= 50 else self.metrics
        return mean(row["tok_per_sec"] for row in tail)

    @property
    def peak_memory_gb(self) -> float:
        return max(row["peak_memory_gb"] for row in self.metrics)

    @property
    def wallclock_hours(self) -> float:
        return sum(row["dt_ms"] for row in self.metrics) / 3_600_000.0

    @property
    def tokens_seen_billion(self) -> float:
        return self.metrics[-1]["total_tokens"] / 1e9

    def metric_series(self, key: str) -> list[float]:
        return [row[key] for row in self.metrics]

    def step_series(self) -> list[int]:
        return [int(row["step"]) for row in self.metrics]


def parse_int(value: str) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return int(value)


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(1, rgb[0])) * 255),
        int(max(0, min(1, rgb[1])) * 255),
        int(max(0, min(1, rgb[2])) * 255),
    )


def adjust_lightness(hex_color: str, amount: float) -> str:
    r, g, b = hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.22, min(0.82, l + amount))
    return rgb01_to_hex(colorsys.hls_to_rgb(h, l, s))


def classify_family(cfg: dict[str, str]) -> tuple[str, str, str]:
    lens = cfg["lens_type"]
    loss = cfg["loss_type"]
    mode = cfg["subset_kl_mode"] or ""
    if loss == "kl":
        family = f"{lens}_kl"
        return family, "full_kl", lens
    if mode == "hajek":
        family = f"{lens}_hajek"
        return family, "subset_hajek", lens
    return family_from_topk(lens), "subset_topk", lens


def family_from_topk(lens: str) -> str:
    return f"{lens}_topk"


def load_run(run_dir: Path) -> Run:
    with (run_dir / "run_config.csv").open() as f:
        config = next(csv.DictReader(f))
    with (run_dir / "metrics.csv").open() as f:
        metrics = [
            {
                "step": float(row["step"]),
                "total_tokens": float(row["total_tokens"]),
                "loss": float(row["loss"]),
                "tok_per_sec": float(row["tok_per_sec"]),
                "dt_ms": float(row["dt_ms"]),
                "microsteps": float(row["microsteps"]),
                "lr": float(row["lr"]),
                "peak_memory_gb": float(row["peak_memory_gb"]),
            }
            for row in csv.DictReader(f)
        ]
    family, objective_label, lens_label = classify_family(config)
    return Run(
        name=run_dir.name,
        config=config,
        metrics=metrics,
        family=family,
        objective_label=objective_label,
        lens_label=lens_label,
        rank=parse_int(config.get("lora_rank", "")),
        k=parse_int(config.get("subset_kl_k", "")),
        tail=parse_int(config.get("subset_kl_k_tail", "")),
    )


def load_runs(root: Path) -> list[Run]:
    runs = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        if (run_dir / "metrics.csv").exists() and (run_dir / "run_config.csv").exists():
            runs.append(load_run(run_dir))
    return runs


def assign_family_colors(runs: list[Run]) -> list[Run]:
    grouped: dict[str, list[Run]] = {}
    for run in runs:
        grouped.setdefault(run.family, []).append(run)

    for family, family_runs in grouped.items():
        family_runs.sort(key=lambda run: run.name)
        base = FAMILY_BASE.get(family, FAMILY_BASE["other"])
        count = len(family_runs)
        for idx, run in enumerate(family_runs):
            if count == 1:
                run.color = base
            else:
                offset = -0.14 + 0.28 * idx / (count - 1)
                run.color = adjust_lightness(base, offset)
    return runs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summary_csv(runs: list[Run], path: Path) -> None:
    ensure_dir(path.parent)
    fields = [
        "name",
        "family",
        "lens_type",
        "objective",
        "rank",
        "k",
        "tail",
        "final_step",
        "target_step",
        "completion_ratio",
        "final_loss",
        "best_loss",
        "avg_tok_per_sec_last50",
        "peak_memory_gb",
        "wallclock_hours",
        "tokens_seen_billion",
        "batch_size",
        "lr_init",
        "tokens_per_step",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "name": run.name,
                    "family": run.family,
                    "lens_type": run.config["lens_type"],
                    "objective": run.objective_label,
                    "rank": run.rank or "",
                    "k": run.k or "",
                    "tail": run.tail or "",
                    "final_step": run.final_step,
                    "target_step": run.target_step,
                    "completion_ratio": f"{run.completion_ratio:.4f}",
                    "final_loss": f"{run.final_loss:.8f}",
                    "best_loss": f"{run.best_loss:.8f}",
                    "avg_tok_per_sec_last50": f"{run.avg_tok_per_sec_last50:.4f}",
                    "peak_memory_gb": f"{run.peak_memory_gb:.4f}",
                    "wallclock_hours": f"{run.wallclock_hours:.4f}",
                    "tokens_seen_billion": f"{run.tokens_seen_billion:.4f}",
                    "batch_size": run.config["batch_size"],
                    "lr_init": run.config["lr_init"],
                    "tokens_per_step": run.config["tokens_per_step"],
                }
            )


def save_family_summary_csv(runs: list[Run], path: Path) -> None:
    ensure_dir(path.parent)
    families = sorted({run.family for run in runs})
    fields = [
        "family",
        "num_runs",
        "mean_final_loss",
        "mean_best_loss",
        "mean_avg_tok_per_sec_last50",
        "mean_peak_memory_gb",
        "mean_wallclock_hours",
        "mean_completion_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for family in families:
            group = [run for run in runs if run.family == family]
            writer.writerow(
                {
                    "family": family,
                    "num_runs": len(group),
                    "mean_final_loss": f"{mean(run.final_loss for run in group):.8f}",
                    "mean_best_loss": f"{mean(run.best_loss for run in group):.8f}",
                    "mean_avg_tok_per_sec_last50": f"{mean(run.avg_tok_per_sec_last50 for run in group):.4f}",
                    "mean_peak_memory_gb": f"{mean(run.peak_memory_gb for run in group):.4f}",
                    "mean_wallclock_hours": f"{mean(run.wallclock_hours for run in group):.4f}",
                    "mean_completion_ratio": f"{mean(run.completion_ratio for run in group):.4f}",
                }
            )


def line_plot(runs: list[Run], key: str, ylabel: str, title: str, outpath: Path, logy: bool = False) -> None:
    ensure_dir(outpath.parent)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fffdf8")
    for run in runs:
        ax.plot(run.step_series(), run.metric_series(key), linewidth=2.1, label=run.name, color=run.color)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def bar_plot(
    runs: list[Run], values: list[float], ylabel: str, title: str, outpath: Path, rotate: bool = True
) -> None:
    ensure_dir(outpath.parent)
    labels = [run.name for run in runs]
    colors = [run.color for run in runs]
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.85), 7), dpi=180)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fffdf8")
    ax.bar(range(len(labels)), values, color=colors, edgecolor="#ffffff", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55 if rotate else 0, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def scatter_plot(runs: list[Run], x_key: str, y_key: str, title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    ensure_dir(outpath.parent)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fffdf8")
    family_handles = []
    seen_families = set()
    for run in runs:
        x = getattr(run, x_key)
        y = getattr(run, y_key)
        base_color = FAMILY_BASE.get(run.family, FAMILY_BASE["other"])
        ax.scatter(x, y, s=65, color=base_color, alpha=0.78, edgecolors="#ffffff", linewidths=0.5)
        if run.family not in seen_families:
            family_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=base_color,
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.5,
                    markersize=8,
                    label=FAMILY_LABELS.get(run.family, run.family),
                )
            )
            seen_families.add(run.family)

    label_runs: list[Run] = []
    incomplete = [run for run in runs if run.completion_ratio < 0.999]
    if incomplete:
        label_runs.extend(incomplete)
    y_sorted = sorted(runs, key=lambda run: getattr(run, y_key))
    for candidate in y_sorted[:2] + y_sorted[-2:]:
        if candidate not in label_runs:
            label_runs.append(candidate)
    x_sorted = sorted(runs, key=lambda run: getattr(run, x_key))
    for candidate in x_sorted[:1] + x_sorted[-1:]:
        if candidate not in label_runs:
            label_runs.append(candidate)
    for run in label_runs:
        ax.annotate(
            run.name,
            (getattr(run, x_key), getattr(run, y_key)),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            alpha=0.9,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(handles=family_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def build_groups(run_map: dict[str, Run]) -> dict[str, list[Run]]:
    def names(*items: str) -> list[Run]:
        return [run_map[name] for name in items if name in run_map]

    groups = {
        "01_full_kl_baselines": names(
            "tuned_kl_baseline",
            "lora_kl_r64",
            "lora_kl_r128",
            "lora_kl_r256",
            "lora_kl_r384",
        ),
        "02_tuned_estimator_comparison": names(
            "tuned_kl_baseline",
            "tuned_subset_topk_k256",
            "tuned_subset_topk_k512",
            "tuned_subset_topk_k768",
            "tuned_subset_topk_k1024",
            "tuned_subset_hajek_k256_tail256",
            "tuned_subset_hajek_k512_tail512",
            "tuned_subset_hajek_k768_tail768",
            "tuned_subset_hajek_k1024_tail1024",
        ),
        "03_tuned_hajek_tail_ablation": names(
            "tuned_kl_baseline",
            "tuned_subset_hajek_k512_tail512",
            "tuned_subset_hajek_k512_tail1024",
            "tuned_subset_hajek_k1024_tail512",
            "tuned_subset_hajek_k1024_tail1024",
        ),
        "06_tuned_topk_vs_hajek_matched": names(
            "tuned_kl_baseline",
            "tuned_subset_topk_k256",
            "tuned_subset_hajek_k256_tail256",
            "tuned_subset_topk_k512",
            "tuned_subset_hajek_k512_tail512",
            "tuned_subset_topk_k768",
            "tuned_subset_hajek_k768_tail768",
            "tuned_subset_topk_k1024",
            "tuned_subset_hajek_k1024_tail1024",
        ),
        "07_lora_topk_vs_hajek_r64": names(
            "tuned_kl_baseline",
            "lora_subset_topk_r64_k256",
            "lora_subset_hajek_r64_k256_tail256",
            "lora_subset_topk_r64_k512",
            "lora_subset_hajek_r64_k512_tail512",
            "lora_subset_topk_r64_k768",
            "lora_subset_hajek_r64_k768_tail768",
            "lora_subset_topk_r64_k1024",
            "lora_subset_hajek_r64_k1024_tail1024",
        ),
    }
    for k in (256, 512, 768, 1024):
        groups[f"04_lora_topk_rank_sweep_k{k}"] = names(
            "tuned_kl_baseline",
            f"lora_subset_topk_r64_k{k}",
            f"lora_subset_topk_r128_k{k}",
            f"lora_subset_topk_r256_k{k}",
            f"lora_subset_topk_r384_k{k}",
        )
    for k, tail in ((256, 256), (512, 512), (768, 768), (1024, 1024)):
        groups[f"05_lora_hajek_rank_sweep_k{k}_tail{tail}"] = names(
            "tuned_kl_baseline",
            f"lora_subset_hajek_r64_k{k}_tail{tail}",
            f"lora_subset_hajek_r128_k{k}_tail{tail}",
            f"lora_subset_hajek_r256_k{k}_tail{tail}",
        )
    return {name: group for name, group in groups.items() if group}


def render_group(group_name: str, runs: list[Run], out_root: Path) -> None:
    group_dir = out_root / group_name
    line_plot(runs, "loss", "Training Loss", f"{group_name}: loss vs step", group_dir / "loss.png")
    line_plot(
        runs,
        "tok_per_sec",
        "Tokens / sec",
        f"{group_name}: throughput vs step",
        group_dir / "tok_per_sec.png",
    )
    line_plot(
        runs,
        "peak_memory_gb",
        "Peak Memory (GB)",
        f"{group_name}: peak memory vs step",
        group_dir / "peak_memory_gb.png",
    )
    ordered = sorted(runs, key=lambda run: run.final_loss)
    bar_plot(
        ordered,
        [run.final_loss for run in ordered],
        "Final Training Loss",
        f"{group_name}: final loss",
        group_dir / "final_loss.png",
    )
    bar_plot(
        ordered,
        [run.avg_tok_per_sec_last50 for run in ordered],
        "Avg Tokens / sec (last 50)",
        f"{group_name}: throughput summary",
        group_dir / "avg_tok_per_sec_last50.png",
    )
    bar_plot(
        ordered,
        [run.peak_memory_gb for run in ordered],
        "Peak Memory (GB)",
        f"{group_name}: memory summary",
        group_dir / "peak_memory_summary.png",
    )
    bar_plot(
        ordered,
        [run.completion_ratio for run in ordered],
        "Completion Ratio",
        f"{group_name}: completion ratio",
        group_dir / "completion_ratio.png",
    )


def render_global_plots(runs: list[Run], out_root: Path) -> None:
    global_dir = out_root / "00_global"
    scatter_plot(
        runs,
        "wallclock_hours",
        "final_loss",
        "Wallclock vs final training loss",
        "Wallclock Hours",
        "Final Training Loss",
        global_dir / "wallclock_vs_final_loss.png",
    )
    scatter_plot(
        runs,
        "avg_tok_per_sec_last50",
        "final_loss",
        "Throughput vs final training loss",
        "Avg Tokens / sec (last 50)",
        "Final Training Loss",
        global_dir / "throughput_vs_final_loss.png",
    )
    scatter_plot(
        runs,
        "peak_memory_gb",
        "final_loss",
        "Peak memory vs final training loss",
        "Peak Memory (GB)",
        "Final Training Loss",
        global_dir / "memory_vs_final_loss.png",
    )

    ordered = sorted(runs, key=lambda run: (run.family, run.name))
    bar_plot(
        ordered,
        [run.final_loss for run in ordered],
        "Final Training Loss",
        "All runs: final training loss",
        global_dir / "all_runs_final_loss.png",
    )
    bar_plot(
        ordered,
        [run.avg_tok_per_sec_last50 for run in ordered],
        "Avg Tokens / sec (last 50)",
        "All runs: throughput summary",
        global_dir / "all_runs_avg_tok_per_sec_last50.png",
    )
    bar_plot(
        ordered,
        [run.peak_memory_gb for run in ordered],
        "Peak Memory (GB)",
        "All runs: peak memory summary",
        global_dir / "all_runs_peak_memory_gb.png",
    )


def save_manifest(groups: dict[str, list[Run]], outpath: Path) -> None:
    ensure_dir(outpath.parent)
    with outpath.open("w") as f:
        json.dump({group: [run.name for run in runs] for group, runs in groups.items()}, f, indent=2, sort_keys=True)


def main() -> None:
    runs = assign_family_colors(load_runs(ROOT))
    ensure_dir(OUT)
    save_summary_csv(runs, OUT / "training_run_summary.csv")
    save_family_summary_csv(runs, OUT / "training_family_summary.csv")
    run_map = {run.name: run for run in runs}
    groups = build_groups(run_map)
    save_manifest(groups, OUT / "group_manifest.json")
    render_global_plots(runs, OUT)
    for group_name, group_runs in groups.items():
        render_group(group_name, group_runs, OUT)


if __name__ == "__main__":
    main()
