#!/usr/bin/env python3
"""
Reorganize checkpoints from ad-hoc experiment directories to the canonical
attribute-based hierarchy:

    checkpoints/{model}/{lens}/{loss}/{sites}/{init}/{tag}/

Where:
    model  = slugified HF model name (e.g. gpt2, llama-3-8b-instruct)
    lens   = lens type + rank  (e.g. lora-r64, tuned, logit)
    loss   = loss slug         (e.g. kl, subset_kl-topk-k128, subset_kl-mc-k256-tail256)
    sites  = activation preset (e.g. residual, gpt2_expanded)
    init   = LoRA init mode    (e.g. init-default, init-ridge)  [lora only]
    tag    = run discriminator (e.g. seed0, lr2em3)

Runs using retired estimators (hajek, frankenstein) or retired init modes
(identity, procrustes, combined, rollout_svd, ridge_ls) are moved to trash/.

Usage:
    # Preview without making any changes:
    python scripts/migrate_checkpoints.py --dry-run

    # Execute:
    python scripts/migrate_checkpoints.py
"""

import argparse
import csv
import math
import os
import re
import shutil
import sys
from pathlib import Path

# ── Retired modes/inits (these go to trash) ───────────────────────────────────
RETIRED_MODES = {"hajek", "frankenstein"}
RETIRED_INITS = {"identity", "procrustes", "combined", "rollout_svd", "ridge_ls"}

# ── Directories that are eval-stage copies; send to trash ─────────────────────
EVAL_STAGE_PREFIXES = {"eval_stage_"}

# ── Non-lens directories (probes, analysis data) ──────────────────────────────
NON_LENS_DIRS = {
    "llama3_70b_feasibility_probe",
    "llama3_70b_feasibility_probe_k512_actualrun",
}


# ── Utility functions ──────────────────────────────────────────────────────────

def slugify_model_name(model_name: str) -> str:
    """
    Convert an HF model name or absolute cluster path to a short filesystem slug.

    Examples
    --------
    'gpt2'                                             → 'gpt2'
    '/path/Llama-3.3-70B-Instruct'                    → 'llama-3-3-70b-instruct'
    '/path/models--meta-llama--Meta-Llama-3-8B-Instruct' → 'meta-llama-3-8b-instruct'
    """
    # Take last path component (strips cluster paths and HF org prefixes)
    name = model_name.strip().split("/")[-1]
    # HF cache dirs look like "models--{org}--{model}"; strip the prefix and take model
    name = re.sub(r"^models--", "", name)
    if "--" in name:
        name = name.split("--")[-1]  # take just the model name portion
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return name


def lr_to_str(lr: float) -> str:
    """
    Format a learning rate as a compact filesystem-safe string.

    Examples: 0.001 → '1em3',  0.0015 → '1p5em3',  0.002 → '2em3'
    """
    if lr == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(lr))))
    mantissa = lr / 10 ** exp
    # Round to at most 2 significant figures
    mantissa_round = round(mantissa, 1)
    mantissa_str = f"{mantissa_round:.1f}".rstrip("0").rstrip(".")
    mantissa_str = mantissa_str.replace(".", "p")
    exp_str = f"em{abs(exp)}" if exp < 0 else f"ep{exp}"
    return f"{mantissa_str}{exp_str}"


def parse_config(csv_path: Path) -> dict:
    """Return the first data row of a run_config.csv as a dict."""
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return next(reader)


def canonical_path(cfg: dict) -> Path:
    """Derive the canonical relative path from a config dict."""
    model_slug = slugify_model_name(cfg.get("model_name", "unknown"))

    lens_type = cfg.get("lens_type", "lora")
    lora_rank = cfg.get("lora_rank", "16") or "16"
    if lens_type == "lora":
        lens_slug = f"lora-r{lora_rank}"
    else:
        lens_slug = lens_type  # "tuned" or "logit"

    loss_type = cfg.get("loss_type", "kl")
    mode = (cfg.get("subset_kl_mode") or "").strip()
    k = (cfg.get("subset_kl_k") or "128").strip()
    k_tail = (cfg.get("subset_kl_k_tail") or "0").strip()
    top_m = (cfg.get("shared_subset_top_m") or "16").strip()
    max_K = (cfg.get("shared_subset_max_K") or "512").strip()

    if loss_type == "subset_kl":
        loss_slug = f"subset_kl-{mode or 'topk'}-k{k}"
        if k_tail and int(k_tail) > 0:
            loss_slug += f"-tail{k_tail}"
    elif loss_type == "shared_subset_kl":
        loss_slug = f"shared_kl-m{top_m}-K{max_K}"
    else:
        loss_slug = loss_type  # "kl" or "ce"

    site_slug = cfg.get("activation_site_preset", "residual") or "residual"

    parts = [model_slug, lens_slug, loss_slug, site_slug]

    # LoRA init (only meaningful for lora)
    if lens_type == "lora":
        lora_init = (
            cfg.get("lora_init")
            or cfg.get("lora_init_strategy")
            or "default_lora"
        ).strip()
        init_map = {
            "default_lora": "init-default",
            "mean_shift": "init-mean_shift",
            "ridge_svd": "init-ridge",
        }
        parts.append(init_map.get(lora_init, f"init-{lora_init}"))

    # Run tag: use lr when non-default, otherwise seed
    DEFAULT_LR = 0.001
    try:
        lr = float(cfg.get("lr_init", "0.001") or "0.001")
    except ValueError:
        lr = DEFAULT_LR

    seed = cfg.get("seed", "0") or "0"

    if abs(lr - DEFAULT_LR) / DEFAULT_LR > 0.01:
        run_tag = f"lr{lr_to_str(lr)}"
    else:
        run_tag = f"seed{seed}"

    parts.append(run_tag)

    result = Path(parts[0])
    for p in parts[1:]:
        result = result / p
    return result


def retire_reason(cfg: dict) -> str | None:
    """Return a reason string if this run should go to trash, else None."""
    mode = (cfg.get("subset_kl_mode") or "").strip()
    if mode in RETIRED_MODES:
        return f"retired mode '{mode}'"

    # Support both new column name (lora_init) and old names (lora_init_strategy,
    # tuned_init_strategy) used in pre-refactor run_config.csv files.
    lora_init = (
        cfg.get("lora_init")
        or cfg.get("lora_init_strategy")
        or cfg.get("tuned_init_strategy")
        or ""
    ).strip()
    if lora_init in RETIRED_INITS:
        return f"retired init '{lora_init}'"

    return None


# ── Core migration logic ───────────────────────────────────────────────────────

class Migrator:
    def __init__(self, root: Path, trash: Path, dry_run: bool):
        self.root = root
        self.trash = trash
        self.dry_run = dry_run
        self.moves: list[tuple[Path, Path]] = []   # (src, dst)
        self.trashed: list[tuple[Path, str]] = []  # (src, reason)
        self.skipped: list[tuple[Path, str]] = []  # (src, reason)
        self.taken: set[Path] = set()             # canonical paths already claimed

    def _move(self, src: Path, dst: Path) -> None:
        if self.dry_run:
            print(f"  MOVE  {src.relative_to(self.root)}")
            print(f"     →  {dst.relative_to(self.root)}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    def _trash(self, src: Path, reason: str) -> None:
        dst = self.trash / src.relative_to(self.root)
        if self.dry_run:
            print(f"  TRASH [{reason}]  {src.relative_to(self.root)}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    def _resolve_conflict(self, canonical: Path) -> Path:
        """If canonical already taken, append _v2, _v3, …"""
        if canonical not in self.taken:
            return canonical
        for i in range(2, 99):
            alt = canonical.parent / (canonical.name + f"_v{i}")
            if alt not in self.taken:
                return alt
        raise RuntimeError(f"Too many conflicts for {canonical}")

    def process_run_dir(self, run_dir: Path, cfg: dict) -> None:
        """Classify a single run directory and record its disposition."""
        reason = retire_reason(cfg)
        if reason:
            self.trashed.append((run_dir, reason))
            self._trash(run_dir, reason)
            return

        rel = canonical_path(cfg)
        dst = self._resolve_conflict(self.root / rel)
        self.taken.add(dst)
        self.moves.append((run_dir, dst))
        self._move(run_dir, dst)

    def process_non_lens_dir(self, d: Path, reason: str) -> None:
        dst = self.root / "analysis" / "feasibility" / d.name
        if self.dry_run:
            print(f"  ANALYSIS [{reason}]  {d.relative_to(self.root)}")
            print(f"         →  {dst.relative_to(self.root)}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dst))

    def remove_empty_dirs(self) -> None:
        """Remove empty directories left behind after moves (bottom-up)."""
        for dirpath, dirnames, filenames in os.walk(self.root, topdown=False):
            p = Path(dirpath)
            # Never remove root itself or trash
            if p == self.root or p == self.trash:
                continue
            # Skip the new canonical tree (it's populated)
            try:
                _ = p.relative_to(self.trash)
                continue
            except ValueError:
                pass
            if not any(p.iterdir()):
                if self.dry_run:
                    print(f"  RMDIR {p.relative_to(self.root)}")
                else:
                    p.rmdir()


def is_eval_stage(top_dir: Path) -> bool:
    return any(top_dir.name.startswith(pfx) for pfx in EVAL_STAGE_PREFIXES)


def is_non_lens_dir(top_dir: Path) -> bool:
    return top_dir.name in NON_LENS_DIRS


def find_run_dirs(root: Path):
    """
    Yield (run_dir, cfg_dict | None) for every leaf checkpoint directory.

    A leaf run dir is one that directly contains run_config.csv or lens_step_*.pt.
    """
    found = set()
    # Prefer dirs that have run_config.csv (most informative)
    for csv_path in sorted(root.rglob("run_config.csv")):
        run_dir = csv_path.parent
        found.add(run_dir)
        yield run_dir, parse_config(csv_path)

    # Also catch dirs with only lens_step_*.pt (no config CSV)
    for pt_path in sorted(root.rglob("lens_step_*.pt")):
        run_dir = pt_path.parent
        if run_dir not in found:
            found.add(run_dir)
            yield run_dir, None


def run(root: Path, trash_root: Path, dry_run: bool) -> None:
    print(f"\n{'DRY RUN — ' if dry_run else ''}Migrating checkpoints in {root}\n")

    migrator = Migrator(root, trash_root, dry_run)
    skip_subtrees: list[Path] = []  # top-level dirs already handled wholesale

    # ── Top-level special directories ─────────────────────────────────────────
    for top_dir in sorted(root.iterdir()):
        if not top_dir.is_dir():
            continue

        if is_eval_stage(top_dir):
            print(f"\n[eval_stage copy] Trashing entire directory: {top_dir.name}")
            migrator._trash(top_dir, "eval_stage duplicate")
            skip_subtrees.append(top_dir)
            continue

        if is_non_lens_dir(top_dir):
            print(f"\n[non-lens probe] Moving to analysis/: {top_dir.name}")
            migrator.process_non_lens_dir(top_dir, "feasibility probe")
            skip_subtrees.append(top_dir)
            continue

    def _under_skip(p: Path) -> bool:
        for s in skip_subtrees:
            try:
                p.relative_to(s)
                return True
            except ValueError:
                pass
        return False

    # ── Individual run directories ─────────────────────────────────────────────
    print("\n=== Run directories ===")
    no_config = []
    for run_dir, cfg in find_run_dirs(root):
        # Skip dirs already handled wholesale above
        if _under_skip(run_dir):
            continue

        # Skip anything under trash or analysis
        try:
            run_dir.relative_to(trash_root)
            continue
        except ValueError:
            pass
        if (root / "analysis").is_dir():
            try:
                run_dir.relative_to(root / "analysis")
                continue
            except ValueError:
                pass

        if cfg is None:
            no_config.append(run_dir)
            continue

        migrator.process_run_dir(run_dir, cfg)

    # ── Report runs with no config ─────────────────────────────────────────────
    if no_config:
        print(f"\n=== {len(no_config)} run dir(s) with no run_config.csv (skipped) ===")
        for d in no_config:
            print(f"  {d.relative_to(root)}")

    # ── Remove leftover empty directories ─────────────────────────────────────
    if not dry_run:
        migrator.remove_empty_dirs()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Moved to canonical:  {len(migrator.moves)}")
    print(f"  Sent to trash:       {len(migrator.trashed)}")
    print(f"  Skipped (no config): {len(no_config)}")

    if migrator.trashed:
        print("\nTrashed runs:")
        for src, reason in migrator.trashed:
            print(f"  {src.relative_to(root)}  [{reason}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "src" / "checkpoints",
        help="Checkpoint root to reorganize (default: src/checkpoints)",
    )
    parser.add_argument(
        "--trash",
        type=Path,
        default=repo_root / "trash",
        help="Trash directory for retired/duplicate runs (default: trash/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without moving anything.",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"ERROR: {args.root} does not exist or is not a directory", file=sys.stderr)
        sys.exit(1)

    run(args.root, args.trash, args.dry_run)


if __name__ == "__main__":
    main()
