# src/loralens/data/hf_pile.py
"""
Streaming interface for the Pile dataset.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset


@dataclass
class PileSourceConfig:
    """Configuration for Pile dataset source."""
    dataset_name: str = "monology/pile-uncopyrighted"
    subset_name: str = None  # this dataset has no subsets
    split: str = "train"
    streaming: bool = True
    shuffle_buffer_size: int = 10_000
    text_field: str = "text"


class PileSource:
    """
    Streaming interface for the monology/pile-uncopyrighted dataset.

    Yields raw text strings, one per HF sample.

    If the env var PILE_LOCAL_DIR is set (or a default path under HF_HOME exists),
    loads directly from local .jsonl.zst shards to avoid HF Hub network requests.
    """

    # Default shard path relative to HF_HOME hub cache
    _HUB_SNAPSHOT = (
        "hub/datasets--monology--pile-uncopyrighted"
        "/snapshots/3be90335b66f24456a5d6659d9c8d208c0357119/train"
    )

    def __init__(self, config: PileSourceConfig) -> None:
        self.config = config

        local_dir = self._resolve_local_dir()

        if local_dir is not None:
            shards = sorted(glob.glob(os.path.join(local_dir, "*.jsonl.zst")))
            if not shards:
                raise FileNotFoundError(
                    f"PILE_LOCAL_DIR={local_dir!r} contains no *.jsonl.zst files"
                )
            ds = load_dataset(
                "json",
                data_files={config.split: shards},
                streaming=True,
                split=config.split,
            )
        else:
            ds = load_dataset(
                config.dataset_name,
                config.subset_name,
                split=config.split,
                streaming=config.streaming,
            )

        if config.streaming:
            ds = ds.shuffle(buffer_size=config.shuffle_buffer_size)

        self.ds = ds

    @classmethod
    def _resolve_local_dir(cls) -> str | None:
        """Return a local shard directory if available, else None."""
        explicit = os.environ.get("PILE_LOCAL_DIR")
        if explicit:
            return explicit

        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidate = os.path.join(hf_home, cls._HUB_SNAPSHOT)
            if os.path.isdir(candidate):
                return candidate

        return None

    def __iter__(self) -> Iterator[str]:
        for sample in self.ds:
            yield sample[self.config.text_field]
