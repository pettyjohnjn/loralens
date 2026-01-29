# src/loralens/data/hf_pile.py
"""
Streaming interface for the Pile dataset.
"""

from __future__ import annotations

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
    
    Parameters
    ----------
    config : PileSourceConfig
        Configuration for the Pile source.
    """

    def __init__(self, config: PileSourceConfig) -> None:
        self.config = config

        ds = load_dataset(
            config.dataset_name,
            config.subset_name,
            split=config.split,
            streaming=config.streaming,
        )

        if config.streaming:
            ds = ds.shuffle(buffer_size=config.shuffle_buffer_size)

        self.ds = ds

    def __iter__(self) -> Iterator[str]:
        for sample in self.ds:
            yield sample[self.config.text_field]
