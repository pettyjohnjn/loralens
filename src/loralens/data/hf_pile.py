# src/loralens/data/hf_pile.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from datasets import load_dataset


@dataclass
class PileSourceConfig:
    dataset_name: str = "monology/pile-uncopyrighted"
    subset_name: str = None                  # this dataset has no subsets
    split: str = "train"
    streaming: bool = True
    shuffle_buffer_size: int = 10_000
    text_field: str = "text"                 # field contains raw text


class PileSource:
    """
    Streaming interface for the monology/pile-uncopyrighted dataset.
    Yields raw text strings, one per HF sample.
    """

    def __init__(self, config: PileSourceConfig):
        self.config = config

        # HF API: subset_name must be None for this dataset
        ds = load_dataset(
            config.dataset_name,
            config.subset_name,
            split=config.split,
            streaming=config.streaming,
        )

        if config.streaming:
            ds = ds.shuffle(buffer_size=config.shuffle_buffer_size)

        self.ds = ds

    def __iter__(self) -> Iterable[str]:
        for sample in self.ds:
            yield sample[self.config.text_field]