# src/loralens/data/hf_pile.py
<<<<<<< HEAD
"""
Streaming interface for the Pile dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
=======

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
>>>>>>> origin/main

from datasets import load_dataset


@dataclass
class PileSourceConfig:
<<<<<<< HEAD
    """Configuration for Pile dataset source."""
    dataset_name: str = "monology/pile-uncopyrighted"
    subset_name: str = None  # this dataset has no subsets
    split: str = "train"
    streaming: bool = True
    shuffle_buffer_size: int = 10_000
    text_field: str = "text"
=======
    dataset_name: str = "monology/pile-uncopyrighted"
    subset_name: str = None                  # this dataset has no subsets
    split: str = "train"
    streaming: bool = True
    shuffle_buffer_size: int = 10_000
    text_field: str = "text"                 # field contains raw text
>>>>>>> origin/main


class PileSource:
    """
    Streaming interface for the monology/pile-uncopyrighted dataset.
<<<<<<< HEAD

    Yields raw text strings, one per HF sample.

    Parameters
    ----------
    config : PileSourceConfig
        Configuration for the Pile source.
    """

    def __init__(self, config: PileSourceConfig) -> None:
        self.config = config

=======
    Yields raw text strings, one per HF sample.
    """

    def __init__(self, config: PileSourceConfig):
        self.config = config

        # HF API: subset_name must be None for this dataset
>>>>>>> origin/main
        ds = load_dataset(
            config.dataset_name,
            config.subset_name,
            split=config.split,
            streaming=config.streaming,
        )

        if config.streaming:
            ds = ds.shuffle(buffer_size=config.shuffle_buffer_size)

        self.ds = ds

<<<<<<< HEAD
    def __iter__(self) -> Iterator[str]:
        for sample in self.ds:
            yield sample[self.config.text_field]
=======
    def __iter__(self) -> Iterable[str]:
        for sample in self.ds:
            yield sample[self.config.text_field]
>>>>>>> origin/main
