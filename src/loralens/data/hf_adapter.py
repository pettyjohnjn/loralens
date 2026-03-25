# src/loralens/data/hf_adapter.py
"""
Adapters for HuggingFace datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class StreamingTextSource:
    """
    Minimal interface: yields strings from an iterable.

    Compatible with ChunkedTextDataset expecting an object with __iter__ -> strings.

    Parameters
    ----------
    iterable : Iterable[str]
        Source of text strings.
    """
    iterable: Iterable[str]

    def __iter__(self) -> Iterator[str]:
        yield from self.iterable
