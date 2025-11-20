from typing import Iterable
from dataclasses import dataclass


@dataclass
class StreamingTextSource:
    """
    Minimal interface: yields strings.
    Compatible with ChunkedTextDataset expecting an object with __iter__ → strings.
    """
    iterable: Iterable[str]

    def __iter__(self):
        yield from self.iterable