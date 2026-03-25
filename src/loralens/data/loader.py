<<<<<<< HEAD
# src/loralens/data/loader.py
"""
Data loading utilities for text datasets.

Provides streaming text sources and chunked tokenization for efficient
training on large text corpora.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
=======
# loralens/data/loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Dict, Any, List, Union

import json
>>>>>>> origin/main

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


<<<<<<< HEAD
@dataclass
class TextSourceConfig:
    """Configuration for text file source."""
    paths: List[Union[str, Path]]
    mode: str = "lines"  # "lines", "whole", or "jsonl"
=======
# Configs

@dataclass
class TextSourceConfig:
    paths: List[Union[str, Path]]
    # How to interpret each file
    # - "lines": each line is a document
    # - "whole": whole file is a single document
    # - "jsonl": each line is a JSON object, use json_field as the text
    mode: str = "lines"
>>>>>>> origin/main
    json_field: str = "text"
    encoding: str = "utf-8"


@dataclass
class ChunkConfig:
<<<<<<< HEAD
    """Configuration for text chunking."""
=======
>>>>>>> origin/main
    seq_len: int
    stride: Optional[int] = None  # if None, no overlap (stride = seq_len)
    drop_remainder: bool = True   # drop final short chunk
    add_special_tokens: bool = True
<<<<<<< HEAD
    document_separated: bool = True  # chunk documents independently


class TextSource:
    """
    Streams raw text documents from disk.

    Parameters
    ----------
    config : TextSourceConfig
        Configuration for the text source.
    """

    def __init__(self, config: TextSourceConfig) -> None:
=======
    # If True, we treat each document independently.
    # If False, we concatenate all docs into one long stream.
    document_separated: bool = True


# Text sources

class TextSource:
    """
    Streams raw text documents from disk.
    """

    def __init__(self, config: TextSourceConfig):
>>>>>>> origin/main
        self.config = config
        self.paths = [Path(p) for p in config.paths]

    def __iter__(self) -> Iterator[str]:
        mode = self.config.mode
        if mode == "lines":
            yield from self._iter_lines()
        elif mode == "whole":
            yield from self._iter_whole_files()
        elif mode == "jsonl":
            yield from self._iter_jsonl()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _iter_lines(self) -> Iterator[str]:
        for path in self.paths:
            with path.open("r", encoding=self.config.encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def _iter_whole_files(self) -> Iterator[str]:
        for path in self.paths:
            with path.open("r", encoding=self.config.encoding) as f:
                text = f.read().strip()
                if text:
                    yield text

    def _iter_jsonl(self) -> Iterator[str]:
        field = self.config.json_field
        for path in self.paths:
            with path.open("r", encoding=self.config.encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get(field, "").strip()
                    if text:
                        yield text


<<<<<<< HEAD
=======
# Chunking utilities

>>>>>>> origin/main
def _yield_chunks_from_ids(
    input_ids: List[int],
    chunk_cfg: ChunkConfig,
) -> Iterator[List[int]]:
<<<<<<< HEAD
    """Yield fixed-length chunks from a list of token IDs."""
=======
>>>>>>> origin/main
    seq_len = chunk_cfg.seq_len
    stride = chunk_cfg.stride or seq_len

    if len(input_ids) == 0:
        return

<<<<<<< HEAD
    start = 0
    while start < len(input_ids):
        end = start + seq_len
        chunk = input_ids[start:end]
        if len(chunk) < seq_len and chunk_cfg.drop_remainder:
            break
        yield chunk
        start += stride


class ChunkedTextDataset(IterableDataset):
    """
    IterableDataset that streams documents, tokenizes, and chunks them.

    Parameters
    ----------
    text_source : TextSource
        Source of text documents.
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer.
    chunk_cfg : ChunkConfig
        Chunking configuration.
    max_docs : Optional[int]
        Maximum number of documents to process.
=======
    if chunk_cfg.document_separated:
        start = 0
        while start < len(input_ids):
            end = start + seq_len
            chunk = input_ids[start:end]
            if len(chunk) < seq_len and chunk_cfg.drop_remainder:
                break
            yield chunk
            start += stride
    else:
        # For concatenated mode, we expect you to concatenate before calling this
        raise RuntimeError(
            "document_separated=False is not handled here; "
            "use a higher-level concatenation wrapper."
        )


# IterableDataset: text -> tokens -> chunks

class ChunkedTextDataset(IterableDataset):
    """
    IterableDataset that:

    - streams documents from TextSource
    - tokenizes with a HuggingFace tokenizer
    - slices into fixed-length token chunks (with optional overlap)
>>>>>>> origin/main
    """

    def __init__(
        self,
<<<<<<< HEAD
        text_source,
        tokenizer: PreTrainedTokenizerBase,
        chunk_cfg: ChunkConfig,
        max_docs: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
=======
        text_source: TextSource,
        tokenizer: PreTrainedTokenizerBase,
        chunk_cfg: ChunkConfig,
        max_docs: Optional[int] = None,
    ):
>>>>>>> origin/main
        super().__init__()
        self.text_source = text_source
        self.tokenizer = tokenizer
        self.chunk_cfg = chunk_cfg
        self.max_docs = max_docs
<<<<<<< HEAD
        self.rank = rank
        self.world_size = world_size

        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, world_size), got rank={self.rank}, "
                f"world_size={self.world_size}"
            )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        chunk_cfg = self.chunk_cfg

        if chunk_cfg.document_separated:
            yield from self._iter_document_separated()
        else:
            yield from self._iter_concatenated()

    def _iter_documents(self) -> Iterator[str]:
        """Shard the document stream across ranks to avoid duplicated training data."""
        for doc_idx, doc in enumerate(self.text_source):
            if doc_idx % self.world_size != self.rank:
                continue
            yield doc

    def _iter_document_separated(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Chunk each document independently."""
        doc_count = 0
        for doc in self._iter_documents():
            if self.max_docs is not None and doc_count >= self.max_docs:
                break
            doc_count += 1

            encoded = self.tokenizer(
                doc,
                add_special_tokens=self.chunk_cfg.add_special_tokens,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = encoded["input_ids"]

            for chunk_ids in _yield_chunks_from_ids(input_ids, self.chunk_cfg):
=======

    def __iter__(self):
        tokenizer = self.tokenizer
        chunk_cfg = self.chunk_cfg
        seq_len = chunk_cfg.seq_len
        stride = chunk_cfg.stride or seq_len

        # Mode 1: document-separated 
        if chunk_cfg.document_separated:
            doc_count = 0
            for doc in self.text_source:
                if self.max_docs is not None and doc_count >= self.max_docs:
                    break
                doc_count += 1

                encoded = tokenizer(
                    doc,
                    add_special_tokens=chunk_cfg.add_special_tokens,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                input_ids = encoded["input_ids"]

                for chunk_ids in _yield_chunks_from_ids(input_ids, chunk_cfg):
                    # No padding here; each chunk is already fixed length
                    input_tensor = torch.tensor(chunk_ids, dtype=torch.long)
                    attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
                    yield {
                        "input_ids": input_tensor,
                        "attention_mask": attention_mask,
                    }

        # Mode 2: concatenated stream across documents
        else:
            buffer: List[int] = []
            doc_count = 0

            for doc in self.text_source:
                if self.max_docs is not None and doc_count >= self.max_docs:
                    break
                doc_count += 1

                encoded = tokenizer(
                    doc,
                    add_special_tokens=chunk_cfg.add_special_tokens,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                input_ids = encoded["input_ids"]

                # Append tokens for this document to the running buffer
                buffer.extend(input_ids)

                # While we have at least one full chunk, emit it
                while len(buffer) >= seq_len:
                    chunk_ids = buffer[:seq_len]
                    # Drop `stride` tokens from the left (stride == seq_len ⇒ no overlap)
                    buffer = buffer[stride:]

                    input_tensor = torch.tensor(chunk_ids, dtype=torch.long)
                    attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
                    yield {
                        "input_ids": input_tensor,
                        "attention_mask": attention_mask,
                    }

            # At the end of the stream, drop any leftover tail if drop_remainder=True
            if not chunk_cfg.drop_remainder and buffer:
                # Emit one final shorter chunk (will need dynamic padding collate_fn)
                chunk_ids = buffer
>>>>>>> origin/main
                input_tensor = torch.tensor(chunk_ids, dtype=torch.long)
                attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
                yield {
                    "input_ids": input_tensor,
                    "attention_mask": attention_mask,
                }

<<<<<<< HEAD
    def _iter_concatenated(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Concatenate all documents into a single token stream."""
        seq_len = self.chunk_cfg.seq_len
        stride = self.chunk_cfg.stride or seq_len

        buffer: List[int] = []
        doc_count = 0

        for doc in self._iter_documents():
            if self.max_docs is not None and doc_count >= self.max_docs:
                break
            doc_count += 1

            encoded = self.tokenizer(
                doc,
                add_special_tokens=self.chunk_cfg.add_special_tokens,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            buffer.extend(encoded["input_ids"])

            while len(buffer) >= seq_len:
                chunk_ids = buffer[:seq_len]
                buffer = buffer[stride:]

                input_tensor = torch.tensor(chunk_ids, dtype=torch.long)
                attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
                yield {
                    "input_ids": input_tensor,
                    "attention_mask": attention_mask,
                }

        # Handle remaining buffer
        if not self.chunk_cfg.drop_remainder and buffer:
            input_tensor = torch.tensor(buffer, dtype=torch.long)
            attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
            yield {
                "input_ids": input_tensor,
                "attention_mask": attention_mask,
            }

=======

# Collate function
>>>>>>> origin/main

def fixed_length_collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
<<<<<<< HEAD
    """Collate function for batches where all sequences are the same length."""
=======
    """
    Collate function for batches where all sequences are already fixed length.
    """
>>>>>>> origin/main
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def dynamic_pad_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
<<<<<<< HEAD
    """Collate function that pads sequences to max length in batch."""
=======
    """
    If you later support variable-length chunks, this pads to max length in batch.
    """
>>>>>>> origin/main
    seqs = [item["input_ids"] for item in batch]
    masks = [item["attention_mask"] for item in batch]

    max_len = max(seq.size(0) for seq in seqs)

    padded_ids = []
    padded_masks = []

    for ids, mask in zip(seqs, masks):
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat(
                [ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)], dim=0
            )
            mask = torch.cat(
                [mask, torch.zeros(pad_len, dtype=mask.dtype)], dim=0
            )
        padded_ids.append(ids)
        padded_masks.append(mask)

<<<<<<< HEAD
    return {
        "input_ids": torch.stack(padded_ids, dim=0),
        "attention_mask": torch.stack(padded_masks, dim=0),
    }
=======
    input_ids = torch.stack(padded_ids, dim=0)
    attention_mask = torch.stack(padded_masks, dim=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}
>>>>>>> origin/main
