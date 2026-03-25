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

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


@dataclass
class TextSourceConfig:
    """Configuration for text file source."""
    paths: List[Union[str, Path]]
    mode: str = "lines"  # "lines", "whole", or "jsonl"
    json_field: str = "text"
    encoding: str = "utf-8"


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    seq_len: int
    stride: Optional[int] = None  # if None, no overlap (stride = seq_len)
    drop_remainder: bool = True   # drop final short chunk
    add_special_tokens: bool = True
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


def _yield_chunks_from_ids(
    input_ids: List[int],
    chunk_cfg: ChunkConfig,
) -> Iterator[List[int]]:
    """Yield fixed-length chunks from a list of token IDs."""
    seq_len = chunk_cfg.seq_len
    stride = chunk_cfg.stride or seq_len

    if len(input_ids) == 0:
        return

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
    """

    def __init__(
        self,
        text_source,
        tokenizer: PreTrainedTokenizerBase,
        chunk_cfg: ChunkConfig,
        max_docs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.text_source = text_source
        self.tokenizer = tokenizer
        self.chunk_cfg = chunk_cfg
        self.max_docs = max_docs

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        tokenizer = self.tokenizer
        chunk_cfg = self.chunk_cfg
        seq_len = chunk_cfg.seq_len
        stride = chunk_cfg.stride or seq_len

        if chunk_cfg.document_separated:
            yield from self._iter_document_separated()
        else:
            yield from self._iter_concatenated()

    def _iter_document_separated(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Chunk each document independently."""
        doc_count = 0
        for doc in self.text_source:
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
                input_tensor = torch.tensor(chunk_ids, dtype=torch.long)
                attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
                yield {
                    "input_ids": input_tensor,
                    "attention_mask": attention_mask,
                }

    def _iter_concatenated(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Concatenate all documents into a single token stream."""
        seq_len = self.chunk_cfg.seq_len
        stride = self.chunk_cfg.stride or seq_len

        buffer: List[int] = []
        doc_count = 0

        for doc in self.text_source:
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


def fixed_length_collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Collate function for batches where all sequences are the same length."""
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def dynamic_pad_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Collate function that pads sequences to max length in batch."""
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

    return {
        "input_ids": torch.stack(padded_ids, dim=0),
        "attention_mask": torch.stack(padded_masks, dim=0),
    }
