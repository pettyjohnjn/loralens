# loralens/data/loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Dict, Any, List, Union

import json

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase


# Configs

@dataclass
class TextSourceConfig:
    paths: List[Union[str, Path]]
    # How to interpret each file
    # - "lines": each line is a document
    # - "whole": whole file is a single document
    # - "jsonl": each line is a JSON object, use json_field as the text
    mode: str = "lines"
    json_field: str = "text"
    encoding: str = "utf-8"


@dataclass
class ChunkConfig:
    seq_len: int
    stride: Optional[int] = None  # if None, no overlap (stride = seq_len)
    drop_remainder: bool = True   # drop final short chunk
    add_special_tokens: bool = True
    # If True, we treat each document independently.
    # If False, we concatenate all docs into one long stream.
    document_separated: bool = True


# Text sources

class TextSource:
    """
    Streams raw text documents from disk.
    """

    def __init__(self, config: TextSourceConfig):
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


# Chunking utilities

def _yield_chunks_from_ids(
    input_ids: List[int],
    chunk_cfg: ChunkConfig,
) -> Iterator[List[int]]:
    seq_len = chunk_cfg.seq_len
    stride = chunk_cfg.stride or seq_len

    if len(input_ids) == 0:
        return

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
    """

    def __init__(
        self,
        text_source: TextSource,
        tokenizer: PreTrainedTokenizerBase,
        chunk_cfg: ChunkConfig,
        max_docs: Optional[int] = None,
    ):
        super().__init__()
        self.text_source = text_source
        self.tokenizer = tokenizer
        self.chunk_cfg = chunk_cfg
        self.max_docs = max_docs

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        tokenizer = self.tokenizer
        chunk_cfg = self.chunk_cfg

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


# Collate function

def fixed_length_collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for batches where all sequences are already fixed length.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def dynamic_pad_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    If you later support variable-length chunks, this pads to max length in batch.
    """
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

    input_ids = torch.stack(padded_ids, dim=0)
    attention_mask = torch.stack(padded_masks, dim=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}