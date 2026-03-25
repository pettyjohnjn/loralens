from __future__ import annotations

from typing import Iterable, Iterator, Literal, Optional, Tuple

from torch.utils.data import DataLoader

from loralens.data.hf_adapter import StreamingTextSource
from loralens.data.hf_pile import PileSource, PileSourceConfig
from loralens.data.loader import (
    ChunkConfig,
    ChunkedTextDataset,
    TextSource,
    TextSourceConfig,
    fixed_length_collate_fn,
)

from .ddp import DDPState


DataSource = Literal["text", "pile"]


class ShardedIterable(Iterable[str]):
    def __init__(self, base: Iterable[str], *, rank: int, world_size: int, start_offset: int = 0):
        self.base = base
        self.rank = rank
        self.world_size = world_size
        self.start_offset = start_offset

    def __iter__(self) -> Iterator[str]:
        for i, doc in enumerate(self.base):
            if ((i + self.start_offset) % self.world_size) == self.rank:
                yield doc


def build_text_source(
    *,
    data_source: DataSource,
    ddp_state: DDPState,
    seed: int,
    text_paths,
    text_mode: str,
    text_json_field: str,
    pile_split: str,
):
    if data_source == "text":
        if not text_paths:
            raise ValueError("data_source='text' but no text_paths were provided.")
        cfg = TextSourceConfig(paths=text_paths, mode=text_mode, json_field=text_json_field)
        base = TextSource(cfg)
    elif data_source == "pile":
        pile_cfg = PileSourceConfig(split=pile_split)
        pile_source = PileSource(pile_cfg)
        base = StreamingTextSource(iterable=pile_source)
    else:
        raise ValueError(f"Unknown data_source={data_source!r}")

    if ddp_state.enabled and ddp_state.world_size > 1:
        base = ShardedIterable(base, rank=ddp_state.rank, world_size=ddp_state.world_size, start_offset=seed)

    return base


def build_dataloader(
    *,
    tokenizer,
    ddp_state: DDPState,
    seed: int,
    data_source: DataSource,
    text_paths,
    text_mode: str,
    text_json_field: str,
    pile_split: str,
    max_seq_len: int,
    stride: Optional[int],
    drop_remainder: bool,
    document_separated: bool,
    max_docs: Optional[int],
    per_gpu_batch_size: int,
) -> DataLoader:
    text_source = build_text_source(
        data_source=data_source,
        ddp_state=ddp_state,
        seed=seed,
        text_paths=text_paths,
        text_mode=text_mode,
        text_json_field=text_json_field,
        pile_split=pile_split,
    )

    stride_ = stride if stride is not None else max_seq_len

    chunk_cfg = ChunkConfig(
        seq_len=max_seq_len,
        stride=stride_,
        drop_remainder=drop_remainder,
        add_special_tokens=True,
        document_separated=document_separated,
    )

    dataset = ChunkedTextDataset(
        text_source=text_source,
        tokenizer=tokenizer,
        chunk_cfg=chunk_cfg,
        max_docs=max_docs,
    )

    return DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        collate_fn=fixed_length_collate_fn,
    )


def next_batch(data_iter, make_iter_fn) -> Tuple[dict, object]:
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = make_iter_fn()
        return next(data_iter), data_iter