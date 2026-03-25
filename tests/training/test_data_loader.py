# tests/training/test_data_loader.py
"""Tests for iterable dataset sharding behavior."""

import torch

from loralens.data.loader import ChunkConfig, ChunkedTextDataset


class DummyTokenizer:
    """Tokenizer stub that maps each document to a fixed token sequence."""

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    ):
        del add_special_tokens, return_attention_mask, return_token_type_ids
        token = int(text)
        return {"input_ids": [token, token + 100]}


def _collect_first_tokens(dataset: ChunkedTextDataset) -> list[int]:
    return [int(sample["input_ids"][0]) for sample in dataset]


def test_document_sharding_splits_stream_across_ranks():
    dataset_rank0 = ChunkedTextDataset(
        text_source=["0", "1", "2", "3"],
        tokenizer=DummyTokenizer(),
        chunk_cfg=ChunkConfig(seq_len=2),
        rank=0,
        world_size=2,
    )
    dataset_rank1 = ChunkedTextDataset(
        text_source=["0", "1", "2", "3"],
        tokenizer=DummyTokenizer(),
        chunk_cfg=ChunkConfig(seq_len=2),
        rank=1,
        world_size=2,
    )

    assert _collect_first_tokens(dataset_rank0) == [0, 2]
    assert _collect_first_tokens(dataset_rank1) == [1, 3]


def test_concatenated_sharding_respects_rank_assignment():
    dataset_rank0 = ChunkedTextDataset(
        text_source=["0", "1", "2", "3"],
        tokenizer=DummyTokenizer(),
        chunk_cfg=ChunkConfig(seq_len=4, document_separated=False),
        rank=0,
        world_size=2,
    )
    dataset_rank1 = ChunkedTextDataset(
        text_source=["0", "1", "2", "3"],
        tokenizer=DummyTokenizer(),
        chunk_cfg=ChunkConfig(seq_len=4, document_separated=False),
        rank=1,
        world_size=2,
    )

    rank0_chunks = list(dataset_rank0)
    rank1_chunks = list(dataset_rank1)

    assert len(rank0_chunks) == 1
    assert len(rank1_chunks) == 1
    assert torch.equal(rank0_chunks[0]["input_ids"], torch.tensor([0, 100, 2, 102]))
    assert torch.equal(rank1_chunks[0]["input_ids"], torch.tensor([1, 101, 3, 103]))


def test_invalid_rank_configuration_raises():
    try:
        ChunkedTextDataset(
            text_source=["0"],
            tokenizer=DummyTokenizer(),
            chunk_cfg=ChunkConfig(seq_len=2),
            rank=1,
            world_size=1,
        )
    except ValueError as exc:
        assert "rank must be in [0, world_size)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid rank/world_size")
