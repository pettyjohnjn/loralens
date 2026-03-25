# tests/data/test_loader.py

import json
from pathlib import Path

import torch

from loralens.data.loader import (
    TextSourceConfig,
    TextSource,
    ChunkConfig,
    ChunkedTextDataset,
    fixed_length_collate_fn,
)


class FakeTokenizer:
    """
    Minimal tokenizer-like object for tests.

    - Maps each character to an integer id (ord % 1000) just to have stable ids.
    - Treats the whole string as a single sequence.
    - Ignores add_special_tokens, etc.
    """

    def __call__(
        self,
        text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    ):
        # Simple deterministic mapping: one "token" per character
        input_ids = [ord(c) % 1000 for c in text]
        return {"input_ids": input_ids}


# TextSource tests


def test_text_source_lines_mode_reads_nonempty_lines(tmp_path):
    file_path = tmp_path / "lines.txt"
    file_path.write_text("first line\n\nsecond line  \n  \nthird\n", encoding="utf-8")

    cfg = TextSourceConfig(paths=[file_path], mode="lines")
    source = TextSource(cfg)

    docs = list(source)
    assert docs == ["first line", "second line", "third"]
    # ensure no empty/whitespace-only lines are yielded
    assert all(doc.strip() for doc in docs)


def test_text_source_whole_mode_reads_whole_file(tmp_path):
    file_path = tmp_path / "whole.txt"
    content = "some text\nwith newlines\n\nand more."
    file_path.write_text(content, encoding="utf-8")

    cfg = TextSourceConfig(paths=[file_path], mode="whole")
    source = TextSource(cfg)

    docs = list(source)
    assert len(docs) == 1
    assert docs[0] == content.strip()


def test_text_source_jsonl_mode_reads_field(tmp_path):
    file_path = tmp_path / "data.jsonl"
    records = [
        {"text": "first"},
        {"text": "second", "extra": 1},
        {"other": "ignored"},
        {"text": "third"},
    ]
    with file_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    cfg = TextSourceConfig(paths=[file_path], mode="jsonl", json_field="text")
    source = TextSource(cfg)

    docs = list(source)
    # "other" field record should be skipped
    assert docs == ["first", "second", "third"]


# ChunkedTextDataset tests


def _make_dataset_from_strings(
    texts,
    seq_len,
    stride=None,
    drop_remainder=True,
    max_docs=None,
):
    # Write texts to tmp file via TextSourceConfig? For simplicity, use a small helper
    class InMemoryTextSource(TextSource):
        def __init__(self, docs):
            # We bypass the file-based logic by overriding __iter__
            self._docs = docs

        def __iter__(self):
            return iter(self._docs)

    chunk_cfg = ChunkConfig(
        seq_len=seq_len,
        stride=stride,
        drop_remainder=drop_remainder,
        add_special_tokens=False,
        document_separated=True,
    )
    tokenizer = FakeTokenizer()
    source = InMemoryTextSource(texts)
    dataset = ChunkedTextDataset(
        text_source=source,
        tokenizer=tokenizer,
        chunk_cfg=chunk_cfg,
        max_docs=max_docs,
    )
    return dataset


def test_chunked_text_dataset_fixed_length_chunks_and_stride():
    # Text with 10 "tokens" (characters)
    texts = ["abcdefghij"]  # len = 10
    seq_len = 4
    stride = 2

    dataset = _make_dataset_from_strings(
        texts=texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=True,
    )

    chunks = list(dataset)
    # Expected token positions (0-based): [0..3], [2..5], [4..7]
    # Last possible start 6 => chunk [6..9] length 4, so also included
    # Actually for len=10, seq_len=4, stride=2:
    # start=0 -> [0,1,2,3]
    # start=2 -> [2,3,4,5]
    # start=4 -> [4,5,6,7]
    # start=6 -> [6,7,8,9]
    # start=8 -> [8,9] (short, dropped)
    assert len(chunks) == 4

    for item in chunks:
        ids = item["input_ids"]
        mask = item["attention_mask"]
        assert ids.shape == (seq_len,)
        assert mask.shape == (seq_len,)
        assert torch.all(mask == 1)


def test_chunked_text_dataset_drop_remainder_false_includes_last_short_chunk():
    # 7 chars => 7 tokens; seq_len=4; stride=4; last chunk is length 3
    texts = ["ABCDEFG"]
    seq_len = 4
    stride = 4

    dataset = _make_dataset_from_strings(
        texts=texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=False,
    )

    chunks = list(dataset)
    # start=0 -> [0,1,2,3] length 4
    # start=4 -> [4,5,6] length 3 (kept when drop_remainder=False)
    assert len(chunks) == 2
    assert chunks[0]["input_ids"].shape[0] == 4
    assert chunks[1]["input_ids"].shape[0] == 3  # not padded by dataset


def test_chunked_text_dataset_max_docs_limits_documents():
    texts = ["doc1", "doc2", "doc3"]
    seq_len = 2
    stride = 2

    dataset = _make_dataset_from_strings(
        texts=texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=True,
        max_docs=2,
    )

    # Each doc has len=4 tokens; with seq_len=2, stride=2 => 2 chunks per doc
    chunks = list(dataset)
    # Only first 2 docs should be used
    assert len(chunks) == 4


# Collate function tests


def test_fixed_length_collate_fn_stacks_batch_correctly():
    # Create a fake batch of 3 sequences of length 5
    batch = []
    for i in range(3):
        ids = torch.full((5,), i, dtype=torch.long)
        mask = torch.ones(5, dtype=torch.long)
        batch.append({"input_ids": ids, "attention_mask": mask})

    out = fixed_length_collate_fn(batch)
    assert set(out.keys()) == {"input_ids", "attention_mask"}

    assert out["input_ids"].shape == (3, 5)
    assert out["attention_mask"].shape == (3, 5)

    # Check that values are preserved
    assert torch.all(out["input_ids"][0] == 0)
    assert torch.all(out["input_ids"][1] == 1)
    assert torch.all(out["input_ids"][2] == 2)