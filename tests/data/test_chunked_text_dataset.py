# tests/data/test_chunked_text_dataset.py

import torch
from torch.utils.data import IterableDataset

from loralens.data.loader import ChunkConfig, ChunkedTextDataset


class DummyTextSource:
    """Minimal TextSource-like stub yielding in-memory strings."""

    def __init__(self, docs):
        self._docs = list(docs)

    def __iter__(self):
        yield from self._docs


class FakeTokenizer:
    """
    Minimal tokenizer stub.

    Maps each character to ord(c) % 1000 and returns a dict with "input_ids".
    This is enough to drive ChunkedTextDataset without depending on HF.
    """

    def __call__(
        self,
        text,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    ):
        ids = [ord(c) % 1000 for c in text]
        return {"input_ids": ids}


def _make_dataset_from_strings(
    texts,
    *,
    seq_len,
    stride,
    drop_remainder=True,
    document_separated=True,
    max_docs=None,
):
    text_source = DummyTextSource(texts)
    tokenizer = FakeTokenizer()
    chunk_cfg = ChunkConfig(
        seq_len=seq_len,
        stride=stride,
        drop_remainder=drop_remainder,
        add_special_tokens=False,
        document_separated=document_separated,
    )
    return ChunkedTextDataset(
        text_source=text_source,
        tokenizer=tokenizer,
        chunk_cfg=chunk_cfg,
        max_docs=max_docs,
    )


def test_doc_separated_simple_chunks_and_stride():
    """
    When document_separated=True, each document is chunked independently
    into non-overlapping seq_len chunks.
    """
    texts = ["abcd", "efgh"]  # 4 chars each
    seq_len = 2
    stride = 2

    dataset = _make_dataset_from_strings(
        texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=True,
        document_separated=True,
    )

    chunks = list(dataset)
    # Each doc length 4, seq_len=2, stride=2 -> 2 chunks per doc -> 4 total
    assert len(chunks) == 4

    # Check the actual IDs correspond to the characters
    # FakeTokenizer: input_ids = [ord(c) % 1000]
    ids_chunks = [c["input_ids"].tolist() for c in chunks]
    expected = [
        [ord("a") % 1000, ord("b") % 1000],
        [ord("c") % 1000, ord("d") % 1000],
        [ord("e") % 1000, ord("f") % 1000],
        [ord("g") % 1000, ord("h") % 1000],
    ]
    assert ids_chunks == expected


def test_concatenated_across_docs_no_overlap():
    """
    When document_separated=False, documents are concatenated into a single
    token stream and then split into fixed-length chunks.
    """
    texts = ["abcd", "ef"]  # concatenated token stream 'abcdef'
    seq_len = 3
    stride = 3  # no overlap

    dataset = _make_dataset_from_strings(
        texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=True,
        document_separated=False,
    )

    chunks = list(dataset)
    # 'abcdef' -> [abc], [def]
    assert len(chunks) == 2

    ids_chunks = [c["input_ids"].tolist() for c in chunks]
    expected_chars = ["abc", "def"]
    expected = [
        [ord(c) % 1000 for c in s] for s in expected_chars
    ]
    assert ids_chunks == expected


def test_concatenated_with_overlap_stride_1():
    """
    Concatenated mode with stride < seq_len should produce overlapping windows.
    """
    texts = ["abcd"]  # token stream 'abcd'
    seq_len = 3
    stride = 1  # sliding window

    dataset = _make_dataset_from_strings(
        texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=True,
        document_separated=False,
    )

    chunks = list(dataset)
    # 'abcd' with seq_len=3, stride=1 -> 'abc', 'bcd'
    assert len(chunks) == 2

    ids_chunks = [c["input_ids"].tolist() for c in chunks]
    expected_chars = ["abc", "bcd"]
    expected = [
        [ord(c) % 1000 for c in s] for s in expected_chars
    ]
    assert ids_chunks == expected


def test_concatenated_drop_remainder_true_drops_tail():
    """
    With drop_remainder=True, any leftover tokens shorter than seq_len
    at the end of the stream are dropped.
    """
    texts = ["abcde"]  # 5 tokens
    seq_len = 3
    stride = 3

    dataset = _make_dataset_from_strings(
        texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=True,
        document_separated=False,
    )

    chunks = list(dataset)
    # 'abcde' -> one full chunk 'abc', tail 'de' is dropped
    assert len(chunks) == 1
    ids = chunks[0]["input_ids"].tolist()
    expected = [ord(c) % 1000 for c in "abc"]
    assert ids == expected


def test_concatenated_drop_remainder_false_keeps_tail():
    """
    With drop_remainder=False, a final shorter chunk is emitted for the tail.
    """
    texts = ["abcde"]  # 5 tokens
    seq_len = 3
    stride = 3

    dataset = _make_dataset_from_strings(
        texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=False,
        document_separated=False,
    )

    chunks = list(dataset)
    # 'abcde' -> 'abc', 'de'
    assert len(chunks) == 2

    ids_chunks = [c["input_ids"].tolist() for c in chunks]
    expected_chars = ["abc", "de"]
    expected = [
        [ord(c) % 1000 for c in s] for s in expected_chars
    ]
    assert ids_chunks == expected


def test_chunked_text_dataset_max_docs_limits_documents():
    """
    max_docs should limit how many documents are *ever* read from the source,
    even in concatenated mode; later documents' tokens must never appear.
    """
    texts = ["doc1", "doc2", "doc3"]
    seq_len = 2
    stride = 2

    dataset = _make_dataset_from_strings(
        texts=texts,
        seq_len=seq_len,
        stride=stride,
        drop_remainder=True,
        document_separated=False,
        max_docs=2,
    )

    chunks = list(dataset)
    # We only ever read the first 2 docs; exact number of chunks is not
    # the main thing here, but for sanity:
    assert len(chunks) > 0

    # Ensure tokens from 'doc3' never appear
    all_ids_flat = torch.cat([c["input_ids"] for c in chunks]).tolist()
    doc3_ids = [ord(c) % 1000 for c in "doc3"]
    for tid in doc3_ids:
        assert tid not in all_ids_flat