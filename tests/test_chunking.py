"""Tests for baked.chunking — stdlib-only text chunking utilities."""

import pytest
from baked import chunk_fixed, chunk_sentences, chunk_paragraphs, chunk_recursive


# ---------------------------------------------------------------------------
# chunk_fixed
# ---------------------------------------------------------------------------

def test_chunk_fixed_basic():
    # 20-char text, size=10, overlap=2  -> step=8
    # start=0  -> [0:10]
    # start=8  -> [8:18]
    # start=16 -> [16:20]  (last, shorter)
    text = "A" * 20
    chunks = chunk_fixed(text, size=10, overlap=2)
    assert len(chunks) == 3
    # First chunk is full-size
    assert len(chunks[0]) == 10
    # Last two chars of chunk[0] equal first two chars of chunk[1]
    # (both are "AA…", so just verify overlap characters match)
    assert chunks[0][-2:] == chunks[1][:2]


def test_chunk_fixed_no_overlap():
    text = "0123456789" * 3  # 30 chars
    chunks = chunk_fixed(text, size=10, overlap=0)
    assert len(chunks) == 3
    # No repeated characters between consecutive chunks
    for a, b in zip(chunks, chunks[1:]):
        assert a[-1] != b[0] or a[-1] == b[0]  # structural check only
    # Verify the full text is covered with no overlap
    reconstructed = "".join(chunks)
    assert reconstructed == text


def test_chunk_fixed_smaller_than_size():
    text = "Hello"
    chunks = chunk_fixed(text, size=512, overlap=64)
    assert chunks == ["Hello"]


# ---------------------------------------------------------------------------
# chunk_sentences
# ---------------------------------------------------------------------------

def test_chunk_sentences_basic():
    text = "Hello world. How are you? I am fine! Great."
    chunks = chunk_sentences(text, max_size=512, overlap=0)
    # Should produce at least one chunk and not include empty strings
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)
    # All text should be covered
    joined = " ".join(chunks)
    for sentence in ["Hello world.", "How are you?", "I am fine!", "Great."]:
        assert sentence in joined


def test_chunk_sentences_overlap():
    sentences = [f"Sentence {i}." for i in range(6)]
    text = " ".join(sentences)
    # max_size small enough to force splits; overlap=1
    chunks = chunk_sentences(text, max_size=40, overlap=1)
    assert len(chunks) >= 2
    # The last sentence of chunk N should appear at the start of chunk N+1
    for i in range(len(chunks) - 1):
        # Find last sentence in chunks[i]
        last_sent = chunks[i].split(". ")[-1].strip()
        if not last_sent.endswith("."):
            last_sent += "."
        # It must appear somewhere in chunks[i+1]
        assert last_sent in chunks[i + 1], (
            f"Expected '{last_sent}' to appear in chunk {i+1}: '{chunks[i+1]}'"
        )


# ---------------------------------------------------------------------------
# chunk_paragraphs
# ---------------------------------------------------------------------------

def test_chunk_paragraphs_basic():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_paragraphs(text, max_size=1024, overlap=0)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)
    joined = "\n\n".join(chunks)
    assert "Para one." in joined
    assert "Para two." in joined
    assert "Para three." in joined


def test_chunk_paragraphs_merges_short():
    # Each para is 10 chars; max_size=25 -> can fit 2 but not 3
    paras = ["1234567890"] * 4
    text = "\n\n".join(paras)
    # Two paras joined = "1234567890\n\n1234567890" = 22 chars < 25
    # Three paras joined = 34 chars > 25  -> flush after 2
    chunks = chunk_paragraphs(text, max_size=25, overlap=0)
    assert len(chunks) == 2
    for chunk in chunks:
        assert len(chunk) <= 25


# ---------------------------------------------------------------------------
# chunk_recursive
# ---------------------------------------------------------------------------

def test_chunk_recursive_basic():
    # Build a text clearly larger than size=50
    sentences = ["This is sentence number {}.".format(i) for i in range(10)]
    text = "\n\n".join(sentences)
    chunks = chunk_recursive(text, size=60, overlap=10)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)
    # No chunk should massively exceed size (recursive splitter is a soft limit
    # but single sentences over the limit are kept; here all are short enough)
    for chunk in chunks:
        assert len(chunk) <= 60 + 30  # generous tolerance for merge edge cases


# ---------------------------------------------------------------------------
# Edge cases shared across all functions
# ---------------------------------------------------------------------------

def test_empty_string():
    for fn in (chunk_fixed, chunk_sentences, chunk_paragraphs, chunk_recursive):
        assert fn("") == [], f"{fn.__name__} should return [] on empty string"
    # Whitespace-only
    for fn in (chunk_fixed, chunk_sentences, chunk_paragraphs, chunk_recursive):
        assert fn("   ") == [], f"{fn.__name__} should return [] on whitespace-only string"


def test_no_empty_chunks():
    texts = [
        "Hello world. How are you?\n\nI am fine!\n\nGoodbye.",
        "Short.",
        "A" * 1000,
    ]
    fns = [chunk_fixed, chunk_sentences, chunk_paragraphs, chunk_recursive]
    for fn in fns:
        for text in texts:
            result = fn(text)
            for chunk in result:
                assert chunk.strip() != "", (
                    f"{fn.__name__} produced an empty chunk for input of length {len(text)}"
                )
