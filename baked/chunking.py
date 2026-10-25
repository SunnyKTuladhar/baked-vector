"""Text chunking utilities — stdlib only (re, math). No external dependencies."""

from __future__ import annotations

import re


def chunk_fixed(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into fixed-size character chunks with overlap.

    Slides a window of *size* chars over *text* with a step of ``size - overlap``.
    Each chunk is stripped; empty chunks are dropped.

    Args:
        text: Input text to split.
        size: Maximum characters per chunk.
        overlap: Number of characters repeated at the start of the next chunk.

    Returns:
        List of non-empty stripped chunks.
    """
    if not text or not text.strip():
        return []

    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be less than size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# Sentence-ending punctuation followed by whitespace or end-of-string.
_SENT_RE = re.compile(r"(?<=[.!?])(?:\s+|$)")


def chunk_sentences(text: str, max_size: int = 512, overlap: int = 1) -> list[str]:
    """Split on sentence boundaries (.!?).

    Sentences are detected by ``.``, ``!``, or ``?`` followed by whitespace or
    end-of-string. ``overlap`` is the number of sentences from the end of the
    previous chunk to repeat at the start of the next chunk (not characters).

    Short sentences are merged into a single chunk until *max_size* would be
    exceeded, then the current chunk is flushed.

    Args:
        text: Input text to split.
        max_size: Soft maximum characters per chunk. A single sentence longer
            than *max_size* is kept as its own chunk.
        overlap: Number of trailing sentences carried into the next chunk.

    Returns:
        List of non-empty stripped chunks.
    """
    if not text or not text.strip():
        return []

    # Split and keep the punctuation attached to the preceding sentence.
    raw = _SENT_RE.split(text)
    sentences = [s.strip() for s in raw if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def _flush() -> None:
        if current:
            joined = " ".join(current)
            if joined.strip():
                chunks.append(joined.strip())

    for sent in sentences:
        # +1 for the space separator
        added_len = len(sent) + (1 if current else 0)
        if current and current_len + added_len > max_size:
            _flush()
            # carry overlap sentences into next chunk, capped to max_size
            tail = current[-overlap:] if overlap else []
            current = list(tail)
            current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
            while current and current_len > max_size and len(current) > 1:
                current.pop(0)
                current_len = sum(len(s) for s in current) + max(0, len(current) - 1)

        current.append(sent)
        current_len += len(sent) + (1 if len(current) > 1 else 0)

    _flush()
    return chunks


def chunk_paragraphs(text: str, max_size: int = 1024, overlap: int = 0) -> list[str]:
    """Split on blank lines, merging short paragraphs up to *max_size*.

    Paragraphs are separated by one or more blank lines (``\\n\\n+``). Consecutive
    short paragraphs are merged (joined with ``\\n\\n``) until adding the next one
    would exceed *max_size*, then the current buffer is flushed. ``overlap=0``
    means no paragraph is repeated; ``overlap=N`` repeats the last N paragraphs
    of each chunk at the start of the next.

    Args:
        text: Input text to split.
        max_size: Soft maximum characters per chunk.
        overlap: Number of trailing paragraphs carried into the next chunk.

    Returns:
        List of non-empty stripped chunks.
    """
    if not text or not text.strip():
        return []

    raw = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in raw if p.strip()]

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep = "\n\n"
    sep_len = len(sep)

    def _flush() -> None:
        if current:
            joined = sep.join(current)
            if joined.strip():
                chunks.append(joined.strip())

    for para in paragraphs:
        added_len = len(para) + (sep_len if current else 0)
        if current and current_len + added_len > max_size:
            _flush()
            # carry overlap paragraphs into next chunk, capped to max_size
            tail = current[-overlap:] if overlap else []
            current = list(tail)
            current_len = sum(len(p) for p in current) + sep_len * max(0, len(current) - 1)
            while current and current_len > max_size and len(current) > 1:
                current.pop(0)
                current_len = sum(len(p) for p in current) + sep_len * max(0, len(current) - 1)

        current.append(para)
        current_len += len(para) + (sep_len if len(current) > 1 else 0)

    _flush()
    return chunks


# Ordered list of separators tried by chunk_recursive.
_RECURSIVE_SEPS = ["\n\n", "\n", ". ", ""]


def chunk_recursive(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """LangChain-style recursive text splitter.

    Tries splitting on ``\\n\\n``, then ``\\n``, then ``". "``, then falls back
    to fixed character splits. Each piece that is still larger than *size* is
    recursively split with the remaining separators.

    Args:
        text: Input text to split.
        size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks on hard cuts.

    Returns:
        List of non-empty stripped chunks.
    """
    if not text or not text.strip():
        return []
    return _recursive_split(text, size, overlap, list(_RECURSIVE_SEPS))


def _recursive_split(
    text: str, size: int, overlap: int, seps: list[str]
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    # Try each separator in order until one produces more than one piece.
    chosen_sep: str | None = None
    splits: list[str] = []
    remaining_seps = list(seps)
    while remaining_seps:
        sep = remaining_seps.pop(0)
        if sep == "":
            # Hard character split — handled after the loop.
            break
        parts = text.split(sep)
        if len(parts) > 1:
            chosen_sep = sep
            splits = [p for p in parts if p.strip()]
            break
    else:
        chosen_sep = None

    if chosen_sep is None:
        # Fall back to fixed character split with overlap.
        return _fixed_split(text, size, overlap)

    # Recursively handle pieces that are still too long.
    next_seps = [s for s in _RECURSIVE_SEPS if s != chosen_sep]
    good: list[str] = []
    for piece in splits:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= size:
            good.append(piece)
        else:
            good.extend(_recursive_split(piece, size, overlap, list(next_seps)))

    # Merge good pieces back up to size.
    return _merge(good, chosen_sep, size, overlap)


def _merge(
    splits: list[str], sep: str, size: int, overlap: int
) -> list[str]:
    """Merge split pieces into chunks up to *size* with *overlap* char carry-over."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(sep)

    for piece in splits:
        piece = piece.strip()
        if not piece:
            continue
        piece_len = len(piece)
        join_len = current_len + sep_len + piece_len if current else piece_len

        if join_len > size and current:
            chunk_text = sep.join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # Build overlap from the tail of current.
            ov: list[str] = []
            ov_len = 0
            for part in reversed(current):
                part_len = len(part)
                candidate = ov_len + (sep_len if ov else 0) + part_len
                if candidate > overlap and ov:
                    break
                ov.insert(0, part)
                ov_len = candidate
            current = ov
            current_len = ov_len
            # Drop overlap from front until overlap + new piece fits.
            while current and current_len + sep_len + piece_len > size:
                current.pop(0)
                current_len = sum(len(p) for p in current) + sep_len * max(0, len(current) - 1)

        current.append(piece)
        current_len = (
            sum(len(p) for p in current) + sep_len * max(0, len(current) - 1)
        )

    if current:
        chunk_text = sep.join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _fixed_split(text: str, size: int, overlap: int) -> list[str]:
    """Hard character split with overlap — last resort."""
    step = max(1, size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
