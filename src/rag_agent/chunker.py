"""Recursive, overlap-aware text chunker that tracks character offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ChunkerConfig:
    """Chunking parameters (measured in characters)."""

    chunk_size: int = 500
    chunk_overlap: int = 80
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")


def _split_recursive(text: str, offset: int, separators: tuple[str, ...], size: int) -> list[tuple[int, int]]:
    """Split ``text`` into contiguous ``(start, end)`` spans no longer than ``size``.

    Tries the coarsest separator first and recurses into oversized pieces with
    finer separators; the separator stays attached to the preceding piece so the
    spans tile the text with no gaps.
    """
    if len(text) <= size:
        return [(offset, offset + len(text))] if text else []
    if not separators:  # no separator left: hard split
        return [(offset + i, offset + min(i + size, len(text))) for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    spans: list[tuple[int, int]] = []
    pos = 0
    parts = text.split(sep)
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if piece:
            spans.extend(_split_recursive(piece, offset + pos, rest, size))
        pos += len(piece)
    return spans


def _section_at(headings: list[tuple[int, str]], position: int) -> str:
    """Nearest Markdown heading that starts at or before ``position``."""
    current = ""
    for start, title in headings:
        if start > position:
            break
        current = title
    return current


def chunk_document(doc: Document, config: ChunkerConfig | None = None) -> list[Chunk]:
    """Chunk a document into overlapping pieces.

    Greedily packs recursive-split spans up to ``chunk_size`` characters; when a
    chunk is emitted, the trailing spans totalling at most ``chunk_overlap``
    characters are carried into the next chunk.
    """
    cfg = config or ChunkerConfig()
    text = doc.text
    spans = _split_recursive(text, 0, cfg.separators, cfg.chunk_size)
    headings = [(m.start(), m.group(1)) for m in _HEADING_LINE_RE.finditer(text)]

    chunks: list[Chunk] = []
    window: list[tuple[int, int]] = []
    fresh = False  # does the window contain spans not yet emitted?

    def emit() -> None:
        start, end = window[0][0], window[-1][1]
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            return
        lead = len(raw) - len(raw.lstrip())
        real_start = start + lead
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{idx}",
                doc_id=doc.doc_id,
                text=stripped,
                start=real_start,
                end=real_start + len(stripped),
                index=idx,
                section=_section_at(headings, real_start),
            )
        )

    for span in spans:
        length = span[1] - span[0]
        if window and (window[-1][1] - window[0][0]) + length > cfg.chunk_size:
            if fresh:
                emit()
            # keep an overlap tail that still leaves room for the new span
            while window and (
                (window[-1][1] - window[0][0]) > cfg.chunk_overlap
                or (window[-1][1] - window[0][0]) + length > cfg.chunk_size
            ):
                window.pop(0)
            fresh = False
        window.append(span)
        fresh = True
    if window and fresh:
        emit()
    return chunks


def chunk_documents(docs: list[Document], config: ChunkerConfig | None = None) -> list[Chunk]:
    """Chunk many documents, preserving order."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, config))
    return out
