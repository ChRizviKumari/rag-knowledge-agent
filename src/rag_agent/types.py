"""Core immutable data types shared across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document prior to chunking."""

    doc_id: str
    title: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document.

    ``start``/``end`` are character offsets into the original document text, so
    ``document.text[start:end].strip() == chunk.text`` always holds.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    index: int
    section: str = ""


@dataclass(frozen=True)
class SearchHit:
    """A chunk with its retrieval score and 1-based rank within a result list."""

    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class Citation:
    """Provenance of a piece of evidence used for an answer."""

    marker: int
    source: str
    chunk_id: str
    section: str
    score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "score": round(self.score, 6),
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class Answer:
    """Final response of the pipeline: generated text plus its citations."""

    question: str
    text: str
    citations: tuple[Citation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "citations": [c.to_dict() for c in self.citations],
        }
