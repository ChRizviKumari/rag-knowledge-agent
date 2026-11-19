"""Re-ranker interface and a dependency-free lexical implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .tokenizer import tokenize
from .types import SearchHit


class Reranker(ABC):
    """Reorders first-stage candidates, typically with a more expensive scorer.

    A cross-encoder model can be plugged in by implementing :meth:`rerank`.
    """

    @abstractmethod
    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        """Return the best ``top_k`` hits reordered, with ranks renumbered from 1."""


def _renumber(hits: Sequence[tuple[SearchHit, float]]) -> list[SearchHit]:
    return [SearchHit(chunk=h.chunk, score=s, rank=i + 1) for i, (h, s) in enumerate(hits)]


class NoOpReranker(Reranker):
    """Keeps the first-stage order (the default)."""

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        return _renumber([(h, h.score) for h in hits[:top_k]])


class LexicalOverlapReranker(Reranker):
    """Scores query-term coverage and adjacent-term (bigram) matches.

    ``score = w_uni * unigram_coverage + w_bi * bigram_coverage + w_prior / first_stage_rank``
    It rewards chunks that contain the *whole* question rather than one rare
    term, and the prior keeps the first-stage ordering as a tie-breaker.
    """

    def __init__(self, w_uni: float = 0.5, w_bi: float = 0.3, w_prior: float = 0.2) -> None:
        self.w_uni, self.w_bi, self.w_prior = w_uni, w_bi, w_prior

    def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        q_tokens = tokenize(query)
        q_uni = set(q_tokens)
        q_bi = set(zip(q_tokens, q_tokens[1:]))
        scored: list[tuple[SearchHit, float]] = []
        for hit in hits:
            toks = tokenize(hit.chunk.text)
            uni = len(q_uni & set(toks)) / len(q_uni) if q_uni else 0.0
            bi = len(q_bi & set(zip(toks, toks[1:]))) / len(q_bi) if q_bi else 0.0
            score = self.w_uni * uni + self.w_bi * bi + self.w_prior / hit.rank
            scored.append((hit, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].rank))
        return _renumber(scored[:top_k])
