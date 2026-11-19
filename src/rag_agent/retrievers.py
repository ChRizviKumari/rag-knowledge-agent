"""Lexical retrievers (BM25, TF-IDF) implemented from scratch with numpy."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

import numpy as np

from .tokenizer import tokenize
from .types import Chunk, SearchHit


class Retriever(ABC):
    """Interface every retriever implements."""

    name: str = "retriever"

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        """(Re)build the index from ``chunks``, discarding any previous state."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Return up to ``k`` hits with positive score, best first (ranks start at 1)."""


def _top_k(scores: np.ndarray, chunks: Sequence[Chunk], k: int) -> list[SearchHit]:
    """Deterministic top-k: stable sort so ties keep corpus order; drop zero scores."""
    if k <= 0 or scores.size == 0:
        return []
    order = np.argsort(-scores, kind="stable")[:k]
    hits: list[SearchHit] = []
    for i in order:
        if scores[i] <= 0.0:
            break
        hits.append(SearchHit(chunk=chunks[int(i)], score=float(scores[i]), rank=len(hits) + 1))
    return hits


class BM25Retriever(Retriever):
    """Okapi BM25 with an inverted index and vectorised scoring.

    score(q, d) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    with the non-negative idf variant ``log(1 + (N - df + 0.5) / (df + 0.5))``.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("require k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self._chunks: list[Chunk] = []
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        self._doc_len = np.zeros(0)
        self._avgdl = 0.0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        n = len(self._chunks)
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._doc_len = np.array([len(t) for t in tokenized], dtype=np.float64)
        self._avgdl = float(self._doc_len.mean()) if n else 0.0
        raw: dict[str, tuple[list[int], list[int]]] = {}
        for doc_idx, toks in enumerate(tokenized):
            for term, tf in Counter(toks).items():
                ids, tfs = raw.setdefault(term, ([], []))
                ids.append(doc_idx)
                tfs.append(tf)
        self._postings = {
            t: (np.array(ids, dtype=np.int64), np.array(tfs, dtype=np.float64)) for t, (ids, tfs) in raw.items()
        }
        self._idf = {t: math.log(1.0 + (n - len(ids) + 0.5) / (len(ids) + 0.5)) for t, (ids, _) in raw.items()}

    def idf(self, term: str) -> float:
        """Inverse document frequency of an (already tokenized) term; 0 if unseen."""
        return self._idf.get(term, 0.0)

    def score_all(self, query: str) -> np.ndarray:
        """BM25 score of every indexed chunk for ``query``."""
        scores = np.zeros(len(self._chunks), dtype=np.float64)
        if not self._chunks or self._avgdl == 0.0:
            return scores
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            ids, tf = posting
            norm = tf + self.k1 * (1.0 - self.b + self.b * self._doc_len[ids] / self._avgdl)
            scores[ids] += self._idf[term] * tf * (self.k1 + 1.0) / norm
        return scores

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        return _top_k(self.score_all(query), self._chunks, k)


class TfidfRetriever(Retriever):
    """Cosine similarity over L2-normalised sublinear TF-IDF vectors (dense numpy matrix).

    tf weight = 1 + ln(tf); idf = ln((1 + N) / (1 + df)) + 1 (smoothed).
    A dense matrix is fine for thousands of chunks; see README for the trade-off.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vocab: dict[str, int] = {}
        self._idf = np.zeros(0)
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for term, tf in Counter(tokens).items():
            j = self._vocab.get(term)
            if j is not None:
                vec[j] = (1.0 + math.log(tf)) * self._idf[j]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._vocab = {t: i for i, t in enumerate(sorted({t for toks in tokenized for t in toks}))}
        n = len(self._chunks)
        df = np.zeros(len(self._vocab))
        for toks in tokenized:
            for t in set(toks):
                df[self._vocab[t]] += 1
        self._idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        self._matrix = (
            np.vstack([self._vectorize(toks) for toks in tokenized])
            if n
            else np.zeros((0, len(self._vocab)), dtype=np.float32)
        )

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if not self._chunks:
            return []
        q = self._vectorize(tokenize(query))
        return _top_k((self._matrix @ q).astype(np.float64), self._chunks, k)
