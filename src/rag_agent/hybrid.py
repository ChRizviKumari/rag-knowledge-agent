"""Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections.abc import Sequence

from .retrievers import Retriever
from .types import Chunk, SearchHit


def reciprocal_rank_fusion(rankings: Sequence[Sequence[SearchHit]], k: int = 60, top_n: int | None = None) -> list[SearchHit]:
    """Fuse ranked lists: ``score(d) = sum_over_lists 1 / (k + rank_in_list(d))``.

    RRF uses ranks only, so it needs no score normalisation between retrievers
    whose scores live on different scales (BM25 vs cosine). Ties are broken by
    first appearance, keeping results deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + position)
            chunks.setdefault(cid, hit.chunk)
    order = {cid: i for i, cid in enumerate(chunks)}
    ranked = sorted(fused, key=lambda cid: (-fused[cid], order[cid]))
    if top_n is not None:
        ranked = ranked[:top_n]
    return [SearchHit(chunk=chunks[cid], score=fused[cid], rank=i + 1) for i, cid in enumerate(ranked)]


class HybridRetriever(Retriever):
    """Runs several retrievers and fuses their rankings with RRF."""

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, candidates: int = 20) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.candidates = candidates

    def index(self, chunks: Sequence[Chunk]) -> None:
        for retriever in self.retrievers:
            retriever.index(chunks)

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        pool = max(self.candidates, k)
        rankings = [r.search(query, pool) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k, top_n=k)
