"""End-to-end RAG pipeline: ingest -> chunk -> index -> retrieve -> rerank -> generate."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .chunker import ChunkerConfig, chunk_documents
from .hybrid import HybridRetriever
from .llm import LLM, MockLLM
from .loader import load_directory
from .rerank import NoOpReranker, Reranker
from .retrievers import BM25Retriever, Retriever, TfidfRetriever
from .types import Answer, Chunk, Citation, Document, SearchHit

MODES = ("bm25", "tfidf", "hybrid")
DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"
DEFAULT_EVAL_SET = Path(__file__).resolve().parents[2] / "data" / "eval_set.json"


class RAGPipeline:
    """Orchestrates retrieval and generation over an in-memory corpus."""

    def __init__(
        self,
        chunker: ChunkerConfig | None = None,
        llm: LLM | None = None,
        reranker: Reranker | None = None,
        candidates: int = 20,
    ) -> None:
        self.chunker = chunker or ChunkerConfig()
        self.llm = llm or MockLLM()
        self.reranker = reranker or NoOpReranker()
        self.candidates = candidates
        self._docs: dict[str, Document] = {}
        self._chunks: list[Chunk] = []
        self._bm25 = BM25Retriever()
        self._tfidf = TfidfRetriever()
        self._hybrid = HybridRetriever([self._bm25, self._tfidf], candidates=candidates)

    # -- ingestion -------------------------------------------------------
    def ingest(self, documents: Iterable[Document]) -> int:
        """Add or replace documents (by ``doc_id``) and rebuild all indexes. Returns chunk count."""
        for doc in documents:
            self._docs[doc.doc_id] = doc
        self._chunks = chunk_documents(list(self._docs.values()), self.chunker)
        for retriever in (self._bm25, self._tfidf):
            retriever.index(self._chunks)
        return len(self._chunks)

    def ingest_directory(self, directory: str | Path = DEFAULT_CORPUS) -> int:
        return self.ingest(load_directory(directory))

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def num_documents(self) -> int:
        return len(self._docs)

    # -- retrieval -------------------------------------------------------
    def _retriever(self, mode: str) -> Retriever:
        if mode == "bm25":
            return self._bm25
        if mode == "tfidf":
            return self._tfidf
        if mode == "hybrid":
            return self._hybrid
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    def retrieve(self, query: str, k: int = 4, mode: str = "hybrid", rerank: bool = True) -> list[SearchHit]:
        """First-stage retrieval of ``candidates`` chunks, then re-ranking down to ``k``."""
        retriever = self._retriever(mode)
        first_stage = retriever.search(query, max(self.candidates, k))
        return self.reranker.rerank(query, first_stage, k) if rerank else first_stage[:k]

    # -- generation ------------------------------------------------------
    def query(self, question: str, k: int = 4, mode: str = "hybrid") -> Answer:
        """Answer ``question`` from the top-``k`` chunks, with a citation per chunk."""
        hits = self.retrieve(question, k=k, mode=mode)
        text = self.llm.generate(question, hits)
        citations = tuple(
            Citation(
                marker=i,
                source=h.chunk.doc_id,
                chunk_id=h.chunk.chunk_id,
                section=h.chunk.section,
                score=h.score,
                snippet=h.chunk.text[:160],
            )
            for i, h in enumerate(hits, start=1)
        )
        return Answer(question=question, text=text, citations=citations)
