"""rag_agent: a retrieval-augmented generation system built from scratch."""

from .chunker import ChunkerConfig, chunk_document
from .hybrid import HybridRetriever, reciprocal_rank_fusion
from .llm import LLM, MockLLM
from .pipeline import RAGPipeline
from .retrievers import BM25Retriever, TfidfRetriever
from .types import Answer, Chunk, Citation, Document, SearchHit

__all__ = [
    "Answer", "BM25Retriever", "Chunk", "ChunkerConfig", "Citation", "Document", "HybridRetriever",
    "LLM", "MockLLM", "RAGPipeline", "SearchHit", "TfidfRetriever", "chunk_document",
    "reciprocal_rank_fusion",
]
__version__ = "0.1.0"
