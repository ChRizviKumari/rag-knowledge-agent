"""Flask API exposing ``/ingest``, ``/query`` and ``/health``."""

from __future__ import annotations

from typing import Literal

from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, ValidationError

from .pipeline import DEFAULT_CORPUS, RAGPipeline
from .types import Document


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    title: str | None = None


class IngestRequest(BaseModel):
    """Either inline ``documents`` or ``sample_corpus: true`` (a server-side path is never accepted)."""

    documents: list[DocumentIn] = Field(default_factory=list)
    sample_corpus: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    mode: Literal["bm25", "tfidf", "hybrid"] = "hybrid"


def create_app(pipeline: RAGPipeline | None = None) -> Flask:
    """Application factory; pass a pipeline to inject dependencies (used by tests)."""
    app = Flask(__name__)
    rag = pipeline or RAGPipeline()

    def bad_request(exc: ValidationError):
        return jsonify({"error": "validation_error", "details": exc.errors(include_url=False, include_context=False)}), 422

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "documents": rag.num_documents, "chunks": len(rag.chunks)})

    @app.post("/ingest")
    def ingest():
        try:
            body = IngestRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            return bad_request(exc)
        if not body.documents and not body.sample_corpus:
            return jsonify({"error": "provide 'documents' or 'sample_corpus': true"}), 400
        if body.sample_corpus:
            rag.ingest_directory(DEFAULT_CORPUS)
        if body.documents:
            rag.ingest([Document(d.id, d.title or d.id, d.text, source="api") for d in body.documents])
        return jsonify({"documents": rag.num_documents, "chunks": len(rag.chunks)}), 201

    @app.post("/query")
    def query():
        try:
            body = QueryRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            return bad_request(exc)
        if rag.num_documents == 0:
            return jsonify({"error": "no documents ingested; call /ingest first"}), 409
        return jsonify(rag.query(body.question, k=body.k, mode=body.mode).to_dict())

    return app
