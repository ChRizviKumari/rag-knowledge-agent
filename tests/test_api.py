import pytest

from rag_agent.api import create_app
from rag_agent.pipeline import RAGPipeline


@pytest.fixture
def client():
    return create_app(RAGPipeline()).test_client()


def test_health_empty(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "ok", "documents": 0, "chunks": 0}


def test_query_before_ingest_conflicts(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 409


def test_ingest_inline_documents_then_query(client):
    docs = [{"id": "zebra", "text": "# Zebras\n\nZebras have black and white stripes that confuse predators."}]
    r = client.post("/ingest", json={"documents": docs})
    assert r.status_code == 201 and r.get_json()["documents"] == 1
    q = client.post("/query", json={"question": "What confuses predators?", "k": 2})
    body = q.get_json()
    assert q.status_code == 200 and "stripes" in body["answer"]
    assert body["citations"][0]["source"] == "zebra" and body["citations"][0]["chunk_id"] == "zebra#0"


def test_ingest_sample_corpus_and_query_modes(client):
    r = client.post("/ingest", json={"sample_corpus": True})
    assert r.get_json()["documents"] == 8
    for mode in ("bm25", "tfidf", "hybrid"):
        body = client.post("/query", json={"question": "How does a circuit breaker work?", "mode": mode}).get_json()
        assert body["citations"][0]["source"] == "resilience_patterns"


def test_ingest_requires_payload(client):
    assert client.post("/ingest", json={}).status_code == 400


def test_ingest_rejects_path_and_bad_documents(client):
    assert client.post("/ingest", json={"documents": [{"id": "", "text": "x"}]}).status_code == 422
    assert client.post("/ingest", json={"path": "/etc"}).status_code == 400  # unknown keys never read the filesystem


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "q", "k": 0}, {"question": "q", "mode": "x"}])
def test_query_validation(client, payload):
    client.post("/ingest", json={"sample_corpus": True})
    r = client.post("/query", json=payload)
    assert r.status_code == 422 and r.get_json()["error"] == "validation_error"


def test_health_after_ingest(client):
    client.post("/ingest", json={"sample_corpus": True})
    body = client.get("/health").get_json()
    assert body["documents"] == 8 and body["chunks"] > 8
