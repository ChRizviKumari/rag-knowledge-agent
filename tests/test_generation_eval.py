import pytest

from rag_agent.evaluation import EvalItem, evaluate, is_relevant, load_eval_set, reciprocal_rank
from rag_agent.llm import NO_ANSWER, AnthropicLLM, MockLLM, build_prompt, get_llm
from rag_agent.pipeline import DEFAULT_EVAL_SET, RAGPipeline
from rag_agent.types import Chunk, SearchHit


def hit(cid: str, text: str, rank: int = 1) -> SearchHit:
    doc = cid.split("#")[0]
    return SearchHit(Chunk(cid, doc, text, 0, len(text), 0), 1.0, rank)


def test_mock_llm_is_extractive_and_cites():
    ctx = [hit("a#0", "Raft elects a leader using randomized timeouts. Unrelated sentence about pizza.")]
    out = MockLLM().generate("How does Raft elect a leader?", ctx)
    assert "randomized timeouts" in out and out.endswith("[1]") and "pizza" not in out


def test_mock_llm_is_deterministic_and_abstains():
    ctx = [hit("a#0", "Completely different topic about gardening and soil.")]
    llm = MockLLM()
    assert llm.generate("quantum chromodynamics?", ctx) == NO_ANSWER
    assert llm.generate("gardening soil", ctx) == llm.generate("gardening soil", ctx)
    assert llm.generate("anything", []) == NO_ANSWER


def test_build_prompt_contains_ids_and_question():
    p = build_prompt("Why?", [hit("doc#3", "some evidence text")])
    assert "[1] (doc, doc#3)" in p and "Question: Why?" in p


def test_llm_factory(monkeypatch):
    assert isinstance(get_llm("mock"), MockLLM)
    with pytest.raises(ValueError):
        get_llm("nope")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()


def test_pipeline_citations_have_source_and_chunk_ids(sample_pipeline):
    ans = sample_pipeline.query("How does a token bucket permit bursts?", k=3)
    assert len(ans.citations) == 3
    top = ans.citations[0]
    assert top.source == "rate_limiting" and top.chunk_id.startswith("rate_limiting#")
    assert [c.marker for c in ans.citations] == [1, 2, 3]
    d = ans.to_dict()
    assert d["citations"][0]["chunk_id"] == top.chunk_id and "[1]" in d["answer"]


def test_pipeline_answer_is_grounded_in_cited_chunks(sample_pipeline):
    chunks = {c.chunk_id: c.text for c in sample_pipeline.chunks}
    ans = sample_pipeline.query("What is a dead-letter queue for?", k=3)
    assert "dead-letter" in ans.text.lower()
    assert any("dead-letter" in chunks[c.chunk_id].lower() for c in ans.citations)


def test_pipeline_modes_and_invalid_mode(sample_pipeline):
    for mode in ("bm25", "tfidf", "hybrid"):
        assert sample_pipeline.retrieve("raft leader", k=2, mode=mode)
    with pytest.raises(ValueError):
        sample_pipeline.retrieve("x", mode="bogus")


def test_pipeline_ingest_replaces_documents(tiny_docs):
    from rag_agent.types import Document

    pipe = RAGPipeline()
    pipe.ingest(tiny_docs)
    n1 = len(pipe.chunks)
    pipe.ingest([Document("cats", "Cats", "# Cats\n\nCats now only talk about quantum physics.")])
    assert pipe.num_documents == 3 and len(pipe.chunks) <= n1
    assert pipe.retrieve("quantum physics", k=1)[0].chunk.doc_id == "cats"


def test_pipeline_is_deterministic(sample_pipeline):
    a = sample_pipeline.query("Why add random jitter to retry delays?").to_dict()
    b = sample_pipeline.query("Why add random jitter to retry delays?").to_dict()
    assert a == b


def test_reciprocal_rank_and_relevance_logic():
    item = EvalItem("q", "?", "doc", ("needle",))
    hits = [hit("other#0", "needle"), hit("doc#0", "no match"), hit("doc#1", "A NEEDLE here")]
    assert not is_relevant(hits[0].chunk, item)  # wrong source
    assert not is_relevant(hits[1].chunk, item)  # right source, no answer phrase
    assert is_relevant(hits[2].chunk, item)
    assert reciprocal_rank(hits, item) == pytest.approx(1 / 3)
    assert reciprocal_rank(hits[:2], item) == 0.0


def test_evaluate_metrics_with_fake_search():
    items = [EvalItem("1", "q1", "d", ("a",)), EvalItem("2", "q2", "d", ("b",))]
    results = {"q1": [hit("d#0", "a", 1)], "q2": [hit("x#0", "z", 1), hit("d#1", "b", 2)]}
    m = evaluate(lambda q, k: results[q][:k], items, ks=(1, 2))
    assert m["recall@1"] == 0.5 and m["recall@2"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 0.5) / 2)
    with pytest.raises(ValueError):
        evaluate(lambda q, k: [], [])


def test_eval_set_is_valid_against_corpus(sample_pipeline):
    items = load_eval_set(DEFAULT_EVAL_SET)
    assert len(items) >= 25
    assert len({i.qid for i in items}) == len(items)
    for item in items:  # every question must have at least one gold chunk to be answerable
        assert any(is_relevant(c, item) for c in sample_pipeline.chunks), item.qid


def test_end_to_end_retrieval_quality_floor(sample_pipeline):
    items = load_eval_set(DEFAULT_EVAL_SET)
    m = evaluate(lambda q, k: sample_pipeline.retrieve(q, k=k, mode="hybrid"), items)
    assert m["recall@5"] >= 0.9 and m["mrr"] >= 0.7
