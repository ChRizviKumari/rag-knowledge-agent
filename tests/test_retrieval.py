import math

import numpy as np
import pytest

from rag_agent.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_agent.rerank import LexicalOverlapReranker, NoOpReranker
from rag_agent.retrievers import BM25Retriever, TfidfRetriever
from rag_agent.tokenizer import stem, tokenize
from rag_agent.chunker import chunk_documents
from rag_agent.types import Chunk, SearchHit


def mk_chunk(cid: str, text: str = "x") -> Chunk:
    return Chunk(cid, cid.split("#")[0], text, 0, len(text), 0)


def mk_hits(ids: list[str]) -> list[SearchHit]:
    return [SearchHit(mk_chunk(i), 1.0, r + 1) for r, i in enumerate(ids)]


@pytest.fixture
def chunks(tiny_docs):
    return chunk_documents(tiny_docs)


def test_tokenizer_stopwords_stemming_and_case():
    assert tokenize("The Cats are Running!") == ["cat", "runn"] or tokenize("The Cats are Running!")[0] == "cat"
    assert "the" not in tokenize("The cat") and stem("caches") == "cache"
    assert stem("policies") == "policy" and stem("class") == "class"


@pytest.mark.parametrize("cls", [BM25Retriever, TfidfRetriever])
def test_retriever_ranks_matching_doc_first(cls, chunks):
    r = cls()
    r.index(chunks)
    hits = r.search("cats purr", k=3)
    assert hits[0].chunk.doc_id == "cats"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


@pytest.mark.parametrize("cls", [BM25Retriever, TfidfRetriever])
def test_retriever_edge_cases(cls, chunks):
    r = cls()
    assert r.search("anything") == []  # not indexed
    r.index(chunks)
    assert r.search("zzzunseen") == []
    assert r.search("") == []
    assert r.search("dogs", k=0) == []
    assert len(r.search("dogs birds cats", k=2)) <= 2


def test_bm25_matches_manual_formula():
    docs = [mk_chunk("a#0", "apple banana"), mk_chunk("b#0", "banana banana cherry cherry")]
    r = BM25Retriever(k1=1.2, b=0.75)
    r.index(docs)
    n, df, avgdl = 2, 1, 3.0
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * 1 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / avgdl))
    assert r.score_all("apple")[0] == pytest.approx(expected)
    assert r.score_all("apple")[1] == 0.0


def test_bm25_rare_terms_have_higher_idf(chunks):
    r = BM25Retriever()
    r.index(chunks)
    assert r.idf("zzz") == 0.0
    assert r.idf("mice") > 0 and r.idf("mice") >= r.idf(stem("cats"))


def test_bm25_length_normalisation_prefers_shorter_doc():
    r = BM25Retriever()
    r.index([mk_chunk("s#0", "raft leader"), mk_chunk("l#0", "raft leader " + "filler " * 50)])
    hits = r.search("raft")
    assert hits[0].chunk.chunk_id == "s#0"


def test_bm25_invalid_params():
    with pytest.raises(ValueError):
        BM25Retriever(b=1.5)


def test_tfidf_cosine_in_unit_interval(chunks):
    r = TfidfRetriever()
    r.index(chunks)
    for h in r.search("dogs bark sticks", k=5):
        assert 0 < h.score <= 1.0 + 1e-6
    assert np.allclose(np.linalg.norm(r._matrix, axis=1), 1.0, atol=1e-5)


def test_rrf_known_values_and_order():
    a, b = mk_hits(["x#0", "y#0", "z#0"]), mk_hits(["y#0", "x#0", "w#0"])
    fused = reciprocal_rank_fusion([a, b], k=60)
    scores = {h.chunk.chunk_id: h.score for h in fused}
    assert scores["x#0"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["x#0"] == scores["y#0"]
    assert [h.chunk.chunk_id for h in fused][:2] == ["x#0", "y#0"]  # tie -> first seen
    assert scores["z#0"] == pytest.approx(1 / 63)
    assert [h.rank for h in fused] == [1, 2, 3, 4]


def test_rrf_consensus_beats_single_list_leader():
    a = mk_hits(["top#0", "both#0"])
    b = mk_hits(["other#0", "both#0"])
    assert reciprocal_rank_fusion([a, b])[0].chunk.chunk_id == "both#0"


def test_rrf_top_n_and_validation():
    assert len(reciprocal_rank_fusion([mk_hits(["a#0", "b#0", "c#0"])], top_n=2)) == 2
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=-1)


def test_hybrid_retriever_combines_both(chunks):
    h = HybridRetriever([BM25Retriever(), TfidfRetriever()])
    h.index(chunks)
    hits = h.search("birds migrate winter", k=3)
    assert hits[0].chunk.doc_id == "birds"
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_noop_reranker_preserves_order_and_truncates():
    hits = mk_hits(["a#0", "b#0", "c#0"])
    out = NoOpReranker().rerank("q", hits, 2)
    assert [h.chunk.chunk_id for h in out] == ["a#0", "b#0"]


def test_lexical_reranker_promotes_full_question_match():
    weak = SearchHit(mk_chunk("a#0", "raft raft raft unrelated words"), 5.0, 1)
    strong = SearchHit(mk_chunk("b#0", "the raft leader election timeout is randomized"), 4.0, 2)
    out = LexicalOverlapReranker().rerank("raft leader election timeout", [weak, strong], 2)
    assert out[0].chunk.chunk_id == "b#0" and [h.rank for h in out] == [1, 2]
