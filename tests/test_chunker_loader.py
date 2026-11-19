from pathlib import Path

import pytest

from rag_agent.chunker import ChunkerConfig, chunk_document
from rag_agent.loader import extract_title, load_directory, load_document
from rag_agent.pipeline import DEFAULT_CORPUS
from rag_agent.types import Document

LONG = "# Title\n\n" + "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(12))


def test_chunks_respect_size_limit():
    cfg = ChunkerConfig(chunk_size=200, chunk_overlap=40)
    chunks = chunk_document(Document("d", "d", LONG), cfg)
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)


def test_offsets_reproduce_text():
    doc = Document("d", "d", LONG)
    for c in chunk_document(doc, ChunkerConfig(200, 40)):
        assert doc.text[c.start:c.end] == c.text


def test_consecutive_chunks_overlap():
    doc = Document("d", "d", LONG)
    chunks = chunk_document(doc, ChunkerConfig(200, 60))
    overlaps = [a.end - b.start for a, b in zip(chunks, chunks[1:])]
    assert any(o > 0 for o in overlaps)
    assert all(o <= 60 for o in overlaps)


def test_zero_overlap_chunks_do_not_overlap():
    doc = Document("d", "d", LONG)
    chunks = chunk_document(doc, ChunkerConfig(200, 0))
    assert all(b.start >= a.end for a, b in zip(chunks, chunks[1:]))


def test_coverage_every_word_is_in_some_chunk():
    doc = Document("d", "d", LONG)
    joined = " ".join(c.text for c in chunk_document(doc, ChunkerConfig(150, 30)))
    for i in range(12):
        assert f"Paragraph {i}." in joined


def test_chunk_ids_unique_and_indexed():
    chunks = chunk_document(Document("doc", "t", LONG), ChunkerConfig(200, 40))
    assert [c.chunk_id for c in chunks] == [f"doc#{i}" for i in range(len(chunks))]


def test_section_metadata_tracks_markdown_headings():
    text = "# Top\n\nintro text here.\n\n## Alpha\n\n" + "alpha " * 30 + "\n\n## Beta\n\n" + "beta " * 30
    chunks = chunk_document(Document("d", "d", text), ChunkerConfig(100, 10))
    assert chunks[0].section == "Top"
    assert any(c.section == "Alpha" for c in chunks) and chunks[-1].section == "Beta"


def test_oversized_token_is_hard_split():
    chunks = chunk_document(Document("d", "d", "x" * 1000), ChunkerConfig(300, 0))
    assert all(len(c.text) <= 300 for c in chunks) and len(chunks) == 4


def test_empty_and_whitespace_documents_yield_no_chunks():
    assert chunk_document(Document("d", "d", "")) == []
    assert chunk_document(Document("d", "d", "  \n\n  ")) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_config_rejected(size, overlap):
    with pytest.raises(ValueError):
        ChunkerConfig(size, overlap)


def test_loader_reads_md_and_txt_and_skips_others(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Alpha Doc\n\nbody")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("plain text")
    (tmp_path / "ignored.pdf").write_text("nope")
    docs = load_directory(tmp_path)
    assert [d.doc_id for d in docs] == ["a", "sub/b"]
    assert docs[0].title == "Alpha Doc" and docs[1].title == "sub/b"


def test_loader_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "missing")
    bad = tmp_path / "x.pdf"
    bad.write_text("x")
    with pytest.raises(ValueError):
        load_document(bad)


def test_extract_title_fallback():
    assert extract_title("no heading", "fb") == "fb"
    assert extract_title("## sub only\n# Real", "fb") == "Real"


def test_bundled_corpus_has_eight_documents():
    assert len(load_directory(DEFAULT_CORPUS)) == 8
