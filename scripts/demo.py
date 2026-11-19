"""Demo: ingest the bundled corpus and answer a few questions with citations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.cli import build_pipeline  # noqa: E402

QUESTIONS = [
    "How many failures can a five-server Raft cluster tolerate?",
    "Why add random jitter to retries?",
    "What is the weakness of a fixed window counter?",
    "How do I bake sourdough bread?",
]

if __name__ == "__main__":
    pipe = build_pipeline(rerank=True)
    print(f"Indexed {pipe.num_documents} documents -> {len(pipe.chunks)} chunks\n")
    for q in QUESTIONS:
        ans = pipe.query(q, k=3)
        print(f"Q: {q}\nA: {ans.text}")
        for c in ans.citations:
            print(f"   [{c.marker}] {c.source} :: {c.chunk_id}  section={c.section!r}  score={c.score:.4f}")
        print()
