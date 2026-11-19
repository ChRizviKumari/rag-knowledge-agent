"""Retrieval evaluation: recall@k and MRR against a hand-written Q/A set.

A chunk is *relevant* to a question when it comes from the gold source document
AND contains at least one of the gold ``answer_contains`` phrases
(case-insensitive). With exactly one answer-bearing region per question,
recall@k equals the hit rate: the fraction of questions with a relevant chunk in
the top k.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .types import Chunk, SearchHit


@dataclass(frozen=True)
class EvalItem:
    qid: str
    question: str
    source: str
    answer_contains: tuple[str, ...]


def load_eval_set(path: str | Path) -> list[EvalItem]:
    """Parse the JSON evaluation file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalItem(r["id"], r["question"], r["source"], tuple(r["answer_contains"])) for r in raw]


def is_relevant(chunk: Chunk, item: EvalItem) -> bool:
    """Gold-chunk test used for every metric."""
    if chunk.doc_id != item.source:
        return False
    lowered = chunk.text.lower()
    return any(phrase.lower() in lowered for phrase in item.answer_contains)


def reciprocal_rank(hits: Sequence[SearchHit], item: EvalItem) -> float:
    """1/rank of the first relevant hit, or 0.0 when none is retrieved."""
    for position, hit in enumerate(hits, start=1):
        if is_relevant(hit.chunk, item):
            return 1.0 / position
    return 0.0


def hit_at_k(hits: Sequence[SearchHit], item: EvalItem, k: int) -> bool:
    return any(is_relevant(h.chunk, item) for h in hits[:k])


def evaluate(
    search: Callable[[str, int], list[SearchHit]],
    items: Sequence[EvalItem],
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Compute ``recall@k`` for each k plus ``mrr`` (cut off at ``max(ks)``) and ``doc_recall@max_k``."""
    if not items:
        raise ValueError("evaluation set is empty")
    max_k = max(ks)
    totals = {k: 0 for k in ks}
    rr_sum = 0.0
    doc_hits = 0
    for item in items:
        hits = search(item.question, max_k)
        rr_sum += reciprocal_rank(hits, item)
        doc_hits += any(h.chunk.doc_id == item.source for h in hits)
        for k in ks:
            totals[k] += hit_at_k(hits, item, k)
    n = len(items)
    metrics = {f"recall@{k}": totals[k] / n for k in ks}
    metrics["mrr"] = rr_sum / n
    metrics[f"doc_recall@{max_k}"] = doc_hits / n
    return metrics
