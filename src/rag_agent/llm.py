"""LLM interface with a deterministic extractive default and an optional Anthropic client."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from .tokenizer import tokenize
from .types import SearchHit

NO_ANSWER = "I don't know based on the provided context."
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def build_prompt(question: str, contexts: Sequence[SearchHit]) -> str:
    """Render numbered context passages and the question into a grounded prompt."""
    blocks = [f"[{i}] ({h.chunk.doc_id}, {h.chunk.chunk_id})\n{h.chunk.text}" for i, h in enumerate(contexts, 1)]
    return (
        "Answer the question using ONLY the numbered context passages. Cite passages "
        "inline like [1]. If the context is insufficient, say you don't know.\n\n"
        "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"
    )


class LLM(ABC):
    """Generates an answer from a question and retrieved, ranked contexts."""

    @abstractmethod
    def generate(self, question: str, contexts: Sequence[SearchHit]) -> str:
        """Return answer text containing ``[n]`` markers that refer to ``contexts[n-1]``."""


class MockLLM(LLM):
    """Deterministic extractive 'LLM': returns the context sentences that best match the question.

    No network, no randomness; used by default and in tests. It is extractive, so
    it can only quote the corpus and never hallucinates (nor paraphrases).
    """

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> str:
        q_terms = set(tokenize(question))
        candidates: list[tuple[float, int, int, str]] = []
        for marker, hit in enumerate(contexts, start=1):
            for pos, raw in enumerate(_SENTENCE_RE.split(hit.chunk.text)):
                sentence = raw.strip().lstrip("#").strip()
                if len(sentence) < 15:
                    continue
                overlap = len(q_terms & set(tokenize(sentence)))
                if overlap:
                    # more overlap first; earlier-ranked context breaks ties
                    candidates.append((-float(overlap), marker, pos, sentence))
        if not candidates:
            return NO_ANSWER
        candidates.sort()
        chosen = candidates[: self.max_sentences]
        chosen.sort(key=lambda c: (c[1], c[2]))  # present in reading order
        return " ".join(f"{s.rstrip()} [{m}]" for _, m, _, s in chosen)


class AnthropicLLM(LLM):  # pragma: no cover - needs network + API key, not exercised in tests
    """Answer generation with the Anthropic Messages API.

    The ``anthropic`` package is imported lazily so the rest of the project has
    no dependency on it. Reads ``ANTHROPIC_API_KEY`` (and optionally
    ``ANTHROPIC_MODEL``) from the environment.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 512) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install the optional dependency: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.max_tokens = max_tokens

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def get_llm(name: str = "mock") -> LLM:
    """Factory: ``"mock"`` (default) or ``"anthropic"``."""
    if name == "mock":
        return MockLLM()
    if name == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"Unknown LLM backend: {name!r}")
