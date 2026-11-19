"""Tokenization shared by every lexical retriever and the extractive LLM."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:/[0-9]+)?")

STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be but by can does do for from has have how if in into is it its
    of on or so such than that the their then there these they this to was were what when
    where which who why will with without not no only also each any all more most one""".split()
)


def stem(token: str) -> str:
    """Very light suffix stripper (plural / -ing / -ed) for matching word forms.

    It is deliberately crude: it only needs to be *consistent* between indexing
    and querying, not linguistically correct.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, *, use_stemming: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, split into alphanumeric tokens, drop stopwords, and stem."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens
