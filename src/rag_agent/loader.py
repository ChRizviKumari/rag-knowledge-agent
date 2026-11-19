"""Load Markdown and plain-text documents from disk."""

from __future__ import annotations

import re
from pathlib import Path

from .types import Document

SUPPORTED_SUFFIXES = (".md", ".txt")
_HEADING_RE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def extract_title(text: str, fallback: str) -> str:
    """Return the first level-1 Markdown heading, or ``fallback``."""
    match = _HEADING_RE.search(text)
    return match.group(1).strip() if match else fallback


def load_document(path: Path, root: Path | None = None) -> Document:
    """Read a single ``.md``/``.txt`` file into a :class:`Document`.

    The ``doc_id`` is the file path relative to ``root`` without its suffix.
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix!r} ({path})")
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(root) if root else Path(path.name)
    doc_id = rel.with_suffix("").as_posix()
    return Document(doc_id=doc_id, title=extract_title(text, doc_id), text=text, source=str(path))


def load_directory(directory: str | Path) -> list[Document]:
    """Recursively load every supported file under ``directory`` (sorted, deterministic)."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [load_document(p, root) for p in files]
