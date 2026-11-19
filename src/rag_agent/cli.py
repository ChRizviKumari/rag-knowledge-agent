"""Command line interface: ``python -m rag_agent {ask,eval,serve}``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .chunker import ChunkerConfig
from .evaluation import evaluate, load_eval_set
from .llm import get_llm
from .pipeline import DEFAULT_CORPUS, DEFAULT_EVAL_SET, MODES, RAGPipeline
from .rerank import LexicalOverlapReranker


def build_pipeline(llm: str = "mock", rerank: bool = False) -> RAGPipeline:
    """Pipeline over the bundled sample corpus."""
    pipe = RAGPipeline(
        chunker=ChunkerConfig(),
        llm=get_llm(llm),
        reranker=LexicalOverlapReranker() if rerank else None,
    )
    pipe.ingest_directory(DEFAULT_CORPUS)
    return pipe


def run_eval() -> dict[str, dict[str, float]]:
    """Evaluate every retrieval configuration on the bundled eval set."""
    items = load_eval_set(DEFAULT_EVAL_SET)
    results: dict[str, dict[str, float]] = {}
    for rerank in (False, True):
        pipe = build_pipeline(rerank=rerank)
        for mode in MODES:
            if rerank and mode != "hybrid":
                continue
            label = f"{mode}+rerank" if rerank else mode
            results[label] = evaluate(lambda q, k, m=mode, p=pipe: p.retrieve(q, k=k, mode=m, rerank=rerank), items)
    return results


def run_chunk_sweep(sizes: Sequence[int] = (150, 250, 500, 800)) -> dict[str, dict[str, float]]:
    """Hybrid (no rerank) retrieval quality as a function of chunk size (overlap = size // 6).

    Caveat: a gold phrase split across a chunk boundary has no relevant chunk, so
    very small sizes are penalised by the metric as well as by retrieval.
    """
    items = load_eval_set(DEFAULT_EVAL_SET)
    out: dict[str, dict[str, float]] = {}
    for size in sizes:
        pipe = RAGPipeline(chunker=ChunkerConfig(chunk_size=size, chunk_overlap=size // 6))
        n_chunks = pipe.ingest_directory(DEFAULT_CORPUS)
        metrics = evaluate(lambda q, k, p=pipe: p.retrieve(q, k=k, mode="hybrid", rerank=False), items)
        out[f"size={size} ({n_chunks}ch)"] = metrics
    return out


def format_table(results: dict[str, dict[str, float]]) -> str:
    cols = list(next(iter(results.values())))
    lines = ["config".ljust(16) + "".join(c.rjust(14) for c in cols)]
    for name, m in results.items():
        lines.append(name.ljust(16) + "".join(f"{m[c]:14.3f}" for c in cols))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag_agent", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="ask a question over the sample corpus")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=4)
    ask.add_argument("--mode", choices=MODES, default="hybrid")
    ask.add_argument("--llm", choices=("mock", "anthropic"), default="mock")
    ask.add_argument("--rerank", action="store_true")
    sub.add_parser("eval", help="print recall@k / MRR for all configurations")
    serve = sub.add_parser("serve", help="start the Flask API")
    serve.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)

    if args.cmd == "ask":
        answer = build_pipeline(args.llm, args.rerank).query(args.question, k=args.k, mode=args.mode)
        print(answer.text)
        for c in answer.citations:
            print(f"  [{c.marker}] {c.source} :: {c.chunk_id} (score={c.score:.4f})")
    elif args.cmd == "eval":
        print(format_table(run_eval()))
    else:
        from .api import create_app

        pipe = build_pipeline()
        create_app(pipe).run(port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
