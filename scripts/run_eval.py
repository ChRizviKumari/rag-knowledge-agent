"""Print the retrieval evaluation table (recall@k, MRR) for every configuration."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.cli import format_table, run_chunk_sweep, run_eval  # noqa: E402

if __name__ == "__main__":
    print("Retrieval configurations (chunk_size=500, overlap=80):")
    print(format_table(run_eval()))
    print("\nHybrid retrieval vs chunk size:")
    print(format_table(run_chunk_sweep()))
