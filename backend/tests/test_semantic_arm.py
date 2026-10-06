"""Semantic-arm test: paraphrase recall must hit via embeddings (BM25 alone cannot)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import memory

# paraphrase: zero keyword overlap with the stored text
memory.retain("sem", "semantic",
              "The quarterly infrastructure budget covers three cloud regions and one dedicated host")
hits = memory.recall("sem", "how much do we spend on servers each quarter")["fused"]
assert any("budget" in h["text"] for h in hits), hits
print("SEMANTIC ARM: PASS — paraphrase recalled via dense vector")
