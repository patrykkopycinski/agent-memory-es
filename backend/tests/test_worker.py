"""Worker test: consolidation across two owners, run_once paths.

The duplicate pair is seeded directly (retain now dedups identical writes), so the
worker's consolidate pass is exercised on legacy duplicates."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import auth, embeddings, memory  # noqa: E402
from app.memory import extract_entities  # noqa: E402
from app.store import es, idx  # noqa: E402
from app.worker import run_once  # noqa: E402


def _seed(owner, text):
    doc = {"kind": "semantic", "owner_id": owner, "visibility": "private", "text": text,
           "entities": extract_entities(text), "occurred_at": "2026-01-01T00:00:00Z",
           "active": True, "embedding": embeddings.embed([text])[0]}
    return es("POST", f"/{idx('semantic')}/_doc?refresh=true", doc)["_id"]


auth.create_key("w1")
auth.create_key("w2")
memory.retain("w1", "semantic", "Python toolchain for repo gates is python3.12 venv")
_seed("w1", "Python toolchain for repo gates is python3.12 venv")
memory.retain("w2", "semantic", "Worker two note: git identity uses committer contact@patrykkopycinski.com")

stats = run_once()  # now proposes + applies
assert "w1" in stats and "w2" in stats, stats
assert stats["w1"]["superseded"] >= 1, stats  # duplicate w1 fact superseded
hits = memory.recall("w2", "git identity committer email")["fused"]
assert any("contact@patrykkopycinski.com" in h["text"] for h in hits), hits
print("WORKER: PASS", stats)
