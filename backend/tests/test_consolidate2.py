"""Consolidate hardening: dry_run default + explicit dispositions + supersede owner check.
dry_run=True (default) returns proposed pairs without mutating. apply=True performs them.
keep_both is a first-class disposition. Supersede asserts same-owner on both ends.

Near-duplicate pairs are seeded directly: retain now dedups identical writes
(dedup-on-write), so consolidation is exercised on legacy/imported duplicates."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import embeddings, memory  # noqa: E402
from app.memory import extract_entities  # noqa: E402
from app.store import es, idx  # noqa: E402


def _seed(owner, text, kind="semantic", visibility="private"):
    doc = {"kind": kind, "owner_id": owner, "visibility": visibility, "text": text,
           "entities": extract_entities(text), "occurred_at": "2026-01-01T00:00:00Z",
           "active": True, "embedding": embeddings.embed([text])[0]}
    return es("POST", f"/{idx(kind)}/_doc?refresh=true", doc)["_id"]


# dedup-on-write: the identical second retain collapses onto the first doc
p1 = memory.retain("c1", "semantic", "Vue is the frontend framework Patryk uses now")
p2 = memory.retain("c1", "semantic", "Vue is the frontend framework Patryk uses now")
assert p2["deduped"] is True and p2["_id"] == p1["_id"], (p1, p2)

# a legacy duplicate (seeded past the write guard) is still consolidated
_seed("c1", "Vue is the frontend framework Patryk uses now")
plan = memory.consolidate("c1", dry_run=True)
assert plan["mutated"] == 0 and plan["proposals"], plan
assert all(p["disposition"] in ("supersede", "keep_both") for p in plan["proposals"]), plan
# apply
done = memory.consolidate("c1", dry_run=False)
assert done["superseded"] >= 1, done
# cross-owner supersede refused
a = memory.retain("c1", "semantic", "owner-c1 only fact for supersede guard")
b = memory.retain("c2", "semantic", "owner-c2 near duplicate for supersede guard")
try:
    memory._supersede("c2", a["_id"], b["_id"])
    raise AssertionError("cross-owner supersede should raise")
except PermissionError:
    pass
print("CONSOLIDATE-HARDENING: PASS", done)
