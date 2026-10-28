"""Tombstone tests: value-keyed rejection blocks the write path, survives rewording."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import memory, tombstone
from app.store import es, PREFIX

tombstone.reject("t1", "Alice works at Microsoft", "superseded: moved to Google Mar 2026")
# exact re-assert blocked
try:
    memory.retain("t1", "semantic", "Alice works at Microsoft")
    raise AssertionError("write should have been rejected")
except ValueError as e:
    assert "rejected value" in str(e)
# reworded re-assert (same salient terms) blocked
try:
    memory.retain("t1", "semantic", "Microsoft is where Alice works")
    raise AssertionError("reworded write should have been rejected")
except ValueError:
    pass
# different owner unaffected
memory.retain("t2", "semantic", "Alice works at Microsoft")
# different value unaffected
memory.retain("t1", "semantic", "Alice works at Google research team")

# Rerunnable on a persistent throwaway ES: wipe the owners this section uses.
for _o in ("t3", "t4"):
    es("POST", f"/{PREFIX}am_rejected/_delete_by_query?refresh=true",
       {"query": {"term": {"owner_id": _o}}})


def _blocked(owner, text):
    try:
        memory.retain(owner, "semantic", text)
    except ValueError as e:
        assert "rejected value" in str(e)
        return True
    return False


# >50 tombstones: the match is written LAST, so it falls outside any first-50 window
for i in range(60):
    tombstone.reject("t3", f"filler{i:03d} unrelated{i:03d} noise{i:03d} words", "filler")
tombstone.reject("t3", "kubernetes cluster upgrade failed staging", "wrong claim")
assert _blocked("t3", "staging kubernetes cluster upgrade failed"), "match beyond first 50 not checked"
assert not _blocked("t3", "completely different banana smoothie recipe")

# thin (2-term) tombstone: exact term set blocks, a longer text sharing both terms does not
tombstone.reject("t4", "redis cache", "wrong")
assert _blocked("t4", "cache redis"), "exact term-set must still block"
assert not _blocked("t4", "redis cache eviction policy tuned for latency"), "thin tombstone over-blocks"
print("TOMBSTONE: PASS")
