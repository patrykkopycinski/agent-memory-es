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
OWNERS = ("t3", "t3b", "t4", "t5", "t6", "t7")
for _o in OWNERS:
    es("POST", f"/{PREFIX}am_rejected/_delete_by_query?refresh=true",
       {"query": {"term": {"owner_id": _o}}})


def _blocked(owner, text):
    try:
        memory.retain(owner, "semantic", text)
    except ValueError as e:
        assert "rejected value" in str(e)
        return True
    return False


class _CountEs:
    """Count ES calls made by the tombstone module."""
    def __enter__(self):
        self.calls = 0
        self.bodies = []
        self._orig = tombstone.es

        def counted(*a, **k):
            self.calls += 1
            self.bodies.append(a[2] if len(a) > 2 else k.get("body"))
            return self._orig(*a, **k)
        tombstone.es = counted
        return self

    def __exit__(self, *exc):
        tombstone.es = self._orig


def _patched(**kw):
    class _P:
        def __enter__(self):
            self.old = {k: getattr(tombstone, k, None) for k in kw}
            for k, v in kw.items():
                setattr(tombstone, k, v)

        def __exit__(self, *exc):
            for k, v in self.old.items():
                setattr(tombstone, k, v)
    return _P()


# --- paging beyond the first page, on the scan path (text above TERMS_CHUNK terms).
# 60 fillers share a probe term and the matching tombstone has the HIGHEST value_key, so it
# sits on the last page; PAGE=5 forces >=12 pages. Any page cap below that misses it.
fill = [f"probe filler{i:03d} unrelated{i:03d} noise{i:03d}" for i in range(60)]
max_filler = max(tombstone._key(tombstone._terms(t)) for t in fill)
n = 0
while tombstone._key(tombstone._terms(f"kubernetes cluster upgrade failed staging v{n}")) <= max_filler:
    n += 1
match_text = f"kubernetes cluster upgrade failed staging v{n}"
reworded = f"staging v{n} kubernetes cluster upgrade failed probe"
for t in fill:
    tombstone.reject("t3", t, "filler")
tombstone.reject("t3", match_text, "wrong claim")
with _patched(PAGE=5, TERMS_CHUNK=3, MAX_PAGES=20), _CountEs() as c:
    assert _blocked("t3", match_text + " " + "probe"), "match on a late page not checked"
    assert c.calls >= 12, f"expected >=12 pages, got {c.calls}"
    assert not _blocked("t3", "completely different banana smoothie recipe probe")

# fail closed: candidate backlog bigger than MAX_PAGES * PAGE is rejected, never allowed
with _patched(PAGE=5, TERMS_CHUNK=3, MAX_PAGES=2):
    rej = tombstone.is_rejected("t3", "completely different banana smoothie recipe probe")
    assert rej and "failing closed" in rej["reason"], rej
    assert _blocked("t3", "completely different banana smoothie recipe probe")

# ES-side matching: 60 tombstones sharing a term cost ONE call even with a tiny page (a scan
# would need 13); the real 80% match is still found. Sort carries a _doc tiebreaker.
for i in range(60):
    tombstone.reject("t3b", f"alice shared{i:03d} unrelated{i:03d}", "filler")
tombstone.reject("t3b", "alice works at microsoft", "wrong claim")
with _patched(PAGE=5), _CountEs() as c:
    assert tombstone.is_rejected("t3b", "alice something else entirely novel") is None
    assert c.calls == 1, f"shared-term tombstones were scanned: {c.calls} calls"
    assert tombstone.is_rejected("t3b", "microsoft is where alice works"), "ES-side 80% match lost"
    assert c.calls == 2, c.calls
    assert {"_doc": "asc"} in c.bodies[0]["sort"], c.bodies[0]["sort"]

# thin (2-term) tombstone: exact term set blocks, a longer text sharing both terms does not
tombstone.reject("t4", "redis cache", "wrong")
assert _blocked("t4", "cache redis"), "exact term-set must still block"
assert not _blocked("t4", "redis cache eviction policy tuned for latency"), "thin tombstone over-blocks"

# legacy record (terms: [], exact value_key only) written straight into the index
legacy = "legacy rejected claim about widgets"
es("POST", f"/{PREFIX}am_rejected/_doc?refresh=true", {
    "value_key": tombstone._key(tombstone._terms(legacy)), "terms": [], "owner_id": "t5",
    "reason": "legacy", "source_id": "", "rejected_at": "2025-01-01T00:00:00Z"})
assert _blocked("t5", legacy), "legacy exact value_key record must block"
assert _blocked("t5", "widgets legacy rejected claim about"), "legacy match is on the term set"
assert not _blocked("t5", legacy + " extra words here"), "legacy record must not fuzzy-match"

# term-less text: no tombstone is written, and none can block other term-less texts
assert tombstone.reject("t6", "ok", "deleted by owner") is None
assert es("POST", f"/{PREFIX}am_rejected/_count", {"query": {"term": {"owner_id": "t6"}}})["count"] == 0
assert not _blocked("t6", "no")
assert not _blocked("t6", "hi :)")

# index readiness is cached and a lost create race counts as success
assert tombstone._index_ready is True
tombstone._index_ready = False
_orig_es = tombstone.es


def _racy(method, path, body=None):
    if method == "GET":
        raise RuntimeError("ES GET -> 404")
    if method == "PUT":
        raise RuntimeError("ES PUT -> 400: resource_already_exists_exception")
    return _orig_es(method, path, body)


tombstone.es = _racy
try:
    tombstone._ensure_index()
finally:
    tombstone.es = _orig_es
assert tombstone._index_ready is True
with _CountEs() as c:
    tombstone._ensure_index()
    assert c.calls == 0, "ensure_index must not hit ES once ready"
print("TOMBSTONE: PASS")
