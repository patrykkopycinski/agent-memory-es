"""Tombstone tests: value-keyed rejection blocks the write path, survives rewording.
Each test wipes the owners it uses, so sections fail (and rerun) independently."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (guard: refuses anything but a throwaway ES)

import pytest
from fastapi.testclient import TestClient

from app import auth, memory, tombstone
from app.main import app
from app.store import es, PREFIX


def _wipe(*owners):
    tombstone._ensure_index()
    es("POST", f"/{PREFIX}am_rejected/_delete_by_query?refresh=true",
       {"query": {"terms": {"owner_id": list(owners)}}})


@pytest.fixture
def owners():
    """Wipe the named owners' tombstones before the test; call as owners('a', 'b')."""
    def wipe(*names):
        _wipe(*names)
        return names if len(names) > 1 else names[0]
    return wipe


def _blocked(owner, text):
    try:
        memory.retain(owner, "semantic", text)
    except ValueError as e:
        assert "rejected value" in str(e)
        return True
    return False


class _CountEs:
    """Count (and record bodies of) ES calls made by the tombstone module."""
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


@pytest.fixture
def patched(monkeypatch):
    return lambda **kw: [monkeypatch.setattr(tombstone, k, v, raising=False) for k, v in kw.items()]


def test_reject_blocks_exact_and_reworded_retain(owners):
    o1, o2 = owners("t1", "t2")
    tombstone.reject(o1, "Alice works at Microsoft", "superseded: moved to Google Mar 2026")
    assert _blocked(o1, "Alice works at Microsoft")
    assert _blocked(o1, "Microsoft is where Alice works")
    memory.retain(o2, "semantic", "Alice works at Microsoft")             # other owner unaffected
    memory.retain(o1, "semantic", "Alice works at Google research team")  # other value unaffected


def test_late_page_match_found_with_tiny_page(owners, patched):
    """60 fillers share a probe term and the match has the HIGHEST value_key, so it sits on the
    last page; PAGE=5 forces >=12 pages. TERMS_CHUNK=3 routes the text through the paged scan."""
    o = owners("t3")
    fill = [f"probe filler{i:03d} unrelated{i:03d} noise{i:03d}" for i in range(60)]
    max_filler = max(tombstone._key(tombstone._terms(t)) for t in fill)
    n = 0
    while tombstone._key(tombstone._terms(f"kubernetes cluster upgrade failed staging v{n}")) <= max_filler:
        n += 1
    match_text = f"kubernetes cluster upgrade failed staging v{n}"
    for t in fill:
        tombstone.reject(o, t, "filler")
    tombstone.reject(o, match_text, "wrong claim")
    patched(PAGE=5, TERMS_CHUNK=3, MAX_PAGES=20)
    with _CountEs() as c:
        assert _blocked(o, match_text + " probe"), "match on a late page not checked"
        assert c.calls >= 12, f"expected >=12 pages, got {c.calls}"
        assert not _blocked(o, "completely different banana smoothie recipe probe")


def test_page_cap_fails_closed_with_distinct_capacity_error(owners, patched):
    o = owners("t3c")
    for i in range(12):
        tombstone.reject(o, f"probe filler{i:03d} unrelated{i:03d} noise{i:03d}", "filler")
    patched(PAGE=5, TERMS_CHUNK=3, MAX_PAGES=2)
    text = "completely different banana smoothie recipe probe"
    rej = tombstone.is_rejected(o, text)
    assert rej and rej.get("capacity") is True and "failing closed" in rej["reason"], rej
    with pytest.raises(ValueError) as ei:
        memory.retain(o, "semantic", text)
    msg = str(ei.value)
    assert "too many candidate tombstones" in msg and "rejected value" not in msg, msg


def test_capacity_error_is_a_4xx_over_rest(owners, patched):
    o = owners("t3d")
    key = auth.create_key(o)
    for i in range(12):
        tombstone.reject(o, f"probe filler{i:03d} unrelated{i:03d} noise{i:03d}", "filler")
    patched(PAGE=5, TERMS_CHUNK=3, MAX_PAGES=2)
    r = TestClient(app).post("/memory/retain", headers={"X-API-Key": key},
                             json={"kind": "semantic", "text": "banana smoothie recipe probe",
                                   "extract_facts": False})
    assert 400 <= r.status_code < 500, (r.status_code, r.text)
    assert "too many candidate tombstones" in r.json()["detail"]


def test_shared_term_tombstones_cost_one_es_call(owners, patched):
    """60 tombstones sharing a term cost ONE call even with a tiny page (a scan would need 13);
    the real 80% match is still found. Sort carries a _doc tiebreaker."""
    o = owners("t3b")
    for i in range(60):
        tombstone.reject(o, f"alice shared{i:03d} unrelated{i:03d}", "filler")
    tombstone.reject(o, "alice works at microsoft", "wrong claim")
    patched(PAGE=5)
    with _CountEs() as c:
        assert tombstone.is_rejected(o, "alice something else entirely novel") is None
        assert c.calls == 1, f"shared-term tombstones were scanned: {c.calls} calls"
        assert tombstone.is_rejected(o, "microsoft is where alice works"), "ES-side 80% match lost"
        assert c.calls == 2, c.calls
        assert {"_doc": "asc"} in c.bodies[0]["sort"], c.bodies[0]["sort"]


def test_thin_tombstone_needs_exact_term_set(owners):
    o = owners("t4")
    tombstone.reject(o, "redis cache", "wrong")
    assert _blocked(o, "cache redis"), "exact term-set must still block"
    longer = "redis cache eviction policy tuned for latency"
    assert not _blocked(o, longer), "thin tombstone over-blocks"
    # pin the ES-side rule itself (n+1 for <3 terms), not just the Python re-check: ES must
    # not even return the thin tombstone as a candidate for the longer text
    with _CountEs() as c:
        tombstone.is_rejected(o, longer)
    resp = es("POST", f"/{PREFIX}am_rejected/_search", c.bodies[0])
    assert resp["hits"]["hits"] == [], resp["hits"]["hits"]


def test_legacy_empty_terms_record_matches_exact_only(owners):
    o = owners("t5")
    legacy = "legacy rejected claim about widgets"
    es("POST", f"/{PREFIX}am_rejected/_doc?refresh=true", {
        "value_key": tombstone._key(tombstone._terms(legacy)), "terms": [], "owner_id": o,
        "reason": "legacy", "source_id": "", "rejected_at": "2025-01-01T00:00:00Z"})
    assert _blocked(o, legacy), "legacy exact value_key record must block"
    assert _blocked(o, "widgets legacy rejected claim about"), "legacy match is on the term set"
    assert not _blocked(o, legacy + " extra words here"), "legacy record must not fuzzy-match"


def test_termless_text_writes_no_tombstone(owners):
    o = owners("t6")
    assert tombstone.reject(o, "ok", "deleted by owner") is None
    assert es("POST", f"/{PREFIX}am_rejected/_count", {"query": {"term": {"owner_id": o}}})["count"] == 0
    assert not _blocked(o, "no")
    assert not _blocked(o, "hi :)")


def test_index_readiness_cached_and_create_race_is_success(monkeypatch):
    tombstone._ensure_index()
    assert tombstone._index_ready is True
    monkeypatch.setattr(tombstone, "_index_ready", False)
    orig = tombstone.es

    def racy(method, path, body=None):
        if method == "GET":
            raise RuntimeError("ES GET -> 404")
        if method == "PUT":
            raise RuntimeError("ES PUT -> 400: resource_already_exists_exception")
        return orig(method, path, body)

    monkeypatch.setattr(tombstone, "es", racy)
    tombstone._ensure_index()
    assert tombstone._index_ready is True
    monkeypatch.setattr(tombstone, "es", orig)
    with _CountEs() as c:
        tombstone._ensure_index()
        assert c.calls == 0, "ensure_index must not hit ES once ready"
