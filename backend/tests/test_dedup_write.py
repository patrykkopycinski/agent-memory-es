"""Cause 1 fix contract: dedup-on-write is exact-text-hash only; near-duplicates are
WRITTEN and linked (supersedes/superseded_by), never silently dropped.

Unit tests monkeypatch the kNN probe for deterministic branch coverage; the
integration tests at the bottom run against live ES (AMES_ES_URL) in the backend
image, like the rest of the suite. They assert the actual regression: two
same-topic sessions with different content must BOTH be retrievable.

  docker compose -f docker-compose.yml -f <vm>.yml run --rm --no-deps \
    -v <repo>:/srv -w /srv/backend ames-backend python -m pytest tests/test_dedup_write.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory  # noqa: E402
from app.store import es, idx, MAPPINGS  # noqa: E402

OWNER = "qdedupwrite"


def _wipe(owner=OWNER):
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": owner}}})


def _count(owner=OWNER, kind="semantic"):
    r = es("POST", f"/{idx(kind)}/_count", {"query": {"term": {"owner_id": owner}}})
    return r["count"]


# ------------------------------------------------------------------ pure unit

def test_text_hash_normalizes_case_and_whitespace():
    a = memory.text_hash("OmniRoute   heap watchdog")
    b = memory.text_hash("omniroute heap watchdog ")
    assert a == b
    assert a != memory.text_hash("omniroute heap watchdog!")
    assert len(a) == 64


def test_scope_filter_is_owner_scoped_only_for_private():
    priv = memory._scope_filter("me", "semantic", "private")["bool"]["filter"]
    assert {"term": {"owner_id": "me"}} in priv
    shared = memory._scope_filter("me", "semantic", "common")["bool"]["filter"]
    assert not any("owner_id" in list(f.get("term", {})) for f in shared)


def test_mapping_declares_text_hash_keyword():
    assert MAPPINGS["mappings"]["properties"]["text_hash"] == {"type": "keyword"}


# ------------------------------------------------- deterministic branch tests

def test_exact_hash_dedup_wins_before_any_knn_probe():
    _wipe()
    calls = []
    orig = memory._nearest_dup
    memory._nearest_dup = lambda *a, **k: calls.append(1) or None
    try:
        first = memory.retain(OWNER, "semantic", "Gateway retry budget is twelve")
        calls[:] = []          # the first write legitimately probes kNN to link near-dups
        again = memory.retain(OWNER, "semantic", "gateway  retry budget is twelve")
    finally:
        memory._nearest_dup = orig
    assert first["deduped"] is False and first["dedup_reason"] == "distinct"
    assert again["deduped"] is True and again["_id"] == first["_id"]
    assert again["dedup_reason"] == "exact_text_hash"
    assert calls == [], "exact-hash duplicates must not pay for an embedding+kNN probe"
    assert _count() == 1


def test_cosine_similarity_alone_never_drops_the_newer_session():
    """A 0.99-cosine drop is NOT safe while the embedder only sees the head: two
    sessions that share an opening line score ~1.0, so dropping on cosine would
    re-lose the newer session. Only an exact text hash may drop."""
    _wipe()
    first = memory.retain(OWNER, "episodic", "user: standup notes live in the wiki\n"
                                             "assistant: noted")
    orig = memory._nearest_dup
    memory._nearest_dup = lambda *a, **k: (first["ids"][0], 1.0)   # identical heads
    try:
        second = memory.retain(
            OWNER, "episodic",
            "user: standup notes live in the wiki\nassistant: noted\n"
            "user: my flight to Lisbon is BA248 on 2024-03-02\nassistant: saved")
    finally:
        memory._nearest_dup = orig
    assert second["deduped"] is False, "cosine 1.0 must not drop a longer session"
    assert second["dedup_reason"] == "near_duplicate_linked", second
    assert _count(kind="episodic") == 2
    got = memory.recall(OWNER, "flight to Lisbon BA248", kinds=["episodic"], size=8)
    assert any("BA248" in r["text"] for r in got["results"])


def test_near_duplicate_is_written_and_linked_both_ways():
    _wipe()
    first = memory.retain(OWNER, "semantic", "Deploy window is Tuesday 09:00 UTC")
    orig = memory._nearest_dup
    memory._nearest_dup = lambda *a, **k: (first["ids"][0], 0.95)
    try:
        second = memory.retain(OWNER, "semantic", "The deploy window moved to Tuesday 09:00 UTC")
    finally:
        memory._nearest_dup = orig
    assert second["deduped"] is False, second
    assert second["dedup_reason"] == "near_duplicate_linked", second
    assert second["supersedes"] == [first["ids"][0]]
    assert _count() == 2, "the newer near-duplicate must be stored"
    old = es("GET", f"/{idx('semantic')}/_doc/{first['ids'][0]}")["_source"]
    assert old["superseded_by"] == second["ids"][0]
    assert old["active"] is True, "near-dup link is coexistence, not a merge"


# ------------------------------------------------------------- integration

def test_same_topic_sessions_are_both_retrievable():
    """The regression that cost 11 of 17 paired losses: session B (same topic as A,
    different content) must be stored AND retrievable by B's own content."""
    _wipe()
    a = memory.retain(OWNER, "episodic", "user: I keep the standup notes in the wiki\n"
                                         "assistant: noted, wiki for standup notes")
    b = memory.retain(OWNER, "episodic", "user: I keep the standup notes in the wiki\n"
                                         "assistant: noted, wiki for standup notes\n"
                                         "user: also my flight to Lisbon is BA248 on 2024-03-02\n"
                                         "assistant: BA248 to Lisbon on 2024-03-02, saved")
    assert a["deduped"] is False and b["deduped"] is False
    assert b["_id"] != a["_id"]
    assert _count(kind="episodic") == 2
    got = memory.recall(OWNER, "flight to Lisbon BA248", kinds=["episodic"], size=8)
    texts = " ".join(r["text"] for r in got["results"])
    assert "BA248" in texts and "2024-03-02" in texts, texts


def test_backfill_note_text_hash_missing_on_legacy_docs():
    """Legacy docs (pre-fix) have no text_hash: they must still be retrievable and
    must not crash the dedup probe; the exact-hash path simply misses them."""
    _wipe()
    r = es("POST", f"/{idx('semantic')}/_doc?refresh=true",
           {"kind": "semantic", "owner_id": OWNER, "visibility": "private",
            "text": "Legacy doc with no text_hash field", "entities": [], "active": True,
            "occurred_at": "2024-01-01T00:00:00"})
    assert memory._dedup_lookup(OWNER, "semantic", "private", None, None) is None
    fresh = memory.retain(OWNER, "semantic", "Legacy doc with no text_hash field")
    assert fresh["deduped"] is False, "hash-miss on legacy docs must not raise"
    assert fresh["_id"] != r["_id"], "documents the known limitation: no backfill yet"
    got = memory.recall(OWNER, "legacy text_hash", kinds=["semantic"], size=5)
    assert len(got["results"]) >= 1
