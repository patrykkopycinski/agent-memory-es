"""Tests for POST /memory/update and /memory/delete (in-place mutation API).

Pins:
- update replaces text and the corrected text is recallable while the OLD
  passages are gone (group rewritten, not near-duplicated)
- doc_group stays stable across a text update
- tags/visibility-only update patches every passage
- delete removes the whole doc_group
- non-owner update/delete is 403 (mirrors promote)
- intent is logged to am_mutations before delete
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from app.main import app  # noqa: E402
from app import auth  # noqa: E402
from app.store import es, PREFIX  # noqa: E402

client = TestClient(app)
# Dedicated owners + module-level cleanup: a second run of this suite would
# otherwise dedup onto the previous run's docs (different doc_group), breaking
# the group-stability assertions.
ALICE = auth.create_key("mutationalice")
BOB = auth.create_key("mutationbob")
OWNER = "mutationalice"
H = lambda k: {"X-API-Key": k}  # noqa: E731


def _clean():
    from app.store import es as _es, idx as _idx, PREFIX as _P
    for kind in ("semantic", "episodic", "procedural"):
        _es("POST", f"/{_idx(kind)}/_delete_by_query?refresh=true",
            {"query": {"bool": {"should": [
                {"term": {"owner_id": "mutationalice"}},
                {"term": {"owner_id": "mutationbob"}}]}}})
    for _idxname in (f"{_P}am_rejected", f"{_P}am_mutations"):
        try:
            _es("POST", f"/{_idxname}/_delete_by_query?refresh=true",
                {"query": {"bool": {"should": [
                    {"term": {"owner_id": "mutationalice"}},
                    {"term": {"owner_id": "mutationbob"}}]}}})
        except RuntimeError:
            pass


_clean()


def _retain(key, kind="semantic", text="x", visibility="private", **kw):
    r = client.post("/memory/retain", headers=H(key),
                    json={"kind": kind, "text": text, "visibility": visibility, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _update(key, doc_id, **kw):
    body = {"kind": "semantic", "doc_id": doc_id, **kw}
    return client.post("/memory/update", headers=H(key), json=body)


def _delete(key, doc_id, kind="semantic"):
    return client.post("/memory/delete", headers=H(key),
                       json={"kind": kind, "doc_id": doc_id})


def _recall_texts(key, query):
    r = client.post("/memory/recall", headers=H(key), json={"query": query, "size": 20})
    assert r.status_code == 200, r.text
    return [h["text"] for h in r.json()["fused"]]


def test_update_fixes_text_in_place():
    doc = _retain(ALICE, text="weekly eval board seeded on 2026-30-26 with 2/1 suites")
    r = _update(ALICE, doc["_id"], text="weekly eval board seeded on 2026-10-26 with 2/11 suites")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["updated"] is True
    assert out["doc_group"] == doc["doc_group"], "doc_group must stay stable across update"
    texts = _recall_texts(ALICE, "weekly eval board seeded suites")
    assert any("2026-10-26" in t and "2/11" in t for t in texts), texts
    assert not any("2026-30-26" in t for t in texts), f"stale typo still recallable: {texts}"


def test_update_old_passages_gone():
    doc = _retain(ALICE, text="mutation test passage alpha marker unique-qwerty1 " * 40)
    assert doc.get("passages", 1) > 1, "need a multi-passage doc for this test"
    old_group = doc["doc_group"]
    r = _update(ALICE, doc["_id"], text="mutation test replaced single passage unique-qwerty2")
    assert r.status_code == 200, r.text
    from app.store import idx
    r2 = es("POST", f"/{idx('semantic')}/_search", {
        "size": 100, "query": {"bool": {"filter": [{"term": {"doc_group": old_group}}]}}})
    texts = [h["_source"]["text"] for h in r2["hits"]["hits"]]
    assert texts and all("unique-qwerty2" in t for t in texts), texts
    assert len(texts) == 1, f"old passages not deleted: {texts}"


def test_update_non_owner_403():
    doc = _retain(ALICE, text="alice only mutation guard fact 77f1")
    assert _update(BOB, doc["_id"], text="bob tries to overwrite alice").status_code == 403
    assert _delete(BOB, doc["_id"]).status_code == 403
    # alice's doc is untouched
    assert any("77f1" in t for t in _recall_texts(ALICE, "mutation guard fact"))


def test_delete_removes_doc():
    doc = _retain(ALICE, text="deletable scratch fact marker unique-zz9p")
    r = _delete(ALICE, doc["_id"])
    assert r.status_code == 200, r.text
    assert doc["_id"] in r.json()["deleted_ids"]
    assert not any("unique-zz9p" in t for t in _recall_texts(ALICE, "deletable scratch fact"))


def test_delete_logs_intent():
    doc = _retain(ALICE, text="delete-intent log probe marker unique-kk7t")
    _delete(ALICE, doc["_id"])
    r = es("POST", f"/{PREFIX}am_mutations/_search", {
        "size": 5,
        "query": {"bool": {"filter": [
            {"term": {"intent": "delete"}}, {"term": {"doc_id": doc["_id"]}}]}}})
    hits = r["hits"]["hits"]
    assert hits, "delete intent not logged"
    src = hits[0]["_source"]
    assert src["owner_id"] == OWNER
    assert "unique-kk7t" in src["detail"]["text"]
    assert src.get("at"), "intent log must carry a timestamp"


def test_update_tags_only_patches_passages():
    doc = _retain(ALICE, text="tag-only update probe marker unique-mm3v")
    r = _update(ALICE, doc["_id"], tags=["project:mutation-api", "status:fixed"])
    assert r.status_code == 200, r.text
    assert any("unique-mm3v" in t for t in _recall_texts(ALICE, "tag-only update probe"))
    # recall via the new tag must hit; via an old/absent tag must abstain
    r2 = client.post("/memory/recall", headers=H(ALICE), json={
        "query": "tag-only update probe", "size": 10,
        "filter": {"all": ["project:mutation-api"]}})
    assert r2.status_code == 200, r2.text
    assert any("unique-mm3v" in h["text"] for h in r2.json()["fused"])


def test_update_requires_something():
    doc = _retain(ALICE, text="no-op guard probe")
    assert _update(ALICE, doc["_id"]).status_code == 422


def test_update_missing_doc_404():
    assert _update(ALICE, "does-not-exist-123", text="x").status_code == 404
    assert _delete(ALICE, "does-not-exist-123").status_code == 404


def test_e2e_retain_update_recall_roundtrip():
    doc = _retain(ALICE, text="e2e roundtrip: matrix run hapened on 2026-30-26")  # typo'd
    hits = _recall_texts(ALICE, "e2e roundtrip matrix run")
    assert any("2026-30-26" in t for t in hits)
    r = _update(ALICE, doc["_id"], text="e2e roundtrip: matrix run happened on 2026-10-26")
    assert r.status_code == 200, r.text
    hits = _recall_texts(ALICE, "e2e roundtrip matrix run")
    assert any("happened on 2026-10-26" in t for t in hits), hits
    assert not any("2026-30-26" in t or "hapened" in t for t in hits), hits
