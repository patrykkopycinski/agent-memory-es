"""Tenant isolation, write-guard and request-bound regressions for the mutation API.

Pins:
- doc_group is caller-chosen, so it is not a tenant boundary: two owners may share
  a group name, but neither can delete or rewrite the other's passages through it
- /memory/update applies promote()'s marker guard when widening visibility
  (patch branch and text-rewrite branch)
- /admin/keys has no default token and refuses an empty one
- ids with '?' or '/' cannot add ES query params or path segments
- recall rejects unknown kinds and clamps size; retain caps text length
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401,E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app import auth, memory  # noqa: E402
from app.store import es, idx  # noqa: E402

client = TestClient(app)
VICTIM, MALLORY = "tgvictim", "tgmallory"
VK, MK = auth.create_key(VICTIM), auth.create_key(MALLORY)
H = lambda k: {"X-API-Key": k}  # noqa: E731


def _clean():
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"terms": {"owner_id": [VICTIM, MALLORY]}}})
    from app.store import PREFIX
    for name in ("am_rejected", "am_mutations"):
        try:
            es("POST", f"/{PREFIX}{name}/_delete_by_query?refresh=true",
               {"query": {"terms": {"owner_id": [VICTIM, MALLORY]}}})
        except RuntimeError:
            pass


_clean()


@pytest.fixture(autouse=True)
def _no_embedder(monkeypatch):
    # Nothing here is about vectors: fail embedding fast so retain/recall take the
    # BM25-only path instead of waiting on an inference endpoint.
    def down(_texts):
        raise RuntimeError("embedder down")
    monkeypatch.setattr("app.embeddings.embed", down)


def _retain(key, text, **kw):
    r = client.post("/memory/retain", headers=H(key),
                    json={"kind": "semantic", "text": text, **kw})
    return r


def _group_owners(group):
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": 100, "query": {"term": {"doc_group": group}}})
    return sorted(h["_source"]["owner_id"] for h in r["hits"]["hits"])


def test_joining_a_foreign_doc_group_cannot_delete_the_victims_passages():
    # The reported attack: retain into the victim's group name, then delete your own doc.
    v = _retain(VK, "victim private note in group tg-a1", doc_group="tg-grp-a1")
    assert v.status_code == 200, v.text
    m = _retain(MK, "mallory joins tg-a1", doc_group="tg-grp-a1")
    assert m.status_code == 200, m.text
    r = client.post("/memory/delete", headers=H(MK),
                    json={"kind": "semantic", "doc_id": m.json()["_id"]})
    assert r.status_code == 200, r.text
    assert _group_owners("tg-grp-a1") == [VICTIM]


def test_delete_cannot_reach_another_owners_passages():
    # Seed the pre-fix state directly (a shared group already exists in ES):
    # the delete must stay inside the caller's own passages.
    for owner, text in ((VICTIM, "victim passage tg-b2"), (MALLORY, "mallory passage tg-b2")):
        es("POST", f"/{idx('semantic')}/_doc?refresh=true", {
            "kind": "semantic", "owner_id": owner, "visibility": "private", "text": text,
            "doc_group": "tg-grp-b2", "passage_index": 0, "passages_total": 1, "active": True})
    mine = es("POST", f"/{idx('semantic')}/_search", {"query": {"bool": {"filter": [
        {"term": {"doc_group": "tg-grp-b2"}}, {"term": {"owner_id": MALLORY}}]}}})["hits"]["hits"]
    r = client.post("/memory/delete", headers=H(MK), json={"kind": "semantic", "doc_id": mine[0]["_id"]})
    assert r.status_code == 200, r.text
    assert _group_owners("tg-grp-b2") == [VICTIM]


def test_update_cannot_rewrite_another_owners_passages():
    for owner, i in ((VICTIM, 0), (MALLORY, 0), (MALLORY, 1)):
        es("POST", f"/{idx('semantic')}/_doc?refresh=true", {
            "kind": "semantic", "owner_id": owner, "visibility": "private",
            "text": f"{owner} passage {i} tg-c3", "doc_group": "tg-grp-c3",
            "passage_index": i, "passages_total": 2, "active": True})
    mine = es("POST", f"/{idx('semantic')}/_search", {"query": {"bool": {"filter": [
        {"term": {"doc_group": "tg-grp-c3"}}, {"term": {"owner_id": MALLORY}}]}}})["hits"]["hits"]
    r = client.post("/memory/update", headers=H(MK),
                    json={"kind": "semantic", "doc_id": mine[0]["_id"], "text": "rewritten tg-c3"})
    # mallory's two passages collapse into her rewrite; the victim's passage is untouched
    assert r.status_code == 200, r.text
    assert _group_owners("tg-grp-c3") == [MALLORY, VICTIM]
    victim = es("POST", f"/{idx('semantic')}/_search", {"query": {"bool": {"filter": [
        {"term": {"doc_group": "tg-grp-c3"}}, {"term": {"owner_id": VICTIM}}]}}})["hits"]["hits"]
    assert victim[0]["_source"]["text"] == f"{VICTIM} passage 0 tg-c3"


def test_same_named_groups_of_two_owners_have_separate_per_doc_budgets():
    from app.memory import _apply_budgets
    hits = [{"id": f"{o}{i}", "kind": "episodic", "owner_id": o, "doc_group": "sess-0",
             "score": 10 - i} for o in ("a", "b") for i in range(3)]
    picked = _apply_budgets(sorted(hits, key=lambda h: -h["score"]), 4, per_doc_budget=2)
    assert sorted(h["owner_id"] for h in picked) == ["a", "a", "b", "b"]


def test_update_widening_visibility_hits_promote_guard():
    doc = _retain(VK, "the staging api key lives in vault path tg-d4").json()
    r = client.post("/memory/update", headers=H(VK),
                    json={"kind": "semantic", "doc_id": doc["_id"], "visibility": "team"})
    assert r.status_code == 422 and "sensitive marker" in r.text, r.text
    src = es("GET", f"/{idx('semantic')}/_doc/{doc['_id']}")["_source"]
    assert src["visibility"] == "private"
    # promote refuses the same doc: update must agree with it
    p = client.post("/memory/promote", headers=H(VK),
                    json={"kind": "semantic", "doc_id": doc["_id"], "to_visibility": "team"})
    assert p.status_code == 422, p.text


def test_update_text_rewrite_while_widening_is_guarded():
    doc = _retain(VK, "harmless note tg-e5").json()
    r = client.post("/memory/update", headers=H(VK), json={
        "kind": "semantic", "doc_id": doc["_id"], "visibility": "common",
        "text": "the prod admin token is in 1password tg-e5"})
    assert r.status_code == 422 and "sensitive marker" in r.text, r.text


def test_update_widening_clean_text_still_works():
    doc = _retain(VK, "team convention: evals run nightly tg-f6").json()
    r = client.post("/memory/update", headers=H(VK),
                    json={"kind": "semantic", "doc_id": doc["_id"], "visibility": "team"})
    assert r.status_code == 200, r.text
    assert es("GET", f"/{idx('semantic')}/_doc/{doc['_id']}")["_source"]["visibility"] == "team"


def test_admin_keys_has_no_default_token(monkeypatch):
    monkeypatch.delenv("AMES_ADMIN_TOKEN", raising=False)
    assert client.post("/admin/keys?owner_id=x", headers={"X-Admin-Token": "dev-admin"}).status_code == 403
    monkeypatch.setenv("AMES_ADMIN_TOKEN", "")
    assert client.post("/admin/keys?owner_id=x", headers={"X-Admin-Token": ""}).status_code == 403
    assert client.post("/admin/keys?owner_id=x").status_code == 403
    monkeypatch.setenv("AMES_ADMIN_TOKEN", "s3cret-tg")
    assert client.post("/admin/keys?owner_id=x", headers={"X-Admin-Token": "dev-admin"}).status_code == 403
    ok = client.post("/admin/keys?owner_id=tgminted", headers={"X-Admin-Token": "s3cret-tg"})
    assert ok.status_code == 200 and ok.json()["owner_id"] == "tgminted"


def test_doc_id_cannot_inject_es_params():
    doc = _retain(VK, "param injection probe tg-g7").json()
    # '?_source_includes=owner_id' would otherwise be parsed by ES as a param and
    # return the doc; encoded, it is just a non-existent id
    r = client.post("/memory/delete", headers=H(VK), json={
        "kind": "semantic", "doc_id": doc["_id"] + "?_source_includes=owner_id"})
    assert r.status_code == 404, r.text
    r = client.post("/memory/delete", headers=H(VK), json={
        "kind": "semantic", "doc_id": "../../" + idx("semantic") + "/_doc/" + doc["_id"]})
    assert r.status_code == 404, r.text
    assert es("GET", f"/{idx('semantic')}/_doc/{doc['_id']}")["found"]


def test_fact_job_retry_rejects_non_hex_ids(monkeypatch):
    # ES would 404 'abc' anyway, so assert the id never reaches an ES path at all
    from app import facts
    paths = []
    monkeypatch.setattr(facts, "es", lambda m, p, b=None: paths.append(p) or {})
    r = client.post("/memory/facts/jobs/abc%3Fpretty%3Dtrue/retry", headers=H(VK))
    assert r.status_code == 404, r.text
    assert paths == [], paths


def test_recall_rejects_unknown_kinds_and_clamps_size():
    r = client.post("/memory/recall", headers=H(VK),
                    json={"query": "x", "kinds": ["semantic", "am_rejected"]})
    assert r.status_code == 422, r.text
    seen = []
    real = memory.es

    def spy(method, path, body=None):
        if path.endswith("/_search") and body and "size" in body:
            seen.append((path, body["size"]))
        return real(method, path, body)
    memory.es = spy
    try:
        r = client.post("/memory/recall", headers=H(VK),
                        json={"query": "x", "kinds": ["semantic"] * 500, "size": 5_000_000})
    finally:
        memory.es = real
    assert r.status_code == 200, r.text
    searched = [p for p, _ in seen if idx("semantic") in p]
    assert len(searched) == 1, searched
    assert max(s for _, s in seen) <= memory.MAX_RECALL_SIZE * 5, seen


def test_retain_caps_text_length():
    r = _retain(VK, "x" * (memory.MAX_TEXT_CHARS + 1))
    assert r.status_code == 422, r.text[:200]


def teardown_module(_m):
    _clean()
