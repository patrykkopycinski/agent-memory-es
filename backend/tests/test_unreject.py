"""POST /memory/unreject: owner-only lift of tombstones (by text with the retain-time matching
rule, or by source_id). Own owners, wiped at import, so reruns on a persistent ES stay green."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (guard: refuses anything but a throwaway ES)

import pytest
from fastapi.testclient import TestClient

from app import auth, memory, tombstone
from app.main import app
from app.store import es, idx, PREFIX

client = TestClient(app)
OWNERS = ("ur_alice", "ur_bob")
ALICE, BOB = (auth.create_key(o) for o in OWNERS)
H = lambda k: {"X-API-Key": k}  # noqa: E731

tombstone._ensure_index()
for _name in (f"{PREFIX}am_rejected", f"{PREFIX}am_mutations"):
    try:
        es("POST", f"/{_name}/_delete_by_query?refresh=true&ignore_unavailable=true",
           {"query": {"terms": {"owner_id": list(OWNERS)}}})
    except RuntimeError:
        pass
es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true",
   {"query": {"terms": {"owner_id": list(OWNERS)}}})


def _retain(key, text):
    return client.post("/memory/retain", headers=H(key),
                       json={"kind": "semantic", "text": text, "extract_facts": False})


def _retract(key, text):
    m = _retain(key, text)
    assert m.status_code == 200, m.text
    doc_id = m.json()["_id"]
    d = client.post("/memory/delete", headers=H(key),
                    json={"kind": "semantic", "doc_id": doc_id, "retract": True})
    assert d.status_code == 200 and d.json()["tombstoned"] is True, d.text
    return doc_id


def _unreject(key, **body):
    return client.post("/memory/unreject", headers=H(key), json=body)


def _tombstone_count(owner):
    return es("POST", f"/{PREFIX}am_rejected/_search",
              {"size": 0, "track_total_hits": True,
               "query": {"term": {"owner_id": owner}}})["hits"]["total"]["value"]


def test_unreject_exact_text_then_retain_succeeds():
    text = "unreject exact text unique ur1 alpha"
    doc_id = _retract(ALICE, text)
    assert _retain(ALICE, text).status_code == 422
    r = _unreject(ALICE, text=text)
    assert r.status_code == 200 and r.json()["lifted"] == 1, r.text
    assert r.json()["source_ids"] == [doc_id]
    assert _retain(ALICE, text).status_code == 200


def test_unreject_fuzzy_reworded_text_lifts_the_matching_tombstone():
    original = "ur2 deploys release pipeline staging cluster weekly"
    doc_id = _retract(ALICE, original)
    reworded = "weekly ur2 release pipeline deploys staging cluster nightly"
    assert tombstone.is_rejected(OWNERS[0], reworded) is not None
    assert _retain(ALICE, reworded).status_code == 422
    r = _unreject(ALICE, text=reworded)
    assert r.status_code == 200 and r.json()["lifted"] == 1, r.text
    assert r.json()["source_ids"] == [doc_id]
    assert tombstone.is_rejected(OWNERS[0], reworded) is None
    assert _retain(ALICE, original).status_code == 200


def test_unreject_by_source_id_lifts_only_that_docs_tombstone():
    keep = _retract(ALICE, "ur3 keeps its tombstone beta gamma")
    lift = _retract(ALICE, "ur3 loses its tombstone delta epsilon")
    r = _unreject(ALICE, source_id=lift)
    assert r.status_code == 200 and r.json()["lifted"] == 1, r.text
    assert r.json()["source_ids"] == [lift]
    assert _retain(ALICE, "ur3 loses its tombstone delta epsilon").status_code == 200
    assert _retain(ALICE, "ur3 keeps its tombstone beta gamma").status_code == 422
    assert keep != lift


def test_unreject_nothing_to_lift_is_200_with_zero():
    for body in ({"text": "ur4 never rejected anything like this"},
                 {"source_id": "no-such-doc-id"}):
        r = _unreject(ALICE, **body)
        assert r.status_code == 200 and r.json()["lifted"] == 0, (body, r.text)
        assert r.json()["source_ids"] == []


@pytest.mark.parametrize("body", [
    {}, {"text": "ur5 both selectors", "source_id": "x"}, {"text": "  "}, {"source_id": ""},
])
def test_unreject_needs_exactly_one_selector(body):
    assert _unreject(ALICE, **body).status_code == 422


def test_unreject_source_id_is_length_bounded():
    ok = "s" * memory.MAX_ID_CHARS
    assert _unreject(ALICE, source_id=ok).status_code == 200
    over = "s" * (memory.MAX_ID_CHARS + 1)
    r = _unreject(ALICE, source_id=over)
    assert r.status_code == 422 and "source_id too long" in r.json()["detail"], r.text
    r = client.post("/mcp/tools/unreject", headers=H(ALICE), json={"source_id": over})
    assert r.status_code == 422 and "source_id too long" in r.json()["detail"], r.text
    logged = es("POST", f"/{PREFIX}am_mutations/_search",
                {"query": {"term": {"doc_id": over}}})["hits"]["total"]["value"]
    assert logged == 0, "an over-limit source_id must not reach the mutation log"


def test_unreject_text_is_size_bounded():
    r = _unreject(ALICE, text="x" * (memory.MAX_TEXT_CHARS + 1))
    assert r.status_code == 422 and "too long" in r.json()["detail"]


def test_unreject_never_touches_another_owners_tombstone():
    text = "ur6 isolation between owners unique zeta"
    doc_id = _retract(ALICE, text)
    _retract(BOB, text)
    for body in ({"text": text}, {"source_id": doc_id}):
        r = _unreject(BOB, **body)
        assert r.status_code == 200, r.text
    assert _tombstone_count(OWNERS[0]) >= 1
    assert _retain(ALICE, text).status_code == 422
    assert _retain(BOB, text).status_code == 200


def test_unreject_logs_intent_before_lifting():
    text = "ur7 mutation log entry unique eta theta"
    doc_id = _retract(ALICE, text)
    assert _unreject(ALICE, text=text).status_code == 200
    hits = es("POST", f"/{PREFIX}am_mutations/_search",
              {"query": {"bool": {"filter": [{"term": {"intent": "unreject"}},
                                              {"term": {"owner_id": OWNERS[0]}}]}}})["hits"]["hits"]
    mine = [h["_source"]["detail"] for h in hits if h["_source"]["detail"].get("text") == text]
    assert mine and mine[0]["source_ids"] == [doc_id] and mine[0]["matched"] == 1, hits


def test_unreject_lists_value_key_for_tombstone_without_source_id():
    text = "ur10 tombstone without source unique lambda mu"
    tombstone.reject(OWNERS[0], text, "written directly")
    r = _unreject(ALICE, text=text)
    assert r.status_code == 200 and r.json()["lifted"] == 1, r.text
    assert r.json()["source_ids"] == [tombstone._key(tombstone._terms(text))]


def test_unreject_tools_dispatcher():
    text = "ur8 tools dispatcher unique iota kappa"
    _retract(ALICE, text)
    r = client.post("/mcp/tools/unreject", headers=H(ALICE), json={"text": text})
    assert r.status_code == 200 and r.json()["result"]["lifted"] == 1, r.text
    assert client.post("/mcp/tools/unreject", headers=H(ALICE), json={}).status_code == 422


def test_unreject_scan_cap_fails_closed_instead_of_lifting_partially(monkeypatch):
    o = OWNERS[1]
    for i in range(12):
        tombstone.reject(o, f"ur9 filler{i:03d} unrelated{i:03d} noise{i:03d}", "filler")
    monkeypatch.setattr(tombstone, "PAGE", 5)
    monkeypatch.setattr(tombstone, "TERMS_CHUNK", 3)
    monkeypatch.setattr(tombstone, "MAX_PAGES", 2)
    before = _tombstone_count(o)
    r = _unreject(BOB, text="ur9 banana smoothie recipe probe")
    assert r.status_code == 422 and "candidate tombstones" in r.json()["detail"], r.text
    assert _tombstone_count(o) == before


def test_unreject_without_rejected_index_lifts_zero(monkeypatch):
    def missing(method, path, body=None):
        raise RuntimeError(f"ES {method} {path} -> 404: index_not_found_exception")
    monkeypatch.setattr(tombstone, "es", missing)
    assert tombstone.lift("ur_alice", ["x"]) == 0
    assert tombstone.by_source("ur_alice", "x") == []
