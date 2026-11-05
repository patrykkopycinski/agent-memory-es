"""DELETE /memory/{kind}/{doc_id}: owner-check, group cascade, remove vs retract, 404."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from app.main import app  # noqa: E402
from app import auth  # noqa: E402

client = TestClient(app)
ALICE = auth.create_key("alice")
BOB = auth.create_key("bob")
H = lambda k: {"X-API-Key": k}  # noqa: E731

from app.store import es, PREFIX, idx  # noqa: E402

# Rerunnability on a persistent ES: retracting deletes tombstone the exact test texts, so
# a second run's retain of the same text would 422. Wipe our owners' docs and
# tombstones at import (both suites share the alice/bob owners).
for kind in ("semantic", "episodic", "procedural"):
    es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
       {"query": {"bool": {"should": [
           {"term": {"owner_id": "alice"}}, {"term": {"owner_id": "bob"}}]}}})
try:
    es("POST", f"/{PREFIX}am_rejected/_delete_by_query?refresh=true",
       {"query": {"bool": {"should": [
           {"term": {"owner_id": "alice"}}, {"term": {"owner_id": "bob"}}]}}})
except RuntimeError:
    pass


def _retain(key, kind="semantic", text="x", doc_group=None):
    r = client.post("/memory/retain", headers=H(key),
                    json={"kind": kind, "text": text, "doc_group": doc_group,
                          "extract_facts": False})
    assert r.status_code == 200, r.text
    return r.json()


def test_delete_owner_ok_and_gone():
    m = _retain(ALICE, text="delete-me owner check unique qz1")
    d = client.delete(f"/memory/semantic/{m['_id']}", headers=H(ALICE))
    assert d.status_code == 200, d.text
    assert d.json()["deleted"] is True
    gone = client.delete(f"/memory/semantic/{m['_id']}", headers=H(ALICE))
    assert gone.status_code == 404


def test_delete_cross_owner_forbidden():
    m = _retain(ALICE, text="alice private delete qz2")
    d = client.delete(f"/memory/semantic/{m['_id']}", headers=H(BOB))
    assert d.status_code == 403
    # alice still can
    assert client.delete(f"/memory/semantic/{m['_id']}", headers=H(ALICE)).status_code == 200


def test_delete_unknown_kind_422():
    assert client.delete("/memory/bogus/whatever", headers=H(ALICE)).status_code == 422


def _retain_status(key, text):
    return client.post("/memory/retain", headers=H(key),
                       json={"kind": "semantic", "text": text, "extract_facts": False})


def test_plain_delete_allows_reretain():
    text = "plain delete removes only unique qz3"
    m = _retain(ALICE, text=text)
    d = client.delete(f"/memory/semantic/{m['_id']}", headers=H(ALICE))
    assert d.status_code == 200 and d.json()["tombstoned"] is False
    assert _retain_status(ALICE, text).status_code == 200


def test_retract_delete_tombstones_text():
    text = "tombstone after retract unique qz3r"
    m = _retain(ALICE, text=text)
    d = client.delete(f"/memory/semantic/{m['_id']}?retract=true", headers=H(ALICE))
    assert d.status_code == 200 and d.json()["tombstoned"] is True
    again = _retain_status(ALICE, text)
    assert again.status_code == 422
    assert "rejected value" in again.json()["detail"]


def test_retract_flag_on_every_entry_point():
    post = _retain(ALICE, text="retract via post body unique qz9a")
    r = client.post("/memory/delete", headers=H(ALICE),
                    json={"kind": "semantic", "doc_id": post["_id"], "retract": True})
    assert r.status_code == 200 and r.json()["tombstoned"] is True
    assert _tombstones_for(post["_id"]) == 1

    tool = _retain(ALICE, text="retract via tools dispatcher unique qz9b")
    r = client.post("/mcp/tools/delete", headers=H(ALICE),
                    json={"kind": "semantic", "doc_id": tool["_id"], "retract": True})
    assert r.status_code == 200 and r.json()["result"]["tombstoned"] is True
    assert _tombstones_for(tool["_id"]) == 1

    for text, call in (
        ("default via post body unique qz9c", lambda i: client.post(
            "/memory/delete", headers=H(ALICE), json={"kind": "semantic", "doc_id": i})),
        ("default via tools dispatcher unique qz9d", lambda i: client.post(
            "/mcp/tools/delete", headers=H(ALICE), json={"kind": "semantic", "doc_id": i})),
        ("explicit false via query unique qz9e", lambda i: client.delete(
            f"/memory/semantic/{i}?retract=false", headers=H(ALICE))),
    ):
        m = _retain(ALICE, text=text)
        assert call(m["_id"]).status_code == 200
        assert _tombstones_for(m["_id"]) == 0, text


def test_tools_delete_rejects_non_bool_retract():
    m = _retain(ALICE, text="non bool retract unique qz9f")
    r = client.post("/mcp/tools/delete", headers=H(ALICE),
                    json={"kind": "semantic", "doc_id": m["_id"], "retract": "yes"})
    assert r.status_code == 422, r.text
    assert client.delete(f"/memory/semantic/{m['_id']}", headers=H(ALICE)).status_code == 200


def test_delete_logs_retract_in_mutation_log():
    m = _retain(ALICE, text="mutation log retract field unique qz9g")
    client.delete(f"/memory/semantic/{m['_id']}?retract=true", headers=H(ALICE))
    hit = es("POST", f"/{PREFIX}am_mutations/_search",
             {"query": {"term": {"doc_id": m["_id"]}}})["hits"]["hits"][0]["_source"]
    assert hit["intent"] == "delete" and hit["detail"]["retract"] is True


def _tombstones_for(doc_id):
    r = es("POST", f"/{PREFIX}am_rejected/_search",
           {"size": 5, "query": {"term": {"source_id": doc_id}}})
    return r["hits"]["total"]["value"]


def test_delete_tombstone_semantic_only():
    plain = _retain(ALICE, text="semantic plain delete flag unique qz6")
    d = client.delete(f"/memory/semantic/{plain['_id']}", headers=H(ALICE)).json()
    assert d["tombstoned"] is False and _tombstones_for(plain["_id"]) == 0
    sem = _retain(ALICE, text="semantic tombstone flag unique qz7")
    d = client.delete(f"/memory/semantic/{sem['_id']}?retract=true", headers=H(ALICE)).json()
    assert d["tombstoned"] is True
    assert _tombstones_for(sem["_id"]) == 1
    for kind in ("episodic", "procedural"):
        m = _retain(ALICE, kind=kind, text=f"{kind} no tombstone unique qz8")
        d = client.delete(f"/memory/{kind}/{m['_id']}?retract=true", headers=H(ALICE)).json()
        assert d["deleted"] is True and d["tombstoned"] is False
        assert _tombstones_for(m["_id"]) == 0


def test_delete_termless_text_writes_no_tombstone():
    m = _retain(ALICE, text="ok")
    d = client.delete(f"/memory/semantic/{m['_id']}?retract=true", headers=H(ALICE)).json()
    assert d["deleted"] is True and d["tombstoned"] is False
    assert _tombstones_for(m["_id"]) == 0
    again = client.post("/memory/retain", headers=H(ALICE),
                        json={"kind": "semantic", "text": "no", "extract_facts": False})
    assert again.status_code == 200, again.text


def test_delete_group_cascade():
    """Multi-passage retain shares doc_group: delete head deletes all passages."""
    filler = " ".join(f"passage filler {i}" for i in range(200))  # > CHUNK_TARGET_CHARS
    m = _retain(ALICE, text=f"head doc unique qz4 {filler}", doc_group="dg_test_del_cascade")
    assert m["passages"] >= 2, m
    assert m["doc_group"] == "dg_test_del_cascade"
    d = client.delete(f"/memory/semantic/{m['_id']}", headers=H(ALICE))
    assert d.status_code == 200, d.text
    assert d.json()["group_deleted"] is True
    # every passage id 404s through the app layer
    for pid in m["ids"]:
        again = client.delete(f"/memory/semantic/{pid}", headers=H(ALICE))
        assert again.status_code == 404, (pid, again.text)
