"""DELETE /memory/{kind}/{doc_id}: owner-check, group cascade, tombstone, 404."""
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

# Rerunnability on a persistent ES: deletes tombstone the exact test texts, so
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


def test_delete_tombstones_text():
    text = "tombstone after delete unique qz3"
    m = _retain(ALICE, text=text)
    assert client.delete(f"/memory/semantic/{m['_id']}", headers=H(ALICE)).status_code == 200
    again = client.post("/memory/retain", headers=H(ALICE),
                        json={"kind": "semantic", "text": text, "extract_facts": False})
    assert again.status_code == 422
    assert "rejected value" in again.json()["detail"]


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
