"""`_get_source` must tell a missing document (404, `found:false`) from a broken store (404,
`index_not_found_exception`). The first is `LookupError` -> client 404 "no such doc"; the second
must surface as a server error, not as "no such doc"."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest
from fastapi.testclient import TestClient

from app import auth, memory
from app.main import app

INDEX_MISSING = RuntimeError(
    'ES GET /am_semantic/_doc/k -> 404: {"error":{"root_cause":[{"type":"index_not_found_exception",'
    '"reason":"no such index [am_semantic]"}],"type":"index_not_found_exception"},"status":404}')
DOC_MISS = RuntimeError('ES GET /am_semantic/_doc/k -> 404: {"_index":"am_semantic","_id":"k","found":false}')

client = TestClient(app, raise_server_exceptions=False)
KEY = auth.create_key("getsource-owner")
H = {"X-API-Key": KEY}


def _es_raising(exc):
    def fake(*a, **k):
        raise exc
    return fake


def test_index_level_404_is_not_reported_as_no_such_doc(monkeypatch):
    monkeypatch.setattr(memory, "es", _es_raising(INDEX_MISSING))
    with pytest.raises(RuntimeError, match="index_not_found_exception"):
        memory._get_source("semantic", "k")


def test_doc_level_miss_is_a_lookup_error(monkeypatch):
    monkeypatch.setattr(memory, "es", _es_raising(DOC_MISS))
    with pytest.raises(LookupError, match="no such semantic doc"):
        memory._get_source("semantic", "k")


def test_edge_index_level_404_is_a_server_error_on_update_and_delete(monkeypatch):
    monkeypatch.setattr(memory, "es", _es_raising(INDEX_MISSING))
    for call in (lambda: client.post("/memory/update", headers=H, json={"kind": "semantic", "doc_id": "k", "text": "x"}),
                 lambda: client.post("/memory/delete", headers=H, json={"kind": "semantic", "doc_id": "k"}),
                 lambda: client.delete("/memory/semantic/k", headers=H)):
        assert call().status_code == 500


def test_edge_doc_level_miss_is_still_a_404(monkeypatch):
    monkeypatch.setattr(memory, "es", _es_raising(DOC_MISS))
    for call in (lambda: client.post("/memory/update", headers=H, json={"kind": "semantic", "doc_id": "k", "text": "x"}),
                 lambda: client.post("/memory/delete", headers=H, json={"kind": "semantic", "doc_id": "k"}),
                 lambda: client.delete("/memory/semantic/k", headers=H)):
        assert call().status_code == 404
