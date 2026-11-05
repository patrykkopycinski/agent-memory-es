"""P0-3: embed degradation is loud. A failed embed must surface in the retain/recall responses
and flip /health to 503; a healthy embed must leave all of that clean (negative control)."""
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest
from fastapi.testclient import TestClient

from app import auth, embeddings, main, memory  # noqa: E402

client = TestClient(main.app)
KEY = auth.create_key("embed-degradation-owner")
H = {"X-API-Key": KEY}


def _boom(texts):
    raise RuntimeError("HTTP Error 403: Forbidden (license expired)")


@pytest.fixture(autouse=True)
def fresh_probe():
    main._embed_probe.update(at=0.0, ok=True, error=None)
    yield
    main._embed_probe.update(at=0.0, ok=True, error=None)


def _retain(text):
    text = f"{text} [{uuid.uuid4().hex[:8]}]"   # unique per run: exact-text dedup would skip the write
    r = client.post("/memory/retain", headers=H,
                    json={"kind": "semantic", "text": text, "visibility": "private"})
    assert r.status_code == 200, r.text
    return r.json()


def test_retain_reports_embed_failure(monkeypatch):
    before = embeddings.FAILURES["count"]
    monkeypatch.setattr(embeddings, "embed", _boom)
    out = _retain("degradation probe: harbour crane inspection log entry one")
    assert out["embedded"] is False
    assert "403" in out["embed_error"]
    assert embeddings.FAILURES["count"] == before + 1
    src = memory.es("GET", f"/{memory.idx('semantic')}/_doc/{out['_id']}")["_source"]
    assert "embedding" not in src          # the doc really is vector-less


def test_retain_healthy_embed_is_clean():
    out = _retain("degradation control: bakery opening hours on the high street")
    assert out["embedded"] is True
    assert "embed_error" not in out


def test_recall_marks_missing_vector_arm(monkeypatch):
    _retain("recall degradation: quarterly harbour crane maintenance schedule")
    monkeypatch.setattr(embeddings, "embed", _boom)
    r = client.post("/memory/recall", headers=H, json={"query": "harbour crane maintenance"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["vector_arm"] is False
    assert body["degraded"] == ["vector_arm"]
    assert "embed_error" in body
    assert body["results"], "BM25-only fallback must still return hits"


def test_recall_healthy_has_vector_arm():
    _retain("recall control: quarterly harbour crane maintenance schedule")
    body = client.post("/memory/recall", headers=H, json={"query": "harbour crane maintenance"}).json()
    assert body["vector_arm"] is True
    assert body["degraded"] == []
    assert "embed_error" not in body


def test_recall_log_line_carries_embed_error(monkeypatch, caplog):
    monkeypatch.setattr(embeddings, "embed", _boom)
    with caplog.at_level("INFO"):
        memory.recall("embed-degradation-owner", "harbour crane")
    assert any("ames.recall" in m and "embed_error" in m for m in caplog.messages)


def test_health_503_when_embed_fails(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", _boom)
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["embed_ok"] is False and body["ok"] is False
    assert "403" in body["embed_error"]


def test_health_200_when_embed_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["embed_ok"] is True
    assert body["fusion"] in ("es_rrf", "app_rrf")
    assert isinstance(body["docs_without_vector_24h"], int)
    assert "license_status" in body and "kinds" in body


def test_health_counts_vectorless_docs_and_caches_probe(monkeypatch):
    base = client.get("/health").json()["docs_without_vector_24h"]
    monkeypatch.setattr(embeddings, "embed", _boom)
    _retain("vectorless doc for the health counter, unique token zyxwv")
    calls = []
    monkeypatch.setattr(embeddings, "embed", lambda t: calls.append(t) or [[0.0]])
    main._embed_probe.update(at=0.0)
    h1 = client.get("/health").json()
    client.get("/health")
    assert len(calls) == 1                  # second hit served from the 60s cache
    assert h1["docs_without_vector_24h"] == base + 1
