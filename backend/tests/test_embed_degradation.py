"""P0-3: embed degradation is loud. A failed embed must surface in the retain/recall responses
and in /health (HTTP 200 + degraded:true; /health/ready is the 503 readiness probe); a healthy
embed must leave all of that clean (negative control)."""
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
    main._health_reset()
    yield
    main._health_reset()


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


def test_health_stays_200_but_reports_degraded_when_embed_fails(monkeypatch):
    """Consumers (Hermes plugin is_available, ames_doctor) treat non-2xx as 'unreachable'."""
    monkeypatch.setattr(embeddings, "embed", _boom)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True and body["embed_ok"] is False and body["vector_arm"] is False
    assert "403" in body["embed_error"]


def test_health_ready_503_when_embed_fails(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", _boom)
    r = client.get("/health/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["degraded"] is True and "403" in body["embed_error"]
    assert body["license_status"] is not None or "license_status" in body


def test_health_and_ready_200_when_embed_ok():
    for path in ("/health", "/health/ready"):
        r = client.get(path)
        assert r.status_code == 200, path
        body = r.json()
        assert body["degraded"] is False and body["embed_ok"] is True and body["ok"] is True
        assert "embed_error" not in body
        assert body["fusion"] in ("es_rrf", "app_rrf")
        assert isinstance(body["docs_without_vector_occurred_24h"], int)
        assert "license_status" in body and "kinds" in body


def test_health_counts_vectorless_docs_and_caches_the_whole_body(monkeypatch):
    base = client.get("/health").json()["docs_without_vector_occurred_24h"]
    monkeypatch.setattr(embeddings, "embed", _boom)
    _retain("vectorless doc for the health counter, unique token zyxwv")
    main._health_reset()
    probes, es_calls = [], []
    monkeypatch.setattr(embeddings, "embed", lambda t: probes.append(t) or [[0.0]])
    import app.store as store
    orig = store.es
    monkeypatch.setattr(store, "es", lambda *a, **k: es_calls.append(a[1]) or orig(*a, **k))
    h1 = client.get("/health").json()
    n_es = len(es_calls)
    client.get("/health")
    client.get("/health/ready")
    assert len(probes) == 1                 # embed probe cached
    assert len(es_calls) == n_es and n_es > 0   # license + counts cached with it, not re-queried
    assert h1["docs_without_vector_occurred_24h"] == base + 1


def test_health_probe_makes_one_attempt_with_a_short_timeout(monkeypatch):
    """A dead embed backend must not hold /health for the 300 s retry budget."""
    import urllib.error
    calls = []

    def dead(req, timeout=None):
        if "_inference" in req.full_url:
            calls.append(timeout)
        raise urllib.error.URLError("connection refused")      # transient => would be retried
    monkeypatch.setattr(embeddings, "_backend", "es")
    monkeypatch.setattr(embeddings.urllib.request, "urlopen", dead)
    monkeypatch.setenv("AMES_HTTP_RETRY_BUDGET_S", "300")
    import time
    t = time.monotonic()
    body = client.get("/health").json()
    assert time.monotonic() - t < 4
    assert calls == [embeddings.PROBE_TIMEOUT_S]            # exactly one attempt, 5 s cap
    assert body["degraded"] is True and "URLError" in body["embed_error"]


def test_stale_health_is_served_while_one_refresher_runs(monkeypatch):
    """The stale body is returned immediately; the lock is not held across the slow probe."""
    import threading
    import time
    client.get("/health")                                    # warm the cache (healthy)
    started, release = threading.Event(), threading.Event()

    def slow(texts):
        started.set()
        release.wait(5)
        raise RuntimeError("now failing")
    monkeypatch.setattr(embeddings, "embed", slow)
    main._health["at"] = time.monotonic() - main.HEALTH_TTL_S - 1     # expire the TTL
    t = time.monotonic()
    stale = client.get("/health").json()                     # must NOT wait for `slow`
    assert time.monotonic() - t < 1
    assert stale["degraded"] is False                        # stale value
    assert started.wait(2)
    second = client.get("/health").json()                    # refresher still running: no 2nd probe
    assert second["degraded"] is False
    release.set()
    for _ in range(50):
        if client.get("/health").json()["degraded"]:
            break
        time.sleep(0.05)
    assert client.get("/health").json()["degraded"] is True  # refreshed value lands
