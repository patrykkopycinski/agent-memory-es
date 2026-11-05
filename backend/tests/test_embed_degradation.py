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


def test_update_reports_embed_failure_and_healthy_is_clean(monkeypatch):
    doc = _retain("update degradation: ferry timetable for the north pier")
    ok = client.post("/memory/update", headers=H, json={
        "kind": "semantic", "doc_id": doc["_id"], "text": "update control: ferry timetable north pier v2 "
                                                         + uuid.uuid4().hex})
    assert ok.status_code == 200, ok.text
    assert ok.json()["embedded"] is True and "embed_error" not in ok.json()
    monkeypatch.setattr(embeddings, "embed", _boom)
    bad = client.post("/memory/update", headers=H, json={
        "kind": "semantic", "doc_id": ok.json()["_id"], "text": "update degraded: ferry timetable north pier v3 "
                                                              + uuid.uuid4().hex})
    assert bad.status_code == 200, bad.text
    assert bad.json()["embedded"] is False and "403" in bad.json()["embed_error"]


def _owner():
    return "embed-degradation-owner"


def test_reflect_passes_through_recall_degradation(monkeypatch):
    _retain("reflect degradation: lighthouse keeper rota for the winter season")
    monkeypatch.setattr(embeddings, "embed", _boom)
    body = memory.reflect(_owner(), "lighthouse keeper rota", llm_answer_fn=False, rounds=1)
    assert body["degraded"] == ["vector_arm"] and "403" in body["embed_error"]


def test_reflect_healthy_is_clean():
    _retain("reflect control: lighthouse keeper rota for the winter season")
    body = memory.reflect(_owner(), "lighthouse keeper rota", llm_answer_fn=False, rounds=1)
    assert body["degraded"] == [] and "embed_error" not in body


def test_idle_gap_past_two_ttls_computes_inline_not_stale(monkeypatch):
    """A one-shot caller after a long idle gap must get the fresh verdict, not the old body."""
    import time
    client.get("/health")                                    # healthy verdict cached
    monkeypatch.setattr(embeddings, "embed", _boom)          # backend now broken
    main._health["at"] = time.monotonic() - 10 * main.HEALTH_TTL_S
    assert client.get("/health").json()["degraded"] is True
    main._health["at"] = time.monotonic() - 10 * main.HEALTH_TTL_S
    assert client.get("/health/ready").status_code == 503


def test_refresher_thread_failure_serves_the_stale_body_and_releases_the_lock(monkeypatch):
    """Thread.start() can raise under thread exhaustion. /health must stay 200 with the stale body
    (not 500), and the lock must not leak or no refresh would ever run again."""
    import threading
    import time
    first = client.get("/health").json()
    main._health["at"] = time.monotonic() - main.HEALTH_TTL_S - 1     # stale but < 2*TTL

    def no_start(self):
        raise RuntimeError("can't start new thread")
    monkeypatch.setattr(threading.Thread, "start", no_start)
    # call the cached builder directly: TestClient itself needs Thread.start
    assert main._health_cached() == first                            # no exception: the handler stays 200
    leaked = not main._health_refresh_lock.acquire(blocking=False)
    main._health_refresh_lock.release()           # held either by us or by the leak: free it so later tests run
    assert not leaked, "lock leaked"
    main._health["at"] = time.monotonic() - main.HEALTH_TTL_S - 1
    assert main._health_cached() == first                            # and it keeps working next time


def test_refresher_thread_failure_with_no_cached_body_computes_inline(monkeypatch):
    import threading
    monkeypatch.setattr(threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("no threads")))
    assert main._health_cached()["ok"] is True                        # nothing cached: inline, no thread needed


def test_startup_warms_the_local_model_and_health_cache_without_blocking(monkeypatch):
    import threading
    calls, gate = [], threading.Event()
    monkeypatch.setattr(embeddings, "backend", lambda: "local")
    monkeypatch.setattr(embeddings, "_local_model", lambda: (gate.wait(5), calls.append("model")))
    monkeypatch.setattr(main, "_health_cached", lambda: calls.append("health"))
    import time
    t = time.monotonic()
    main._warm_embedder_in_background()
    assert time.monotonic() - t < 1                           # startup does not wait for the load
    assert calls == []
    gate.set()
    for _ in range(50):
        if calls == ["model", "health"]:
            break
        time.sleep(0.05)
    assert calls == ["model", "health"]


def test_warmup_failure_never_propagates_and_skips_model_for_other_backends(monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "backend", lambda: "es")
    monkeypatch.setattr(embeddings, "_local_model", lambda: calls.append("model"))
    monkeypatch.setattr(main, "_health_cached", lambda: calls.append("health"))
    main._warm_embedder()
    assert calls == ["health"]                                # no ONNX load on es/omniroute
    monkeypatch.setattr(embeddings, "backend", lambda: "local")
    monkeypatch.setattr(embeddings, "_local_model", lambda: (_ for _ in ()).throw(RuntimeError("no model")))
    main._warm_embedder()                                     # must not raise


def test_startup_hook_schedules_the_warmup_after_ensure_indices_and_survives_thread_failure(monkeypatch):
    import threading
    order = []
    monkeypatch.setattr(main, "ensure_indices", lambda: order.append("indices"))
    monkeypatch.setattr(main.facts, "ensure_queue", lambda: order.append("queue"))
    monkeypatch.setattr(main.pages_mod, "ensure_pages_index", lambda: order.append("pages"))
    monkeypatch.setattr(main.mm_mod, "ensure_models_index", lambda: order.append("models"))
    monkeypatch.setattr(main, "_warm_embedder", lambda: order.append("warm"))
    main._startup()
    for _ in range(50):
        if "warm" in order:
            break
        import time
        time.sleep(0.05)
    assert order[:4] == ["indices", "queue", "pages", "models"] and order[-1] == "warm"

    def no_start(self):
        raise RuntimeError("can't start new thread")
    monkeypatch.setattr(threading.Thread, "start", no_start)
    main._startup()                                           # a failed warm-up thread must not fail startup


def _hold_local_load(monkeypatch):
    """Start a real _local_model() load that parks inside load_local, holding _local_lock."""
    import threading
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(embeddings, "backend", lambda: "local")
    monkeypatch.setattr(embeddings, "_local", None)

    def slow_load(*a, **kw):
        entered.set()
        release.wait(10)
        return ("session", "tokenizer")
    monkeypatch.setattr(embeddings, "load_local", slow_load)
    loader = threading.Thread(target=embeddings._local_model, daemon=True)
    loader.start()
    assert entered.wait(5), "load never started"
    return release, loader


def test_health_answers_loading_at_once_while_the_local_model_loads(monkeypatch):
    """/health must not queue on _local_lock for the whole load: it answers 200 + degraded with
    embed_error 'loading', /health/ready is 503, and the transient body is not cached, so both
    recover as soon as the load finishes."""
    import time
    release, loader = _hold_local_load(monkeypatch)
    try:
        t = time.monotonic()
        r = client.get("/health")
        assert time.monotonic() - t < 3, "health waited for the model load"
        assert r.status_code == 200
        body = r.json()
        assert body["degraded"] is True and body["embed_ok"] is False and body["embed_error"] == "loading"
        assert client.get("/health/ready").status_code == 503
    finally:
        release.set()
        loader.join(5)
    monkeypatch.setattr(embeddings, "embed", lambda texts: [[0.0]])
    assert embeddings._local is not None
    assert client.get("/health/ready").status_code == 200       # 'loading' was not cached
    assert client.get("/health").json()["degraded"] is False


def test_probe_with_the_model_loaded_does_not_report_loading(monkeypatch):
    """Negative control: a held lock is irrelevant once the model is in memory."""
    monkeypatch.setattr(embeddings, "backend", lambda: "local")
    monkeypatch.setattr(embeddings, "_local", ("session", "tokenizer"))
    monkeypatch.setattr(embeddings, "embed", lambda texts: [[0.0]])
    with embeddings._local_lock:
        embeddings.probe()


def _fake_local(monkeypatch, gate, loaded_in):
    import threading
    monkeypatch.setattr(embeddings, "backend", lambda: "local")
    monkeypatch.setattr(embeddings, "_local", None)
    monkeypatch.setattr(embeddings, "_loader", None)
    monkeypatch.setattr(embeddings, "_load_failure", None)

    def load(*a, **kw):
        loaded_in.append(threading.current_thread())
        gate.wait(10)
        return ("session", "tokenizer")
    monkeypatch.setattr(embeddings, "load_local", load)
    monkeypatch.setattr(embeddings, "local_embed_with", lambda s, t, texts: [[0.0] for _ in texts])


def test_probe_with_no_warmup_thread_never_loads_inline_and_recovers(monkeypatch):
    """The startup warm-up thread may never have started, or a load may begin between the probe's
    lock check and its embed. The probe must report 'loading' at once, load the model on a
    background thread of its own, and report healthy once that load finishes."""
    import threading
    import time
    gate, loaded_in = threading.Event(), []
    _fake_local(monkeypatch, gate, loaded_in)
    try:
        t = time.monotonic()
        body = client.get("/health").json()
        assert time.monotonic() - t < 3, "health waited for (or ran) the model load"
        assert body["embed_error"] == "loading" and body["degraded"] is True
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health").json()["embed_error"] == "loading"      # still loading
        loaders = [th for th in threading.enumerate() if th.name == "embed-probe-load"]
        assert len(loaders) == 1, f"{len(loaders)} loader threads queued behind the load"
    finally:
        gate.set()
    for _ in range(100):
        if embeddings._local is not None:
            break
        time.sleep(0.05)
    assert len(loaded_in) == 1, "one background load, not one per probe"
    assert loaded_in[0] is not threading.current_thread() and loaded_in[0].name == "embed-probe-load"
    again = client.get("/health").json()
    assert again["embed_ok"] is True and "embed_error" not in again
    assert client.get("/health/ready").status_code == 200


def test_probe_reports_a_failed_background_load_instead_of_loading_forever(monkeypatch):
    import threading
    import time
    _fake_local(monkeypatch, threading.Event(), [])

    def broken(*a, **kw):
        raise embeddings.ModelUnavailable("model files missing at /nowhere")
    monkeypatch.setattr(embeddings, "load_local", broken)
    assert client.get("/health").json()["embed_error"] == "loading"           # the load starts
    for _ in range(100):
        if embeddings._load_failure is not None:
            break
        time.sleep(0.05)
    body = client.get("/health").json()
    assert "ModelUnavailable" in body["embed_error"] and "/nowhere" in body["embed_error"]
