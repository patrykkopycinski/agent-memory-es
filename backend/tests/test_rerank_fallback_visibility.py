"""A rerank that was requested but not applied must say so (rerank.fallback_reason), and the
rerank budget/depth must be bounded. Timeout tests use a real stalled socket, not a stub.
"""
import http.server
import os
import sys
import threading
import time
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, reranker  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qrerankfb"
Q = "quarterly warehouse relocation budget plan"
HITS = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}, {"id": "c", "text": "z"}]


class _Stall(http.server.BaseHTTPRequestHandler):
    def do_POST(self):          # accepts the connection, never answers within the budget
        time.sleep(3)

    def log_message(self, *a):
        pass


@pytest.fixture
def stalled_es(monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stall)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # The reranker resolves the ES URL lazily via its own imported _es_url binding;
    # patch only that binding so seeding/recall still reach the real cluster.
    monkeypatch.setattr(reranker, "_es_url", lambda: "http://127.0.0.1:%d" % srv.server_port)
    monkeypatch.setattr(reranker, "RERANK_TIMEOUT", 0.3)
    yield
    srv.shutdown()


def test_default_budget_is_interactive():
    assert 0 < reranker.RERANK_TIMEOUT <= 5


def test_real_socket_timeout_is_reported_as_timeout(stalled_es):
    t0 = time.monotonic()
    out = reranker.rerank(Q, HITS, top_n=3)
    assert time.monotonic() - t0 < 2.5            # bounded by the budget, not the 3 s stall
    assert out["reranked"] is False and out["fallback_reason"] == "timeout"
    assert [h["id"] for h in out["hits"]] == ["a", "b", "c"]


@pytest.mark.parametrize("exc,reason", [
    (urllib.error.URLError("connection refused"), "unavailable"),
    (ConnectionResetError("reset"), "unavailable"),
    (urllib.error.HTTPError("u", 404, "nf", {}, None), "unavailable"),
    (urllib.error.HTTPError("u", 403, "license", {}, None), "unavailable"),
    (urllib.error.HTTPError("u", 500, "boom", {}, None), "error"),
    (ValueError("bad json"), "error"),
])
def test_failure_reasons(monkeypatch, exc, reason):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(reranker.urllib.request, "urlopen", boom)
    out = reranker.rerank(Q, HITS, top_n=3)
    assert out["reranked"] is False and out["fallback_reason"] == reason


def test_unusable_body_is_error(monkeypatch):
    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return b'{"no_rerank_key": []}'
    monkeypatch.setattr(reranker.urllib.request, "urlopen", lambda *a, **k: R())
    out = reranker.rerank(Q, HITS, top_n=3)
    assert out["fallback_reason"] == "error"


def _items(n):
    return [{"id": "p%d" % i, "doc_group": "g%d" % (i % 4), "kind": "episodic",
             "score": 1.0 - i * 0.01, "text": "t%d" % i} for i in range(n)]


def test_fused_meta_reports_timeout_and_keeps_order(monkeypatch):
    monkeypatch.setattr(reranker, "rerank", lambda q, h, top_n=5, model=None, timeout=None:
                        {"hits": h[:top_n], "reranked": False, "fallback_reason": "timeout", "ms": 2001})
    items = _items(12)
    new, info = memory._rerank_fused_meta(Q, items)
    assert [h["id"] for h in new] == [h["id"] for h in items]
    assert info == {"requested": True, "applied": False, "fallback_reason": "timeout",
                    "depth": 12, "ms": 2001}


def test_fused_meta_success_has_no_fallback_reason(monkeypatch):
    def ok(q, h, top_n=5, model=None, timeout=None):
        return {"hits": [dict(x, rerank_score=1.0) for x in reversed(h)], "reranked": True, "ms": 40}
    monkeypatch.setattr(reranker, "rerank", ok)
    _, info = memory._rerank_fused_meta(Q, _items(6))
    assert info == {"requested": True, "applied": True, "fallback_reason": None, "depth": 6, "ms": 40}


def test_too_few_candidates_is_explicit(monkeypatch):
    monkeypatch.setattr(reranker, "rerank", lambda *a, **k: pytest.fail("must not call"))
    _, info = memory._rerank_fused_meta(Q, _items(1))
    assert info["applied"] is False and info["fallback_reason"] == "too_few_candidates"


@pytest.mark.parametrize("asked,sent", [(10, 10), (30, 30), (5000, 50), (0, 2), (-4, 2), ("zz", 50)])
def test_depth_is_clamped(monkeypatch, asked, sent):
    seen = {}

    def spy(q, h, top_n=5, model=None, timeout=None):
        seen["n"] = len(h)
        return {"hits": h, "reranked": False, "fallback_reason": "error", "ms": 1}
    monkeypatch.setattr(reranker, "rerank", spy)
    _, info = memory._rerank_fused_meta(Q, _items(80), depth=asked)
    assert seen["n"] == sent and info["depth"] == sent


# ------------------------------------------------------------------ through recall() + HTTP, real ES
def _seed():
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", "/%s/_delete_by_query?refresh=true" % idx(kind), {"query": {"term": {"owner_id": OWNER}}})
    for g in range(4):
        memory.retain(OWNER, "episodic",
                      "\n".join("user: s%d quarterly warehouse relocation budget plan item %03d %s" % (g, i, "q" * 60)
                                for i in range(30)),
                      doc_group="sess-%d" % g, occurred_at="2023-05-10T09:00:00")


def test_recall_surfaces_timeout_via_a_real_stalled_endpoint(stalled_es):
    _seed()
    off = memory.recall(OWNER, Q, kinds=["episodic"], size=3, as_of="2023-05-30", rerank=False)
    on = memory.recall(OWNER, Q, kinds=["episodic"], size=3, as_of="2023-05-30", rerank=True)
    assert "rerank" not in off                        # additive: absent unless requested
    assert on["reranked"] is False
    assert on["rerank"]["requested"] is True and on["rerank"]["applied"] is False
    assert on["rerank"]["fallback_reason"] == "timeout"
    assert on["rerank"]["depth"] >= 2 and on["rerank"]["ms"] >= 250
    assert [h["id"] for h in on["results"]] == [h["id"] for h in off["results"]]


def test_http_request_depth_reaches_the_reranker_and_response_has_rerank_block(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app, RecallIn
    from app import auth
    assert RecallIn(query="x").rerank_depth is None
    _seed()
    seen = {}

    def spy(q, h, top_n=5, model=None, timeout=None):
        seen["n"] = len(h)
        return {"hits": h[:top_n], "reranked": False, "fallback_reason": "timeout", "ms": 7}
    monkeypatch.setattr(reranker, "rerank", spy)
    c = TestClient(app); key = auth.create_key(OWNER)
    r = c.post("/memory/recall", headers={"X-API-Key": key},
               json={"query": Q, "kinds": ["episodic"], "size": 3, "as_of": "2023-05-30",
                     "rerank": True, "rerank_depth": 3})
    assert r.status_code == 200
    body = r.json()
    assert seen["n"] == 3
    assert body["rerank"] == {"requested": True, "applied": False, "fallback_reason": "timeout",
                              "depth": 3, "ms": 7}
