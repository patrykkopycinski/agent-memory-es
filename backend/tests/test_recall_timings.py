"""Recall latency + context-cost instrumentation: additive `timings_ms`, `ames.recall` log line,
and the conftest guard that keeps the suite off the production ES port."""
import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, reranker  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qtimings-owner-7f3a"
QUERY = "quarterly harbour crane maintenance schedule"
TEXTS = [
    "The harbour crane maintenance schedule moves to the first week of each quarter.",
    "Crane operators rotate every four hours during quarterly maintenance windows.",
    "Unrelated note about a bakery opening on the high street.",
    "Warehouse relocation budget was approved for the second quarter.",
]
BASE_KEYS = {"total", "embed", "search", "by_kind", "es_took", "fuse"}


@pytest.fixture(scope="module", autouse=True)
def seeded():
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", "/%s/_delete_by_query?refresh=true" % idx(kind),
           {"query": {"term": {"owner_id": OWNER}}})
    for i, t in enumerate(TEXTS):
        memory.retain(OWNER, "episodic", t, "private", doc_group="tm-%d" % i)
    es("POST", "/%s/_refresh" % idx("episodic"))
    yield


def _numbers(t):
    for k, v in t.items():
        if k == "by_kind":
            assert isinstance(v, dict) and v
            assert all(isinstance(x, (int, float)) and x >= 0 for x in v.values())
        else:
            assert isinstance(v, (int, float)) and v >= 0, (k, v)


def test_timings_present_and_non_negative_without_rerank():
    out = memory.recall(OWNER, QUERY, rerank=False)
    t = out["timings_ms"]
    assert BASE_KEYS <= set(t)
    _numbers(t)
    assert "rerank" not in t           # rerank key only when rerank ran
    assert "filter_resolve" not in t   # only when a filter is given
    assert t["total"] >= t["search"]
    assert set(t["by_kind"]) == set(memory.KINDS)
    assert t["search"] == pytest.approx(sum(t["by_kind"].values()), abs=0.01)
    assert out["context_chars"] == sum(len(h.get("text", "")) for h in out["evidence"])
    assert out["context_chars"] > 0


def test_rerank_timing_present_only_when_rerank_ran(monkeypatch):
    def fake(query, hits, top_n=5, model=None):
        return {"hits": list(reversed(hits[:top_n])), "reranked": True}
    monkeypatch.setattr(reranker, "rerank", fake)
    out = memory.recall(OWNER, QUERY, rerank=True)
    assert out["reranked"] is True
    assert "rerank" in out["timings_ms"]
    _numbers(out["timings_ms"])
    assert out["timings_ms"]["total"] >= out["timings_ms"]["rerank"]


def test_rerank_timing_recorded_when_endpoint_falls_back(monkeypatch):
    monkeypatch.setattr(reranker, "rerank",
                        lambda q, hits, top_n=5, model=None: {"hits": hits[:top_n], "reranked": False})
    out = memory.recall(OWNER, QUERY, rerank=True)
    assert out["reranked"] is False
    assert "rerank" in out["timings_ms"]    # rerank ran (and fell back): the cost was paid


def test_filter_resolve_timing_when_filter_given():
    out = memory.recall(OWNER, QUERY, rerank=False, filter={"none": ["zz:never"]})
    t = out["timings_ms"]
    assert "filter_resolve" in t and t["filter_resolve"] >= 0
    _numbers(t)


def test_abstained_filter_still_reports_timings():
    out = memory.recall(OWNER, QUERY, filter={"narrow_any": [{"tags": ["name:qqqq"], "resolve": "fuzzy"}]})
    assert out["abstained"] is True and out["results"] == []
    t = out["timings_ms"]
    assert "filter_resolve" in t and "total" in t
    assert out["context_chars"] == 0
    assert "search" not in t


def test_recall_log_line_has_no_query_or_owner(caplog):
    with caplog.at_level(logging.INFO, logger="ames.recall"):
        memory.recall(OWNER, QUERY, size=5, rerank=False)
    lines = [r.getMessage() for r in caplog.records if r.name == "ames.recall"]
    assert len(lines) == 1
    assert lines[0].startswith("ames.recall ")
    payload = json.loads(lines[0][len("ames.recall "):])
    assert payload["k"] == 5 and payload["rerank"] is False
    assert payload["filtered"] is False and payload["abstained"] is False
    assert BASE_KEYS <= set(payload["timings_ms"])
    assert payload["owner"] == memory._owner_hash(OWNER) and payload["owner"] != OWNER
    assert OWNER not in lines[0]
    assert "harbour" not in lines[0] and "crane" not in lines[0]


def test_existing_fields_unchanged():
    out = memory.recall(OWNER, QUERY, rerank=False)
    for k in ("query", "results", "evidence", "fact_count", "chunk_count", "by_kind", "fused",
              "abstained", "doc_groups", "reranked"):
        assert k in out


# ---------------------------------------------------------------- prod-ES guard (conftest)
def test_guard_refuses_prod_port_with_empty_prefix():
    from conftest import refuse_prod_es
    for url in ("http://localhost:9268", "http://127.0.0.1:9268/", "http://host.docker.internal:9268"):
        with pytest.raises(RuntimeError):
            refuse_prod_es(url, "")
        with pytest.raises(RuntimeError):
            refuse_prod_es(url, "   ")


def test_guard_allows_prod_port_with_prefix_and_other_ports():
    from conftest import refuse_prod_es
    refuse_prod_es("http://localhost:9268", "amtest_")
    refuse_prod_es("http://127.0.0.1:19281", "")
    refuse_prod_es("http://localhost:92680", "")


# ---------------------------------------------------------------- HTTP surface
def test_http_recall_returns_timings_and_emits_log_line(caplog):
    from fastapi.testclient import TestClient
    from app import auth
    from app.main import app
    key = auth.create_key(OWNER)
    with TestClient(app) as c, caplog.at_level(logging.INFO, logger="ames.recall"):
        r = c.post("/memory/recall", json={"query": QUERY, "rerank": False},
                   headers={"X-API-Key": key})
    assert r.status_code == 200
    body = r.json()
    assert BASE_KEYS <= set(body["timings_ms"]) and body["context_chars"] > 0
    assert any(rec.name == "ames.recall" for rec in caplog.records)
    # the app configures a real handler so the line is not lost under uvicorn's logging setup
    assert logging.getLogger("ames.recall").handlers
    assert logging.getLogger("ames.recall").level == logging.INFO
