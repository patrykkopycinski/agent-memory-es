"""Recall returns hits best-confidence first.

Clients take the first hit as the answer, so the response order must already be the best one:
`results` is confidence-descending, stable on ties (fused order kept), hits without a confidence
last. The sort runs BEFORE the per-doc budgets and the size cut, so a confident hit that sat
deep in the fused order is not dropped by `size`.

The recall tests run the real search path against the throwaway test ES; the confidence signal is
replaced by one that inverts the lexical score, which is what makes the fused order and the
confidence order disagree deterministically.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

import pytest  # noqa: E402

from app import memory  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qconforder"
QUERY = "quarterly warehouse relocation budget"
# each doc matches fewer query terms than the one before: fused order is d0 > d1 > d2 > d3 > d4
DOCS = [
    "quarterly warehouse relocation budget",
    "quarterly warehouse relocation notes",
    "quarterly warehouse inventory notes",
    "quarterly staffing notes",
    "unrelated quarterly",
]


def _wipe():
    es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true",
       {"query": {"term": {"owner_id": OWNER}}})


@pytest.fixture
def seeded():
    _wipe()
    ids = [memory.retain(OWNER, "semantic", t, extract_facts=False)["_id"] for t in DOCS]
    yield ids
    _wipe()


def _inverted(monkeypatch):
    """Confidence that rises as the lexical score falls: opposes the fused order."""
    monkeypatch.setattr(memory, "_confidence",
                        lambda bm25, cosine, have_vec: round(1.0 / (1.0 + (bm25 or 0.0)), 6))


def _recall(**kw):
    return memory.recall(OWNER, QUERY, kinds=["semantic"], rerank=False, **kw)


def _confs(out):
    return [h["confidence"] for h in out["results"]]


def test_results_are_confidence_descending_when_fused_order_disagrees(seeded, monkeypatch):
    _inverted(monkeypatch)
    out = _recall(size=5)
    assert len(out["results"]) == 5
    # every seeded doc is recallable (subset, wide window: at k=5 shared-visibility docs
    # from other owners can legitimately outrank the lowest-confidence seeded docs)
    wide = {h["id"] for h in _recall(size=50)["results"]}
    assert set(seeded) <= wide
    assert _confs(out) == sorted(_confs(out), reverse=True)
    # the fixture really disagrees: the fused (score) order is not the order returned
    assert [h["id"] for h in sorted(out["results"], key=lambda h: -h["score"])] != [h["id"] for h in out["results"]]
    assert out["fused"] == out["results"]                  # deprecated alias carries the same order


def test_http_recall_returns_confidence_descending(seeded, monkeypatch):
    from fastapi.testclient import TestClient
    from app import auth
    from app.main import app
    _inverted(monkeypatch)
    r = TestClient(app).post("/memory/recall", headers={"X-API-Key": auth.create_key(OWNER)},
                             json={"query": QUERY, "kinds": ["semantic"], "size": 5, "rerank": False})
    assert r.status_code == 200, r.text
    confs = [h["confidence"] for h in r.json()["results"]]
    assert len(confs) == 5 and confs == sorted(confs, reverse=True)


def test_sort_happens_before_the_size_cut(seeded, monkeypatch):
    _inverted(monkeypatch)
    full = _recall(size=5)["results"]
    best_two = [h["id"] for h in full[:2]]                                   # already confidence-sorted
    fused_top_two = [h["id"] for h in sorted(full, key=lambda h: -h["score"])[:2]]
    assert best_two != fused_top_two, "fixture must put the most confident hits outside the fused top 2"
    out = _recall(size=2)
    # cutting to 2 BEFORE sorting would keep the fused top 2 and drop the most confident hits
    assert [h["id"] for h in out["results"]] == best_two
    assert _confs(out) == sorted(_confs(out), reverse=True)


def test_equal_confidence_keeps_fused_order(seeded, monkeypatch):
    monkeypatch.setattr(memory, "_confidence", lambda *a, **k: 0.5)
    out = _recall(size=5)
    scores = [h["score"] for h in out["results"]]
    assert scores == sorted(scores, reverse=True)
    assert [h["id"] for h in out["results"]][:2] == seeded[:2]


def test_missing_confidence_does_not_crash_and_keeps_fused_order(seeded, monkeypatch):
    monkeypatch.setattr(memory, "_confidence", lambda *a, **k: None)
    out = _recall(size=5)
    scores = [h["score"] for h in out["results"]]
    assert scores == sorted(scores, reverse=True) and len(scores) == 5
    assert out["abstain_hint"] is True


def test_sort_by_confidence_ties_and_none_units():
    hits = [{"id": "a", "confidence": None}, {"id": "b", "confidence": 0.4}, {"id": "c", "confidence": 0.9},
            {"id": "d"}, {"id": "e", "confidence": 0.4}, {"id": "f", "confidence": 0.0}]
    # 0.9 first; the 0.4 tie keeps input order (b before e); 0.0 is a real value and beats None/missing,
    # which go last in input order (a before d)
    assert [h["id"] for h in memory._sort_by_confidence(hits)] == ["c", "b", "e", "f", "a", "d"]
    assert [h["id"] for h in hits] == ["a", "b", "c", "d", "e", "f"]    # input is not mutated
    assert memory._sort_by_confidence([]) == []


def test_rerank_order_is_not_overridden(monkeypatch):
    """A cross-encoder order the caller asked for (and that ran) outranks the confidence sort."""
    ordered = [{"id": "x", "kind": "semantic", "score": 0.1, "confidence": 0.1, "rerank_rank": 0},
               {"id": "y", "kind": "semantic", "score": 0.9, "confidence": 0.9, "rerank_rank": 1}]
    assert [h["id"] for h in memory._apply_budgets(ordered, 2)] == ["x", "y"]


def test_apply_budgets_by_confidence_orders_the_window_and_breaks_ties_by_score():
    items = [{"id": "a", "kind": "semantic", "score": 0.9, "confidence": 0.2},
             {"id": "b", "kind": "semantic", "score": 0.8, "confidence": 0.7},
             {"id": "c", "kind": "semantic", "score": 0.7, "confidence": 0.7},
             {"id": "d", "kind": "semantic", "score": 0.6}]
    assert [h["id"] for h in memory._apply_budgets(items, 4, by_confidence=True)] == ["b", "c", "a", "d"]
    assert [h["id"] for h in memory._apply_budgets(items, 4)] == ["a", "b", "c", "d"]   # default unchanged
