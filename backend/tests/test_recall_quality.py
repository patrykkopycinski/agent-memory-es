"""Recall-quality + dedup-on-write tests for the quality_decisions changes.

Covers: flat fused `results` shape, per-kind/per-doc result budgets, occurred_at
recency arm, min_score abstention, dedup-on-write. Integration tests run against
live ES (AMES_ES_URL). Run in the backend image:

  docker run --rm --network host -v <repo>:/srv -w /srv/backend \
    -e AMES_ES_URL=http://127.0.0.1:19200 \
    -e AMES_INDEX_PREFIX=amtest_rq_ \
    backend-ames-backend python -m pytest tests/test_recall_quality.py -v
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import memory  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qtest"


def _wipe(owner=OWNER):
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": owner}}})


def _count(owner, kind="semantic"):
    r = es("POST", f"/{idx(kind)}/_count",
           {"query": {"term": {"owner_id": owner}}})
    return r["count"]


# ---------------------------------------------------------------- flat shape

def test_flat_results_shape():
    _wipe()
    memory.retain(OWNER, "semantic", "OmniRoute gateway heap limit is 12288MB at startup")
    memory.retain(OWNER, "episodic", "At the standup we triaged the gateway heap threshold")
    out = memory.recall(OWNER, "gateway heap limit", size=8)
    assert out.get("results"), out
    for item in out["results"]:
        assert {"kind", "score", "visibility"}.issubset(item), item
        assert item["kind"] in ("semantic", "episodic", "procedural"), item
        assert isinstance(item["score"], (int, float))
    # by_kind / fused are deprecated additive fields kept for compat.
    assert "by_kind" in out and out["by_kind"]["semantic"], out["by_kind"]
    assert out["fused"] == out["results"]
    assert out["abstained"] is False


# ------------------------------------------------------------------- budgets

def test_apply_budgets_per_kind_and_per_doc():
    ordered = ([{"id": f"s{i}", "kind": "semantic", "score": 0.5 - i * 0.01} for i in range(6)]
               + [{"id": f"e{i}", "kind": "episodic", "score": 0.4 - i * 0.01} for i in range(6)])
    window = memory._apply_budgets(ordered, size=8, per_kind_budget=4, per_doc_budget=3)
    assert len(window) == 8
    kinds = [h["kind"] for h in window]
    assert kinds.count("semantic") <= 4 and kinds.count("episodic") <= 4, kinds
    assert kinds.count("episodic") >= 1 and kinds.count("semantic") >= 1, kinds


def test_apply_budgets_caps_one_doc_group():
    # one huge (chunked) retain must not eat the window
    big = [{"id": f"big{i}", "kind": "semantic", "doc_group": "bigdoc",
            "score": 0.9 - i * 0.001} for i in range(6)]
    other = [{"id": f"o{i}", "kind": "semantic", "score": 0.1 - i * 0.001} for i in range(3)]
    window = memory._apply_budgets(big + other, size=8, per_kind_budget=8, per_doc_budget=3)
    from_big = [h for h in window if h.get("doc_group") == "bigdoc"]
    assert len(from_big) == 3, from_big
    assert len(window) == 6


def test_per_kind_budget_fairness_live():
    _wipe()
    memory.retain(OWNER, "semantic", "Alpha gateway retry policy is exponential backoff")
    memory.retain(OWNER, "episodic", "Alpha gateway incident was a backoff retry storm")
    out = memory.recall(OWNER, "alpha gateway backoff", size=8)
    kinds = {h["kind"] for h in out["results"]}
    assert kinds == {"semantic", "episodic"}, out["results"]


# ------------------------------------------------------------------- recency

def test_recency_decay_math():
    now = dt.datetime(2026, 10, 4, tzinfo=dt.timezone.utc)
    fresh = memory.recency_decay("2026-10-04T00:00:00Z", now)
    half = memory.recency_decay("2026-09-04T00:00:00Z", now)     # 30 days
    cold = memory.recency_decay("2024-10-04T00:00:00Z", now)     # 2 years
    assert 0.99 <= fresh <= 1.0, fresh
    assert abs(half - 0.5) < 0.03, half
    assert cold < 0.01, cold
    assert memory.recency_decay(None, now) == 0.0


def test_recency_arm_is_wired_into_score():
    _wipe()
    memory.retain(OWNER, "semantic", "Quarterly capacity review covers shard rebalance")
    out = memory.recall(OWNER, "shard rebalance capacity", size=8)
    assert out["results"]
    for h in out["results"]:
        assert "recency" in h and "rrf" in h
        assert abs(h["score"] - (h["rrf"] + h["recency"])) < 1e-9, h


def test_recency_boost_orders_recent_first(monkeypatch):
    _wipe()
    old = "The nightly capacity report lists shard rebalance targets for the cluster"
    new = "Capacity planning notes: shard rebalance target moved after the incident review"
    memory.retain(OWNER, "semantic", old, occurred_at="2024-01-01T00:00:00Z")
    memory.retain(OWNER, "semantic", new, occurred_at="2026-10-03T00:00:00Z")

    monkeypatch.setattr(memory, "RECENCY_WEIGHT", 5.0)
    # the recency arm only orders the FUSED list; the (default-on) cross-encoder rerank replaces
    # that order by relevance, so this contract is tested on the fused path (rerank=False).
    boosted = memory.recall(OWNER, "shard rebalance targets", size=8, rerank=False)["results"]
    recent_first = next(h for h in boosted if h["text"] == new)
    old_second = next(h for h in boosted if h["text"] == old)
    assert boosted.index(recent_first) < boosted.index(old_second), [h["text"] for h in boosted]

    monkeypatch.setattr(memory, "RECENCY_WEIGHT", 0.0)
    flat = memory.recall(OWNER, "shard rebalance targets", size=8)["results"]
    assert flat.index(next(h for h in flat if h["text"] == old)) < \
        flat.index(next(h for h in flat if h["text"] == new)), [h["text"] for h in flat]


# ----------------------------------------------------------------- min_score

def test_min_score_abstains_when_nothing_clears_floor():
    _wipe()
    memory.retain(OWNER, "semantic", "The blue deploy pipeline runs on Thursday")
    out = memory.recall(OWNER, "blue deploy pipeline", size=8, min_score=1e9)
    assert out["results"] == []
    assert out["abstained"] is True


def test_min_score_post_filters_and_keeps_best():
    _wipe()
    memory.retain(OWNER, "semantic", "The blue deploy pipeline runs on Thursday")
    memory.retain(OWNER, "semantic", "Falcon tracing exporter samples one request in ten")
    full = memory.recall(OWNER, "blue deploy pipeline thursday", size=8)
    assert len(full["results"]) >= 1
    top = max(h["score"] for h in full["results"])
    floor = top - 1e-6   # scores are recomputed per call (live recency), so allow epsilon
    floored = memory.recall(OWNER, "blue deploy pipeline thursday", size=8, min_score=floor)
    assert floored["results"], floored
    assert all(h["score"] >= floor for h in floored["results"])
    assert floored["abstained"] is False


# ------------------------------------------------------------ dedup on write

def test_dedup_on_write_exact_hash_only_near_dupes_are_linked():
    """Cause 1: only an EXACT normalized-text match dedupes. A near-duplicate is
    WRITTEN and linked (supersedes/superseded_by), because e5-small sees only the
    first ~512 tokens and same-topic sessions score 0.93-0.97 — dropping those
    silently lost the NEWER session and 38 gold sessions across the 100 questions."""
    _wipe("qdedup")
    first = memory.retain("qdedup", "semantic", "OmniRoute heap watchdog warns at 9450MB")
    assert first["deduped"] is False and first["dedup_reason"] == "distinct"
    # exact text (whitespace/case-normalized) still collapses onto the same doc
    again = memory.retain("qdedup", "semantic", "OmniRoute  heap watchdog warns at 9450MB")
    assert again["deduped"] is True and again["_id"] == first["_id"], again
    assert again["dedup_reason"] == "exact_text_hash", again
    assert again["deduped_against_owner"] == "qdedup", again
    assert _count("qdedup") == 1
    # near-duplicate: WRITTEN, not dropped, and linked to the earlier doc
    near = memory.retain("qdedup", "semantic",
                         "The OmniRoute heap watchdog warns at 9450MB before restart")
    assert near["deduped"] is False and near["_id"] != first["_id"], near
    assert near["dedup_reason"] == "near_duplicate_linked", near
    assert near["supersedes"] == [first["ids"][0]], near
    assert _count("qdedup") == 2
    # the OLD doc is still active (coexistence link, not a merge): both retrievable
    old = es("GET", f"/{idx('semantic')}/_doc/{first['ids'][0]}")["_source"]
    assert old["active"] is True and old["superseded_by"] == near["ids"][0], old
    # an unrelated fact is neither deduped nor linked
    distinct = memory.retain("qdedup", "semantic",
                             "Vue is the frontend framework used for the dashboard")
    assert distinct["deduped"] is False and distinct["_id"] not in (first["_id"], near["_id"])
    assert distinct["dedup_reason"] == "distinct"
    assert _count("qdedup") == 3


def test_dedup_respects_visibility_scope():
    _wipe("qscope1")
    _wipe("qscope2")
    text = "Shared retry budget for the platform gateway is twelve attempts"
    a = memory.retain("qscope1", "semantic", text, visibility="private")
    assert a["deduped"] is False
    # another owner's private doc is a different scope: never dedup across it
    b = memory.retain("qscope2", "semantic", text, visibility="private")
    assert b["deduped"] is False and b["_id"] != a["_id"]
    assert _count("qscope2") == 1
    # same text in the shared scope collapses onto the existing common doc
    c = memory.retain("qscope2", "semantic", text, visibility="common")
    assert c["deduped"] is False
    d = memory.retain("qscope1", "semantic", text, visibility="common")
    assert d["deduped"] is True and d["_id"] == c["_id"], d
    assert d["deduped_against_owner"] == "qscope2", d  # cross-owner collapse reports doc owner
