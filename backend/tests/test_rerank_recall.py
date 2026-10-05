"""Round 5: ES-native rerank of the fused top-50, BEFORE the per-doc budget + size cut.
The output budget is unchanged (size passages). Falls back honestly when the endpoint is down.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, reranker  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qrerank"
Q = "quarterly warehouse relocation budget plan"


def _items(n=12, groups=4):
    return sorted([{"id": "p%d" % i, "doc_group": "g%d" % (i % groups), "kind": "episodic",
                    "score": 1.0 - i * 0.01, "text": "t%d" % i} for i in range(n)],
                  key=lambda x: -x["score"])


def _fake(order):
    """reranker.rerank stand-in: returns hits in `order` (ids) with descending scores."""
    def f(query, hits, top_n=5, model=None):
        by = {h["id"]: h for h in hits}
        out = []
        for r, i in enumerate(order):
            if i in by:
                h = dict(by[i]); h["rerank_score"] = 10.0 - r; out.append(h)
        return {"hits": out[:top_n], "reranked": True}
    return f


# ------------------------------------------------------------------ pure units
def test_rerank_reorders_by_the_cross_encoder_not_the_fused_score(monkeypatch):
    monkeypatch.setattr(reranker, "rerank", _fake(["p9", "p3", "p0"]))
    new, ok = memory._rerank_fused(Q, _items())
    assert ok and [h["id"] for h in new[:3]] == ["p9", "p3", "p0"]


def test_rerank_rank_is_a_dense_0_based_order_matching_the_list(monkeypatch):
    monkeypatch.setattr(reranker, "rerank", _fake(["p9", "p3", "p0"]))
    new, _ = memory._rerank_fused(Q, _items())
    assert [h["rerank_rank"] for h in new] == list(range(len(new)))


def test_score_stays_the_fused_score_and_rerank_score_is_separate(monkeypatch):
    monkeypatch.setattr(reranker, "rerank", _fake(["p9"]))
    new, _ = memory._rerank_fused(Q, _items())
    top = new[0]
    assert top["id"] == "p9" and abs(top["score"] - (1.0 - 9 * 0.01)) < 1e-9     # untouched
    assert top["rerank_score"] == 10.0
    assert all("rerank_score" in h for h in new[:1])


def test_no_candidate_is_lost_when_the_endpoint_returns_a_subset(monkeypatch):
    monkeypatch.setattr(reranker, "rerank", _fake(["p5"]))
    items = _items()
    new, ok = memory._rerank_fused(Q, items)
    assert ok and sorted(h["id"] for h in new) == sorted(h["id"] for h in items)
    assert new[0]["id"] == "p5"


def test_candidates_beyond_depth_stay_below_every_reranked_one(monkeypatch):
    monkeypatch.setattr(reranker, "rerank", _fake(["p2", "p1", "p0"]))
    new, _ = memory._rerank_fused(Q, _items(12), depth=3)
    assert [h["id"] for h in new[:3]] == ["p2", "p1", "p0"]
    assert [h["id"] for h in new[3:]] == ["p%d" % i for i in range(3, 12)]
    assert all(new[2]["rerank_rank"] < h["rerank_rank"] for h in new[3:])
    assert all("rerank_score" not in h for h in new[3:])


def test_only_the_top_depth_is_sent_to_the_endpoint(monkeypatch):
    seen = {}

    def spy(query, hits, top_n=5, model=None):
        seen["n"] = len(hits)
        return {"hits": hits, "reranked": False}
    monkeypatch.setattr(reranker, "rerank", spy)
    memory._rerank_fused(Q, _items(12), depth=5)
    assert seen["n"] == 5


def test_endpoint_unavailable_keeps_the_fused_order_and_says_so(monkeypatch):
    monkeypatch.setattr(reranker, "rerank", lambda q, h, top_n=5, model=None: {"hits": h[:top_n], "reranked": False})
    items = _items()
    new, ok = memory._rerank_fused(Q, items)
    assert ok is False and [h["id"] for h in new] == [h["id"] for h in items]


def test_fewer_than_two_candidates_are_not_sent(monkeypatch):
    called = []
    monkeypatch.setattr(reranker, "rerank", lambda *a, **k: called.append(1) or {"hits": [], "reranked": True})
    new, ok = memory._rerank_fused(Q, _items(1))
    assert ok is False and not called


def test_http_error_in_reranker_degrades_not_raises(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(reranker.urllib.request, "urlopen", boom)
    out = reranker.rerank(Q, [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}], top_n=2)
    assert out["reranked"] is False and len(out["hits"]) == 2


def test_socket_level_failure_in_reranker_also_degrades(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection reset")
    monkeypatch.setattr(reranker.urllib.request, "urlopen", boom)
    out = reranker.rerank(Q, [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}], top_n=2)
    assert out["reranked"] is False


def test_rerank_is_opt_in_by_default():
    # driver decision: merged as OPT-IN; the default must stay off
    assert memory.RERANK_DEFAULT is False
    assert memory.RERANK_DEPTH == 50
    assert reranker.RERANK_TIMEOUT >= 1


def test_rerank_env_flag_and_timeout_are_read_from_config():
    import importlib
    os.environ["AMES_RERANK"] = "1"
    importlib.reload(memory)
    assert memory.RERANK_DEFAULT is True
    os.environ["AMES_RERANK"] = "0"
    importlib.reload(memory)
    assert memory.RERANK_DEFAULT is False
    os.environ["AMES_RERANK_TIMEOUT"] = "7"
    importlib.reload(reranker)
    assert reranker.RERANK_TIMEOUT == 7
    os.environ.pop("AMES_RERANK"); os.environ.pop("AMES_RERANK_TIMEOUT")
    importlib.reload(memory); importlib.reload(reranker)
    assert memory.RERANK_DEFAULT is False and reranker.RERANK_TIMEOUT >= 1


# ------------------------------------------------------------------ integration with recall()
def _wipe():
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", "/%s/_delete_by_query?refresh=true" % idx(kind), {"query": {"term": {"owner_id": OWNER}}})


def _seed():
    _wipe()
    for g in range(4):
        memory.retain(OWNER, "episodic",
                      "\n".join("user: s%d quarterly warehouse relocation budget plan item %03d %s" % (g, i, "q" * 60)
                                for i in range(30)),
                      doc_group="sess-%d" % g, occurred_at="2023-05-10T09:00:00")


def test_recall_applies_rerank_before_the_size_cut(monkeypatch):
    _seed()
    base = memory.recall(OWNER, Q, kinds=["episodic"], size=3, as_of="2023-05-30", rerank=False)
    assert base["reranked"] is False
    worst = None

    def pick_last(query, hits, top_n=5, model=None):
        nonlocal worst
        worst = hits[-1]["id"]
        h = dict(hits[-1]); h["rerank_score"] = 99.0
        return {"hits": [h] + [dict(x, rerank_score=1.0 - i) for i, x in enumerate(hits[:-1])], "reranked": True}
    monkeypatch.setattr(reranker, "rerank", pick_last)
    r = memory.recall(OWNER, Q, kinds=["episodic"], size=3, as_of="2023-05-30", rerank=True)
    assert r["reranked"] is True
    assert r["results"][0]["id"] == worst            # a candidate from OUTSIDE the old top-3 got in
    assert worst not in [x["id"] for x in base["results"]]


def test_output_budget_is_unchanged_by_rerank(monkeypatch):
    _seed()
    r = memory.recall(OWNER, Q, kinds=["episodic"], size=5, as_of="2023-05-30", rerank=True)
    assert len(r["results"]) <= 5
    gs = [x["doc_group"] for x in r["results"]]
    assert max(gs.count(g) for g in set(gs)) <= memory.PER_DOC_BUDGET     # budget still applied AFTER rerank


def test_per_doc_budget_still_binds_after_rerank(monkeypatch):
    _seed()
    r = memory.recall(OWNER, Q, kinds=["episodic"], size=8, as_of="2023-05-30", rerank=True, per_doc=1)
    gs = [x["doc_group"] for x in r["results"]]
    assert gs and max(gs.count(g) for g in set(gs)) == 1


def test_rerank_false_never_calls_the_endpoint(monkeypatch):
    _seed()
    monkeypatch.setattr(reranker, "rerank", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    r = memory.recall(OWNER, Q, kinds=["episodic"], size=3, as_of="2023-05-30", rerank=False)
    assert r["reranked"] is False


def test_rerank_none_follows_the_module_default(monkeypatch):
    _seed()
    monkeypatch.setattr(memory, "RERANK_DEFAULT", False)
    monkeypatch.setattr(reranker, "rerank", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert memory.recall(OWNER, Q, kinds=["episodic"], size=3, as_of="2023-05-30")["reranked"] is False
    monkeypatch.setattr(memory, "RERANK_DEFAULT", True)
    monkeypatch.setattr(reranker, "rerank", _fake([]))
    assert memory.recall(OWNER, Q, kinds=["episodic"], size=3, as_of="2023-05-30")["reranked"] is True


def test_api_model_has_rerank_and_http_reaches_recall(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app, RecallIn
    from app import auth
    assert RecallIn(query="x").rerank is None and RecallIn(query="x", rerank=False).rerank is False
    _seed()
    c = TestClient(app); key = auth.create_key(OWNER)
    r = c.post("/memory/recall", headers={"X-API-Key": key},
               json={"query": Q, "kinds": ["episodic"], "size": 3, "as_of": "2023-05-30", "rerank": False})
    assert r.status_code == 200 and r.json()["reranked"] is False


def test_window_is_ordered_by_rerank_rank_not_by_score(monkeypatch):
    # the cross-encoder puts the LOWEST fused score first; _apply_budgets must respect it
    monkeypatch.setattr(reranker, "rerank", _fake(["p9", "p0", "p1"]))
    new, _ = memory._rerank_fused(Q, _items(12))
    w = memory._apply_budgets(new, 3, per_doc_budget=3)
    assert [h["id"] for h in w] == ["p9", "p0", "p1"]


def test_min_score_is_a_fused_scale_filter_and_scores_keep_the_invariant():
    _seed()
    r = memory.recall(OWNER, Q, kinds=["episodic"], size=5, as_of="2023-05-30", rerank=True)
    for h in r["results"]:
        assert abs(h["score"] - (h["rrf"] + h["recency"])) < 1e-9, h
    top = max(h["score"] for h in r["results"])
    f = memory.recall(OWNER, Q, kinds=["episodic"], size=5, as_of="2023-05-30", rerank=True, min_score=top - 1e-6)
    assert f["results"] and f["abstained"] is False


def test_rerank_supersedes_the_recency_ordering_by_design():
    """Documented interaction: recency reorders near-ties in the FUSED list; with rerank on the
    window order is the cross-encoder's. `score` still equals rrf + recency (it is not rewritten)."""
    _wipe()
    old = "quarterly warehouse relocation budget plan details for the shard rebalance"
    new = "shard rebalance"
    memory.retain(OWNER, "semantic", old, occurred_at="2024-01-01T00:00:00Z")
    memory.retain(OWNER, "semantic", new, occurred_at="2026-10-03T00:00:00Z")
    r = memory.recall(OWNER, "quarterly warehouse relocation budget plan", kinds=["semantic"], size=5, rerank=True)
    assert r["reranked"] is True
    assert r["results"][0]["text"] == old and all("rerank_rank" in h for h in r["results"])


def test_items_the_endpoint_did_not_return_never_carry_a_rerank_score(monkeypatch):
    # endpoint returns only p0..p2 of 12 with scores; items beyond depth AND the omitted head
    # items must not claim a cross-encoder score they never got
    def leaky(query, hits, top_n=5, model=None):
        return {"hits": [dict(h, rerank_score=5.0) for h in hits[:2]]
                + [dict(h, rerank_score=1.0) for h in hits[4:]], "reranked": True}   # drops hits[2:4]
    monkeypatch.setattr(reranker, "rerank", leaky)
    items = [dict(h, rerank_score=7.0) for h in _items(12)]      # stale scores from some earlier pass
    new, _ = memory._rerank_fused(Q, items, depth=8)
    by = {h["id"]: h for h in new}
    assert "rerank_score" in by["p0"] and "rerank_score" in by["p5"]
    for tail_id in ("p8", "p9", "p10", "p11"):
        assert "rerank_score" not in by[tail_id], tail_id            # beyond depth
    for omitted_id in ("p2", "p3"):                                  # in `head`, not returned
        assert "rerank_score" not in by[omitted_id], omitted_id


def test_http_rerank_true_actually_reaches_recall_and_default_stays_off(monkeypatch):
    """Opt-in contract end to end: with the default OFF, only a plumbed `rerank: true` can turn it
    on. (rerank=False alone cannot detect a dropped field once the default is off.)"""
    from fastapi.testclient import TestClient
    from app.main import app
    from app import auth
    _seed()
    monkeypatch.setattr(reranker, "rerank", _fake([]))      # a reranker that "works"
    c = TestClient(app); key = auth.create_key(OWNER)

    def call(**extra):
        r = c.post("/memory/recall", headers={"X-API-Key": key},
                   json=dict(query=Q, kinds=["episodic"], size=3, as_of="2023-05-30", **extra))
        assert r.status_code == 200, r.text
        return r.json()["reranked"]

    assert call() is False                       # default: off
    assert call(rerank=True) is True             # request opt-in
    assert call(rerank=False) is False
    # the MCP/tool path takes the same field
    from app import main as _m
    assert "rerank" in _m.RecallIn.model_fields if hasattr(_m.RecallIn, "model_fields") else True
