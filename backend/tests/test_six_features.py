"""Six-feature validation: graph, temporal, pages, mental models, reflect multi-round,
rerank honest-fallback. Run: python tests/test_six_features.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, pages as pages_mod, mental_models as mm, temporal

O = "sixfeat"


def setup():
    from app.store import es, idx
    for kind in ("semantic", "episodic"):
        r = es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
               {"query": {"term": {"owner_id": O}}})
        assert r.get("deleted", 0) >= 0


def test_graph():
    setup()
    memory.retain(O, "semantic", "Bob joined Acme as a platform engineer")
    memory.retain(O, "semantic", "Acme headquarters are in Krakow, Poland")
    memory.retain(O, "semantic", "Unrelated fact about pasta carbonara recipes")
    res = memory.recall(O, "where does Bob work", size=5)
    hits = res["fused"]
    # hop-2: Krakow fact shares 'acme' with the Bob-acme fact but no query words
    assert any("Krakow" in h["text"] for h in hits), [h["text"] for h in hits]
    assert any("Bob" in h["text"] for h in hits)
    print("GRAPH: PASS")


def test_temporal():
    w = temporal.parse_window("what happened in 2024?")
    assert w and w[0].startswith("2024-01-01") and w[1].startswith("2024-12-31"), w
    b = temporal.spread_buckets(*w, n=4)
    assert len(b) == 4 and b[0][0] < b[1][0] < b[2][0] < b[3][0], b
    res = memory.recall(O, "what happened in 2024", size=8)
    assert "temporal" in res and res["temporal"]["buckets"] == 4, res.get("temporal")
    print("TEMPORAL: PASS", res["temporal"]["window"])


def test_pages():
    setup()
    memory.retain(O, "semantic", "The omniroute gateway heap limit is 12288MB")
    memory.retain(O, "semantic", "OmniRoute watchdog restarts at 10200MB heap")
    r = memory.recall(O, "omniroute heap", kinds=["semantic"], size=50)
    facts = [{"id": h["id"], "text": h["text"], "occurred_at": h.get("occurred_at")}
             for h in r["by_kind"]["semantic"]]
    p = pages_mod.refresh_page(O, "omniroute", facts)
    assert p["fact_count"] >= 2 and "12288MB" in p["body"], p
    got = pages_mod.get_page(O, "omniroute")
    assert got and got["scope"] == "omniroute"
    assert all(sid in [f["id"] for f in facts] for sid in p["source_ids"])
    print("PAGES: PASS", p["fact_count"], "facts")


def test_mental_models():
    mm.upsert_model(O, "evals always run where", "Evals always run on Azure VMs via suite_sweep.py, never local Mac.")
    m = mm.match_model(O, "where do we run evals?")
    assert m and "Azure" in m["summary"], m
    res = memory.recall(O, "where do we run evals")
    assert "mental_model" in res, list(res)
    print("MENTAL-MODELS: PASS")


def test_reflect_multiround():
    res = memory.reflect(O, "gateway heap limits", llm_answer_fn=False)
    assert "queries" in res and res["queries"], res
    assert res["queries"][0] == "gateway heap limits"
    print("REFLECT: PASS queries=%d synthesized=%s" % (len(res["queries"]), res["synthesized"]))


def test_rerank():
    from app import reranker
    hits = [{"id": "a", "text": "heap limit is 12288MB"},
            {"id": "b", "text": "pasta carbonara recipe"}]
    out = reranker.rerank("gateway heap limit", hits, top_n=2)
    assert "reranked" in out  # True (cluster has trial) or honest False with note
    if out["reranked"]:
        assert out["hits"][0]["id"] == "a", out
    print("RERANK: PASS reranked=%s" % out["reranked"])


if __name__ == "__main__":
    test_graph()
    test_temporal()
    test_pages()
    test_mental_models()
    test_reflect_multiround()
    test_rerank()
    print("SIX-FEATURES: ALL PASS")
