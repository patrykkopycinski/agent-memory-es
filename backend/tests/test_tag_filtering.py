"""Tag filtering end to end against the live (test-prefixed) Elasticsearch.

Covers: stored shape, caller tags, LLM extraction via a stubbed extractor (fail-closed), each filter
clause, the fuzzy resolver against the stored vocabulary, abstention, composition with the existing
recall knobs, owner isolation, and the BACKWARDS-COMPATIBILITY guarantee (no tags / labels / filter
=> identical stored documents and identical ES request bodies).

Fixtures use invented neutral subjects (fruit, colours); no benchmark vocabulary.

    docker compose ... run --rm --no-deps -v <repo>:/srv -w /srv/backend ames-backend \
        python -m pytest tests/test_tag_filtering.py -q
"""
import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, tagfilter as tf  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qtagfilter"
OTHER = "qtagfilter-other"


def _wipe():
    for kind in ("semantic", "episodic", "procedural"):
        for o in (OWNER, OTHER):
            es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
               {"query": {"term": {"owner_id": o}}})


@pytest.fixture(autouse=True)
def clean():
    _wipe()
    yield
    _wipe()


def _put(text, tags=None, owner=OWNER, group=None, kind="semantic", **kw):
    return memory.retain(owner, kind, text, doc_group=group, tags=tags, **kw)


def _ids(r):
    """doc_groups of THIS test's owner only: the index is shared with other test files whose
    common-visibility docs are legitimately visible to a filter that only excludes (`none`)."""
    return [h.get("doc_group") or h["id"] for h in r["results"] if h["owner_id"] == OWNER]


# ── stored shape + migration ─────────────────────────────────────────────────

def test_mapping_has_a_keyword_tags_field_on_every_kind():
    for kind in ("semantic", "episodic", "procedural"):
        m = es("GET", f"/{idx(kind)}/_mapping")[idx(kind)]["mappings"]["properties"]
        assert m["tags"] == {"type": "keyword"}


def test_ensure_indices_adds_tags_to_an_index_created_before_the_field_existed():
    from app import store
    probe = store.PREFIX + "am_migprobe"
    old = copy.deepcopy(store.MAPPINGS)
    del old["mappings"]["properties"]["tags"]
    try:
        es("DELETE", "/" + probe)
    except RuntimeError:
        pass
    es("PUT", "/" + probe, old)
    es("PUT", f"/{probe}/_mapping", {"properties": {"tags": {"type": "keyword"}}})   # what ensure_indices does
    assert es("GET", f"/{probe}/_mapping")[probe]["mappings"]["properties"]["tags"]["type"] == "keyword"
    es("DELETE", "/" + probe)


def test_caller_tags_are_normalised_deduped_and_stored_on_every_passage():
    long_text = ("apple orchard notes. " * 200)
    r = _put(long_text, tags=["Fruit:Apple", "fruit:apple", "Colour:Red_Green"], group="g1")
    assert r["passages"] > 1 and r["tags"] == ["fruit:apple", "colour:red green"]
    docs = es("POST", f"/{idx('semantic')}/_search", {"size": 50, "query": {"term": {"doc_group": "g1"}}})
    assert len(docs["hits"]["hits"]) == r["passages"]
    for h in docs["hits"]["hits"]:
        assert h["_source"]["tags"] == ["fruit:apple", "colour:red green"]


def test_no_tags_means_no_tags_key_in_the_stored_document_and_response():
    r = _put("plain note about nothing in particular", group="g2")
    assert "tags" not in r and "extraction" not in r
    src = es("GET", f"/{idx('semantic')}/_doc/{r['_id']}")["_source"]
    assert "tags" not in src


def test_malformed_caller_tags_are_rejected_before_anything_is_written():
    with pytest.raises(ValueError):
        _put("x note", tags=["nocolon"], group="g3")
    n = es("POST", f"/{idx('semantic')}/_count", {"query": {"term": {"owner_id": OWNER}}})["count"]
    assert n == 0


# ── extraction (stubbed LLM): fail closed, dedup first ───────────────────────

GROUPS = [
    {"key": "name", "type": "multi-text", "description": "names of the subject"},
    {"key": "state", "type": "value", "optional": False, "description": "is it current",
     "values": [{"value": "current"}, {"value": "historical"}]},
]


def test_labels_extract_into_stored_tags(monkeypatch):
    seen = []

    def fake(system, user):
        seen.append(user)
        return {"name": ["Strawberry", "strawberries"], "state": "current"}
    monkeypatch.setattr("app.llm.chat_json", fake)
    r = _put("A note about growing strawberries on a balcony, kept as a reminder.", labels=GROUPS, group="g4")
    assert r["tags"] == ["name:strawberry", "name:strawberries", "state:current"]
    assert r["extraction"] == {"missing": [], "capped": False, "truncated": False}
    assert len(seen) == 1                                    # one call, all groups
    src = es("GET", f"/{idx('semantic')}/_doc/{r['_id']}")["_source"]
    assert src["tags"] == r["tags"]


def test_caller_tags_and_extracted_tags_merge_without_duplicates(monkeypatch):
    monkeypatch.setattr("app.llm.chat_json", lambda s, u: {"name": ["Plum"], "state": "current"})
    r = _put("Plums ripen late in the season, according to the garden diary.",
             tags=["name:plum", "source:diary"], labels=GROUPS, group="g5")
    assert r["tags"] == ["name:plum", "source:diary", "state:current"]


def test_failed_extraction_refuses_the_write_and_writes_nothing(monkeypatch):
    def boom(system, user):
        raise RuntimeError("llm down")
    monkeypatch.setattr("app.llm.chat_json", boom)
    with pytest.raises(tf.ExtractionError):
        _put("A note that cannot be labelled right now, so must not be stored.", labels=GROUPS, group="g6")
    n = es("POST", f"/{idx('semantic')}/_count", {"query": {"term": {"owner_id": OWNER}}})["count"]
    assert n == 0


def test_exact_duplicate_write_makes_no_llm_call(monkeypatch):
    calls = []
    monkeypatch.setattr("app.llm.chat_json", lambda s, u: calls.append(1) or {"name": ["x"], "state": "current"})
    text = "A distinctive sentence about gooseberries that is written twice, identically."
    _put(text, labels=GROUPS, group="g7")
    again = _put(text, labels=GROUPS, group="g7b")
    assert again["deduped"] is True and calls == [1]


def test_required_group_without_a_valid_value_is_reported_and_untagged(monkeypatch):
    monkeypatch.setattr("app.llm.chat_json", lambda s, u: {"name": ["Fig"], "state": "unknown"})
    r = _put("Figs and their ambiguous status in this particular note about fruit.", labels=GROUPS, group="g8")
    assert r["tags"] == ["name:fig"] and r["extraction"]["missing"] == ["state"]


def test_no_labels_never_touches_the_llm(monkeypatch):
    monkeypatch.setattr("app.llm.chat_json", lambda s, u: pytest.fail("LLM must not be called"))
    _put("A note written with no labels requested at all, about quinces.", group="g9")


# ── recall filter ─────────────────────────────────────────────────────────────

def _corpus():
    _put("The orchard grows pears and the pears are sweet in autumn.", tags=["name:pear", "state:current"], group="pear-1")
    _put("An older note: pears used to be stored in the cellar all winter.", tags=["name:pear", "state:historical"], group="pear-old")
    _put("Cherries bloom early and the cherries need netting against birds.", tags=["name:cherry", "state:current"], group="cherry-1")
    _put("Strawberries sprawl across the bed and the strawberries want straw.", tags=["name:strawberry", "state:current"], group="straw-1")
    _put("A note about the weather that carries no tags whatsoever in this corpus.", group="untagged")


def test_all_clause_requires_every_tag():
    _corpus()
    r = memory.recall(OWNER, "pears", filter={"all": ["name:pear", "state:current"]})
    assert _ids(r) == ["pear-1"]


def test_any_clause_is_or_within_a_group_and_and_across_groups():
    _corpus()
    r = memory.recall(OWNER, "orchard fruit", size=10, filter={"any": [["name:pear", "name:cherry"], ["state:current"]]})
    assert sorted(_ids(r)) == ["cherry-1", "pear-1"]


def test_none_clause_excludes():
    _corpus()
    r = memory.recall(OWNER, "pears", size=10, filter={"none": ["state:historical"]})
    assert "pear-old" not in _ids(r) and "pear-1" in _ids(r)


def test_narrow_exact_restricts_to_the_named_subject():
    _corpus()
    r = memory.recall(OWNER, "tell me about cherries", size=10,
                      filter={"narrow_any": [{"tags": ["name:cherry"], "resolve": "exact"}]})
    assert _ids(r) == ["cherry-1"] and r["filter"]["resolved"] == {"name:cherry": ["name:cherry"]}


def test_narrow_fuzzy_resolves_a_misspelling_against_stored_values():
    _corpus()
    r = memory.recall(OWNER, "strawbery", size=10,
                      filter={"narrow_any": [{"tags": ["name:strawbery"], "resolve": "fuzzy"}]})
    assert _ids(r) == ["straw-1"]
    assert r["filter"]["resolved"] == {"name:strawbery": ["name:strawberry"]}


def test_fuzzy_resolution_only_sees_tags_stored_for_the_requested_key():
    _corpus()
    _put("Unrelated: a colour note.", tags=["shade:strawberry"], group="shade-1")
    r = memory.recall(OWNER, "strawbery", size=10,
                      filter={"narrow_any": [{"tags": ["name:strawbery"], "resolve": "fuzzy"}]})
    assert "shade-1" not in _ids(r)


# ── abstention ────────────────────────────────────────────────────────────────

def test_narrow_that_names_nothing_stored_abstains_with_no_results():
    _corpus()
    r = memory.recall(OWNER, "tell me about volcanoes", size=10,
                      filter={"narrow_any": [{"tags": ["name:volcano"], "resolve": "fuzzy"}]})
    assert r["results"] == [] and r["abstained"] is True and r["filter"]["applied"] is True


def test_abstention_never_falls_back_to_unfiltered_topk():
    _corpus()
    unfiltered = memory.recall(OWNER, "pears", size=10)
    assert len(unfiltered["results"]) > 0
    r = memory.recall(OWNER, "pears", size=10, filter={"all": ["name:does-not-exist"]})
    assert r["results"] == [] and r["abstained"] is True and r["fused"] == []


def test_empty_narrow_resolution_skips_elasticsearch_entirely(monkeypatch):
    _corpus()
    real = memory.es
    searches = []

    def spy(method, path, body=None):
        if path.endswith("/_search") and body and "aggs" not in body:
            searches.append(path)
        return real(method, path, body)
    monkeypatch.setattr(memory, "es", spy)
    r = memory.recall(OWNER, "pears", filter={"narrow_any": [{"tags": ["name:qqqq"], "resolve": "fuzzy"}]})
    assert r["abstained"] is True and searches == []


def test_present_but_empty_narrow_abstains():
    _corpus()
    r = memory.recall(OWNER, "pears", filter={"narrow_any": []})
    assert r["results"] == [] and r["abstained"] is True


def test_filter_matching_something_is_not_abstained():
    _corpus()
    r = memory.recall(OWNER, "pears", filter={"all": ["name:pear"]})
    assert r["abstained"] is False and len(r["results"]) == 2


# ── composition + isolation ───────────────────────────────────────────────────

def test_filter_composes_with_min_score_and_per_doc_and_size():
    _corpus()
    r = memory.recall(OWNER, "pears", size=1, per_doc=1, filter={"all": ["name:pear"]})
    assert len(r["results"]) == 1
    r = memory.recall(OWNER, "pears", min_score=99.0, filter={"all": ["name:pear"]})
    assert r["results"] == [] and r["abstained"] is True


def test_filter_is_owner_scoped_for_results_and_for_the_fuzzy_vocabulary():
    _corpus()
    _put("Another owner writes about durian only.", tags=["name:durian"], owner=OTHER, group="durian-1")
    r = memory.recall(OWNER, "durian", filter={"narrow_any": [{"tags": ["name:durian"], "resolve": "exact"}]})
    assert r["results"] == [] and r["abstained"] is True
    r = memory.recall(OWNER, "durien", filter={"narrow_any": [{"tags": ["name:durien"], "resolve": "fuzzy"}]})
    assert r["results"] == [] and r["filter"]["resolved"] == {}        # the vocabulary is not shared


def test_malformed_filter_is_a_filtererror_not_an_unfiltered_search():
    with pytest.raises(tf.FilterError):
        memory.recall(OWNER, "pears", filter={"all": ["nocolon"]})
    with pytest.raises(tf.FilterError):
        memory.recall(OWNER, "pears", filter={"bogus": 1})


def test_filtered_recall_skips_the_graph_arm_and_the_mental_model_tier(monkeypatch):
    _corpus()
    called = []
    import app.graph as g
    monkeypatch.setattr(g, "graph_expand", lambda *a, **k: called.append("graph") or [])
    import app.mental_models as mm
    monkeypatch.setattr(mm, "match_model", lambda *a, **k: called.append("model") or {"leak": True})
    r = memory.recall(OWNER, "pears", filter={"all": ["name:pear"]})
    assert called == [] and "mental_model" not in r
    called.clear()
    r = memory.recall(OWNER, "pears")                  # unfiltered path keeps both
    assert called == ["graph", "model"] and r["mental_model"] == {"leak": True}


def test_temporal_arm_is_restricted_by_the_filter():
    _put("Pears were harvested last week and weighed.", tags=["name:pear"], group="t-pear", kind="episodic",
         occurred_at="2026-09-28T10:00:00Z")
    _put("Cherries were harvested last week and counted.", tags=["name:cherry"], group="t-cherry", kind="episodic",
         occurred_at="2026-09-28T11:00:00Z")
    r = memory.recall(OWNER, "what did we harvest last week", size=10, as_of="2026-10-03",
                      filter={"all": ["name:pear"]})
    assert set(_ids(r)) == {"t-pear"}


# ── backwards compatibility: no tags / labels / filter => unchanged ──────────

def _capture(monkeypatch):
    bodies = []
    real = memory.es

    def spy(method, path, body=None):
        if path.endswith("/_search"):
            bodies.append(copy.deepcopy(body))
        return real(method, path, body)
    monkeypatch.setattr(memory, "es", spy)
    return bodies


def test_recall_without_a_filter_sends_the_exact_pre_change_es_bodies(monkeypatch):
    _corpus()
    bodies = _capture(monkeypatch)
    memory.recall(OWNER, "pears", size=8)
    main = [b for b in bodies if "retriever" in b or "query" in b][:3]
    assert main, "no search captured"
    vis = {"bool": {"should": [{"term": {"owner_id": OWNER}}, {"terms": {"visibility": ["team", "common"]}}]}}
    base = [{"term": {"active": True}}, vis]
    for b in main:
        if "retriever" in b:
            std, knn = b["retriever"]["rrf"]["retrievers"]
            assert std["standard"]["query"]["bool"]["filter"] == base
            assert "must_not" not in std["standard"]["query"]["bool"]
            assert knn["knn"]["filter"] == base                       # a plain list, as before
        else:
            assert b["query"]["bool"]["filter"] == base and "must_not" not in b["query"]["bool"]
    assert not any("aggs" in b for b in bodies)                      # no vocabulary scan


def test_recall_without_a_filter_has_the_same_response_keys_as_before():
    _corpus()
    r = memory.recall(OWNER, "pears")
    assert set(r) >= {"query", "results", "by_kind", "fused", "abstained", "doc_groups", "reranked"}
    assert "filter" not in r and r["abstained"] is False


def test_empty_filter_object_is_exactly_the_unfiltered_path():
    _corpus()
    a = memory.recall(OWNER, "pears", size=5)
    b = memory.recall(OWNER, "pears", size=5, filter={})
    c = memory.recall(OWNER, "pears", size=5, filter={"all": [], "any": [[]], "none": []})
    key = lambda r: [(h["id"], round(h["score"], 9)) for h in r["results"]]   # noqa: E731
    assert key(a) == key(b) == key(c) and "filter" not in b and "filter" not in c


def test_retain_without_tags_or_labels_stores_the_same_fields_as_before():
    r = _put("A completely ordinary memory, written the way every existing caller writes it.", group="g-compat")
    src = es("GET", f"/{idx('semantic')}/_doc/{r['_id']}")["_source"]
    assert set(src) == {"kind", "owner_id", "visibility", "text", "text_hash", "doc_group", "passage_index",
                        "passages_total", "entities", "occurred_at", "active"}   # == main@615dde0, probed
    assert set(r) == {"_id", "ids", "doc_group", "passages", "deduped", "dedup_reason", "owner_id",
                      "visibility", "occurred_at", "text_hash"}


# ── the BM25-only fallback (embedder unavailable) honours the filter too ─────

def _no_embedder(monkeypatch):
    def down(texts):
        raise RuntimeError("embedder down")
    monkeypatch.setattr("app.embeddings.embed", down)


def test_bm25_only_fallback_still_applies_the_filter(monkeypatch):
    _corpus()
    _no_embedder(monkeypatch)
    bodies = _capture(monkeypatch)
    r = memory.recall(OWNER, "pears", size=10, filter={"all": ["name:pear"], "none": ["state:historical"]})
    assert _ids(r) == ["pear-1"]
    plain = [b for b in bodies if "query" in b and "retriever" not in b]
    assert plain, "the fallback (no rrf retriever) body was not exercised"
    for b in plain:
        assert {"term": {"tags": "name:pear"}} in b["query"]["bool"]["filter"]
        assert b["query"]["bool"]["must_not"] == [{"terms": {"tags": ["state:historical"]}}]


def test_bm25_only_fallback_without_a_filter_is_unchanged(monkeypatch):
    _corpus()
    _no_embedder(monkeypatch)
    bodies = _capture(monkeypatch)
    memory.recall(OWNER, "pears", size=10)
    plain = [b for b in bodies if "query" in b and "retriever" not in b]
    assert plain and all("must_not" not in b["query"]["bool"] for b in plain)
    assert all(len(b["query"]["bool"]["filter"]) == 2 for b in plain)
