"""Unit tests for store._guard_path (AMES_TEST_GUARD=1 request filter).

Mutation-proof target: EVERY comma-separated index name must be checked —
'/amtest_x,am_semantic/_delete_by_query' must be refused even though the
first index passes the prefix check. Cluster-level APIs are allowlisted,
not blanket-permitted: /_bulk, /_reindex, /_aliases, /_mget,
/_delete_by_query at cluster level are all refused.

Index names are built from store.PREFIX (the suite's AMES_INDEX_PREFIX), so
the assertions track whatever prefix the run actually uses. All checks live
inside test functions: a broken guard must FAIL, not error at collection.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (must precede app imports)

from app import store

P = store.PREFIX  # e.g. "amtest_" in the suite, "" in prod


def _ok(path):
    store._guard_path(path, "GET")  # must not raise


def _refused(path):
    try:
        store._guard_path(path, "POST")
    except RuntimeError:
        return
    raise AssertionError(f"guard ALLOWED dangerous path: {path}")


def _refused_get(path):
    try:
        store._guard_path(path, "GET")
    except RuntimeError:
        return
    raise AssertionError(f"guard ALLOWED dangerous GET path: {path}")


def test_mixed_index_lists_refused():
    # mixed: the second name is a PROD index even though the first is prefixed
    _refused(f"/{P}x,am_semantic/_delete_by_query")
    _refused(f"/am_semantic,{P}x/_search")  # order must not matter
    _refused(f"/{P}x,*/_search")
    _refused(f"/{P}x,/_search")


def test_wildcards_and_all_refused():
    _refused("/_all/_delete_by_query")
    _refused(f"/{P}*/_delete_by_query")
    _refused("/am_*/_search")


def test_cluster_level_allowlist_refuses_bulk_apis():
    _refused("/_bulk")
    _refused("/_reindex")
    _refused("/_aliases")
    _refused("/_mget")
    _refused("/_delete_by_query")
    _refused("/_all/_search")
    _refused_get("/_bulk")
    _refused_get("/_reindex")


def test_index_template_allowed_only_for_the_test_prefix():
    store._guard_path(f"/_index_template/{P}am_mutations", "PUT")
    store._guard_path(f"/_index_template/{P}am_mutations?x=1", "GET")
    if P:
        _refused("/_index_template/am_mutations")
    _refused(f"/_index_template/{P}x/y")
    _refused(f"/_index_template/{P}*")
    _refused("/_index_template/")
    # comma lists and percent-encoded separators/wildcards: ES decodes and splits these
    _refused(f"/_index_template/{P}rx,logs")
    _refused(f"/_index_template/{P}rx,%2A")
    _refused(f"/_index_template/{P}rx%2C%2A")
    _refused(f"/_index_template/{P}r%2A")
    _refused(f"/_index_template/{P}rx%2Clogs")
    _refused(f"/_index_template/{P}rx%252Clogs")      # double-encoded
    _refused(f"/_index_template/{P}rx%2Fy")
    _refused(f"/_index_template/{P}rx,logs?x=1")


def test_percent_encoded_index_lists_refused():
    _refused(f"/{P}rx%2Clogs/_doc")
    _refused(f"/{P}rx%2C%2A/_search")
    _refused(f"/{P}r%2A/_delete_by_query")
    _refused(f"/{P}rx%252Clogs/_doc")
    _refused(f"/{P}rx%2Fy/_doc")
    _refused(f"/am_semantic%2C{P}x/_search")
    _refused_get(f"/{P}rx%2Clogs/_search")
    _refused_get("/%5Fall/_search")
    _refused("/_all%2F_search")


def test_encoded_document_ids_still_pass():
    # ids are percent-encoded by the callers (_q) and sit after the index segment
    _ok(f"/{P}semantic/_doc/a%2Fb%3Fc%2Cd%2A")
    _ok(f"/{P}semantic/_update/a%2Fb?refresh=true")
    store._guard_path(f"/{P}semantic/_doc/a%2Fb", "PUT")


def test_comma_probe_against_real_es_is_refused_and_template_survives():
    probe, other = f"{P}zzprobe", f"{P}zzprobe_other"
    for name in (probe, other):
        store.es("PUT", f"/_index_template/{name}", {
            "index_patterns": [f"{name}_idx"], "template": {"settings": {"number_of_replicas": 0}}})
    try:
        for bad in (f"{probe},{other}", f"{probe}%2C{other}", f"{probe}%2C%2A", f"{probe},%2A"):
            try:
                store.es("DELETE", f"/_index_template/{bad}")
            except RuntimeError as exc:
                assert "ES test guard" in str(exc), exc
            else:
                raise AssertionError(f"guard let through DELETE of {bad}")
        got = store.es("GET", f"/_index_template/{other}")["index_templates"]
        assert [t["name"] for t in got] == [other]
        assert len(store.es("GET", f"/_index_template/{probe}")["index_templates"]) == 1
    finally:
        for name in (probe, other):
            store.es("DELETE", f"/_index_template/{name}")


def test_legitimate_paths_pass():
    _ok(f"/{P}episodic/_doc")
    _ok(f"/{P}semantic/_search")
    _ok(f"/{P}semantic/_delete_by_query?refresh=true")
    _ok(f"/{P}semantic,{P}episodic/_msearch")
    _ok("/_cat/indices")
    _ok("/_cluster/health")
    _ok("/_inference/text_embedding/x")
    _ok("")
