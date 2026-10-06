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


def test_legitimate_paths_pass():
    _ok(f"/{P}episodic/_doc")
    _ok(f"/{P}semantic/_search")
    _ok(f"/{P}semantic/_delete_by_query?refresh=true")
    _ok(f"/{P}semantic,{P}episodic/_msearch")
    _ok("/_cat/indices")
    _ok("/_cluster/health")
    _ok("/_inference/text_embedding/x")
    _ok("")
