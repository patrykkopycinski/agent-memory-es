"""Unit tests for store._guard_path (AMES_TEST_GUARD=1 request filter).

Mutation-proof target: EVERY comma-separated index name must be checked —
'/amtest_x,am_semantic/_delete_by_query' must be refused even though the
first index passes the prefix check. Cluster-level APIs are allowlisted,
not blanket-permitted: /_bulk, /_reindex, /_aliases, /_mget,
/_delete_by_query at cluster level are all refused.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (must precede app imports)

from app import store


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


# --- every comma-separated index name is checked, wildcards/_all refused ---
_refused("/amtest_x,am_semantic/_delete_by_query")   # mixed: 2nd name is prod
_refused("/am_semantic,amtest_x/_search")            # order must not matter
_refused("/_all/_delete_by_query")
_refused("/amtest_*/_delete_by_query")
_refused("/am_*/_search")
_refused("/amtest_x,*/_search")

# --- cluster-level: allowlist only, dangerous bulk APIs refused ---
_refused("/_bulk")
_refused("/_reindex")
_refused("/_aliases")
_refused("/_mget")
_refused("/_delete_by_query")
_refused("/_all/_search")
_refused_get("/_bulk")
_refused_get("/_reindex")

# --- legitimate paths still pass ---
_ok("/amtest_episodic/_doc")
_ok("/amtest_semantic/_search")
_ok("/amtest_semantic/_delete_by_query?refresh=true")
_ok("/amtest_semantic,amtest_episodic/_msearch")
_ok("/_cat/indices")
_ok("/_cluster/health")
_ok("/_inference/text_embedding/x")
_ok("")

# --- prefix-mismatched single index refused (original behaviour) ---
_refused("/am_semantic/_search")
_refused("/am_models/_count")

print("test_guard.py: ALL PASS")
