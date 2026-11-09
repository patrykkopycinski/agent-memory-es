"""_log_mutation ensures the am_mutations index once per process, not on every call."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app import memory
from app.store import PREFIX, es

INDEX = f"{PREFIX}am_mutations"
OWNER = "mutlog_owner"


@pytest.fixture()
def calls(monkeypatch):
    """Record every ES request issued through memory.es while delegating to the real one."""
    seen = []

    def spy(method, path, body=None, *a, **k):
        seen.append((method, path))
        return es(method, path, body, *a, **k)

    monkeypatch.setattr(memory, "es", spy)
    memory._forget_mutations_index()
    yield seen
    memory._forget_mutations_index()
    memory._ensure_mutations_index()      # leave the shared index with the pinned mapping


def _drop_index():
    es("DELETE", f"/{INDEX}?ignore_unavailable=true")


def _drop_template():
    # an index created before the template existed does not inherit its mapping
    try:
        es("DELETE", f"/_index_template/{INDEX}")
    except RuntimeError as exc:
        if "-> 404" not in str(exc):
            raise


def test_repeat_calls_issue_the_setup_requests_at_most_once(calls):
    _drop_index()
    ids = [memory._log_mutation("update", OWNER, "semantic", f"d{i}", {}) for i in range(4)]
    assert all(ids)
    creates = [c for c in calls if c[0] == "PUT" and c[1] == f"/{INDEX}"]
    mappings = [c for c in calls if c[1].endswith("/_mapping")]
    gets = [c for c in calls if c[0] == "GET"]
    assert len(creates) == 1 and len(mappings) == 0 and len(gets) == 1, calls
    assert len([c for c in calls if c[0] == "POST"]) == 4


def test_existing_index_is_never_created_over(calls):
    memory._log_mutation("update", OWNER, "semantic", "warm", {})     # makes sure it exists
    calls.clear()
    memory._forget_mutations_index()
    for i in range(3):
        memory._log_mutation("update", OWNER, "semantic", f"d{i}", {})
    assert [c for c in calls if c[0] == "PUT" and c[1] == f"/{INDEX}"] == [], calls
    assert len([c for c in calls if c[0] == "GET"]) == 1


def test_index_missing_the_pinned_fields_gets_a_mapping_update_once(calls):
    _drop_index()
    _drop_template()
    es("PUT", f"/{INDEX}", {"mappings": {"properties": {"intent": {"type": "keyword"}}}})
    for i in range(3):
        memory._log_mutation("update", OWNER, "semantic", f"d{i}", {})
    mappings = [c for c in calls if c[1].endswith("/_mapping")]
    assert len(mappings) == 1, calls
    props = es("GET", f"/{INDEX}/_mapping")[INDEX]["mappings"]["properties"]
    assert props["doc_id"]["type"] == "keyword" and props["at"]["type"] == "date"


def test_index_deleted_underneath_is_re_ensured_on_the_next_call(calls, monkeypatch):
    memory._log_mutation("update", OWNER, "semantic", "warm", {})
    assert memory._mutations_ready == INDEX
    _drop_index()
    # a cluster with auto-create disabled answers the write with index_not_found
    real, state = memory.es, {"n": 0}

    def no_autocreate(method, path, body=None, *a, **k):
        if method == "POST" and path.startswith(f"/{INDEX}/_doc"):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError(f"ES POST {path} -> 404: index_not_found_exception")
        return real(method, path, body, *a, **k)

    monkeypatch.setattr(memory, "es", no_autocreate)
    calls.clear()
    assert memory._log_mutation("update", OWNER, "semantic", "after-delete", {})
    assert len([c for c in calls if c[0] == "PUT" and c[1] == f"/{INDEX}"]) == 1
    assert memory._mutations_ready == INDEX
    props = es("GET", f"/{INDEX}/_mapping")[INDEX]["mappings"]["properties"]
    assert props["doc_id"]["type"] == "keyword"       # re-created with the pinned mapping, not dynamic


def test_index_recreated_by_a_write_gets_the_pinned_mapping(calls):
    memory._log_mutation("update", OWNER, "semantic", "warm", {})
    assert memory._mutations_ready == INDEX
    _drop_index()
    # real cluster, auto-create on: the write recreates the index itself
    assert memory._log_mutation("update", OWNER, "semantic", "after-delete", {})
    props = es("GET", f"/{INDEX}/_mapping")[INDEX]["mappings"]["properties"]
    assert props["doc_id"]["type"] == "keyword" and props["at"]["type"] == "date"
    hits = es("POST", f"/{INDEX}/_search", {"query": {"term": {"doc_id": "after-delete"}}})
    assert hits["hits"]["total"]["value"] >= 1


def test_index_template_is_installed_once_per_process(calls):
    for i in range(4):
        memory._log_mutation("update", OWNER, "semantic", f"d{i}", {})
    templates = [c for c in calls if c[1] == f"/_index_template/{INDEX}"]
    assert len(templates) == 1 and templates[0][0] == "PUT", calls
    assert INDEX in es("GET", f"/_index_template/{INDEX}")["index_templates"][0]["index_template"]["index_patterns"]


def test_conflicting_existing_mapping_fails_at_most_once(calls, caplog):
    _drop_index()
    _drop_template()
    es("PUT", f"/{INDEX}", {"mappings": {"properties": {
        "intent": {"type": "keyword"}, "doc_id": {"type": "text"}}}})
    try:
        with caplog.at_level("WARNING", logger="ames.mutations"):
            ids = [memory._log_mutation("update", OWNER, "semantic", f"c{i}", {}) for i in range(5)]
        assert all(ids)
        assert len([c for c in calls if c[1].endswith("/_mapping")]) <= 1, calls
        assert len([c for c in calls if c[0] == "GET" and c[1].startswith(f"/{INDEX}?")]) == 1, calls
        assert memory._mutations_ready == INDEX
        assert len([r for r in caplog.records if r.name == "ames.mutations"]) == 1
    finally:
        _drop_index()


def test_non_strict_failure_is_swallowed_and_strict_raises(monkeypatch):
    memory._forget_mutations_index()

    def fail(method, path, body=None, *a, **k):
        raise RuntimeError(f"ES {method} {path} -> 503: unavailable")

    monkeypatch.setattr(memory, "es", fail)
    assert memory._log_mutation("update", OWNER, "semantic", "x", {}) is None
    with pytest.raises(RuntimeError):
        memory._log_mutation("update", OWNER, "semantic", "x", {}, strict=True)


def test_strict_raises_when_the_write_stays_index_not_found(monkeypatch):
    real = memory.es

    def never(method, path, body=None, *a, **k):
        if method == "POST" and path.startswith(f"/{INDEX}/_doc"):
            raise RuntimeError(f"ES {method} {path} -> 404: index_not_found_exception")
        return real(method, path, body, *a, **k)

    monkeypatch.setattr(memory, "es", never)
    assert memory._log_mutation("update", OWNER, "semantic", "x", {}) is None
    with pytest.raises(RuntimeError):
        memory._log_mutation("update", OWNER, "semantic", "x", {}, strict=True)


def test_governance_outcome_update_still_applies(calls):
    from app import governance
    row = memory._log_mutation("update", OWNER, "semantic", "g1", {"a": 1}, strict=True)
    governance._lost(row)
    got = es("GET", f"/{INDEX}/_doc/{row}")["_source"]
    assert got["detail"]["outcome"] == "conflict"


def teardown_module(_m):
    es("POST", f"/{INDEX}/_delete_by_query?refresh=true&ignore_unavailable=true",
       {"query": {"term": {"owner_id": OWNER}}})
    es("DELETE", f"/_index_template/{INDEX}")
