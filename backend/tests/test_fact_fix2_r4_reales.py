"""Round 4 regression tests; use FIX2_ES on an isolated ES instance."""
import threading

import pytest

from app import facts
from test_fact_fix2_r3_reales import (
    BACKFILLS, QUEUE, doc, runner, seed, setup,
)


@pytest.mark.parametrize("fails", [False, True], ids=["completed", "error-pending"])
def test_finish_retries_heartbeat_conflict(monkeypatch, setup, fails):
    seed()
    renewals = []

    def process(*args, **kwargs):
        if fails:
            raise ValueError("extraction failed")

    runner(monkeypatch, process)
    monkeypatch.setattr(facts, "NO_PROGRESS_LIMIT", 2)
    monkeypatch.setattr(facts.time, "sleep", lambda _: None)
    transport = facts.es

    def conflicting_es(method, path, body=None):
        if (method == "POST" and f"/{QUEUE}/_update/job?" in path
                and body and body.get("doc", {}).get("state") in ("completed", "pending")
                and not renewals):
            renewals.append(facts._renew_job_lease("job", doc(QUEUE, "job")["lease_token"]))
        return transport(method, path, body)

    monkeypatch.setattr(facts, "es", conflicting_es)
    assert facts.run_backfill_once()
    job = doc(QUEUE, "job")
    assert renewals == [True]
    if fails:
        assert job["state"] == "pending" and job["attempts"] == 1
        assert doc(BACKFILLS, "bf")["state"] == "failed"
        # The retrying job is delayed; the backfill reports no progress promptly.
    else:
        assert job["state"] == "completed"
        assert doc(BACKFILLS, "bf")["state"] == "completed"


def test_abort_requeues_without_charging_attempt(monkeypatch, setup):
    seed()
    abort = threading.Event()
    monkeypatch.setattr(facts, "process", lambda *a, **kw: abort.set())
    assert facts.run_once("owner", "job", on_claim=lambda *_: None, abort=abort)
    job = doc(QUEUE, "job")
    assert job["state"] == "pending"
    assert job["attempts"] == 0
    assert facts._parse_iso(job["next_at"]) <= facts._now()


def test_abort_before_semantic_create(monkeypatch, setup):
    semantic = facts.idx("semantic")
    abort = threading.Event()
    puts = []
    original = facts.es

    def transport(method, path, body=None):
        if path == "/" + facts.idx("episodic") + "/_doc/src":
            return {"_source": {"owner_id": "owner", "active": True,
                                "visibility": "private", "occurred_at": "2024-01-01T00:00:00Z",
                                "doc_group": "src"}}
        if method == "GET" and path.startswith("/" + semantic + "/_doc/"):
            raise RuntimeError("-> 404: absent")
        if method == "PUT" and path.startswith("/" + semantic + "/_create/"):
            puts.append(path)
            return {}
        return original(method, path, body)

    monkeypatch.setattr(facts, "es", transport)
    monkeypatch.setattr(facts, "_passages", lambda *a: [("src", "A fact.")])
    monkeypatch.setattr(facts, "_document_facts", lambda *a: ["A fact."])
    monkeypatch.setattr(facts.embeddings, "embed", lambda *a: [[0.1]])
    monkeypatch.setattr(facts, "_replacement_ids", lambda *a: abort.set() or [])
    with pytest.raises(RuntimeError, match="heartbeat abort"):
        facts.process("src", "owner", abort=abort)
    assert puts == []
