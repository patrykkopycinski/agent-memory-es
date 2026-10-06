"""r7-v4 fix-2 rework f: real-ES lease recovery tests (isolated index prefix, cleaned up).

Requires a live ES at localhost:9268 (the spike cluster). Uses prefix fix2test_ only.
  PYTHONPATH=backend python3 -m pytest backend/tests/test_fact_fix2_reales.py
"""
import datetime as dt
import json
import os
import urllib.error
import urllib.request

import pytest

from app import facts

ES = os.environ.get("FIX2_ES", "http://localhost:9268")


def _http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(ES + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _raw_es(method, path, body=None):
    status, obj = _http(method, path, body)
    if status >= 400:
        raise RuntimeError(f"ES {method} {path} -> {status}: {json.dumps(obj)[:200]}")
    return obj


ISOLATED = {
    "QUEUE": "fix2test_am_fact_jobs",
    "BACKFILLS": "fix2test_am_fact_backfills",
    "CACHE": "fix2test_am_fact_cache",
}


@pytest.fixture()
def isolated_facts(monkeypatch):
    """Point facts' queue/backfill indices at an isolated prefix on real ES."""
    monkeypatch.setattr(facts, "QUEUE", ISOLATED["QUEUE"])
    monkeypatch.setattr(facts, "BACKFILLS", ISOLATED["BACKFILLS"])
    monkeypatch.setattr(facts, "CACHE", ISOLATED["CACHE"])
    monkeypatch.setattr(facts, "es",
                        lambda method, path, body=None: _raw_es(method, path, body))
    _raw_es("PUT", "/" + ISOLATED["QUEUE"], facts.JOB_MAPPING)
    yield
    for idx in ISOLATED.values():
        _es_delete(idx)


def _es_delete(idx):
    try:
        _raw_es("DELETE", "/" + idx)
    except RuntimeError:
        pass


def _seed_job(job_id, owner, state, lease_until, attempts=0, next_at=None):
    now = facts._iso(facts._now())
    _raw_es("PUT", "/%s/_doc/%s?refresh=true" % (ISOLATED["QUEUE"], job_id), {
        "owner_id": owner, "source_id": "src", "state": state, "attempts": attempts,
        "next_at": next_at or now, "created_at": now,
        "lease_until": lease_until, "lease_token": "deadtoken"})


def test_reales_orphaned_job_reclaimed_and_completed(isolated_facts, monkeypatch):
    """Seed running + expired lease (2h old, live stored format); real claim +
    run_once with the LLM/embedding path stubbed; assert completed, lease/token
    changed, and bounded ES calls."""
    job_id, owner = "fix2orphan1", "fix2-owner"
    old = facts._iso(facts._now() - dt.timedelta(hours=2))
    _seed_job(job_id, owner, "running", old)
    before = _raw_es("GET", "/%s/_doc/%s" % (ISOLATED["QUEUE"], job_id))["_source"]
    process_calls = []

    def fake_process(source_id, owner_id):
        process_calls.append((source_id, owner_id))

    monkeypatch.setattr(facts, "process", fake_process)
    es_calls = []

    def counting_es(method, path, body=None):
        es_calls.append(method + " " + path.split("?")[0])
        return _raw_es(method, path, body)

    monkeypatch.setattr(facts, "es", counting_es)
    assert facts.run_once(owner, job_id) is True
    after = _raw_es("GET", "/%s/_doc/%s" % (ISOLATED["QUEUE"], job_id))["_source"]
    assert after["state"] == "completed"
    assert after.get("lease_token") != before["lease_token"]
    assert process_calls == [("src", owner)]
    assert len(es_calls) <= 6, es_calls      # search + update + get + finish: bounded


def test_reales_backfill_claim_lease_is_short(isolated_facts):
    """Orphan-BACKFILL recovery latency: claim_backfill must set a SHORT lease
    (BACKFILL_LEASE_SECONDS, heartbeat-renewed), so a dead holder is reclaimable
    in ~90s, not a full LEASE_SECONDS."""
    owner = "fix2-owner"
    now = facts._iso(facts._now())
    _raw_es("PUT", "/%s/_doc/bf1?refresh=true" % ISOLATED["BACKFILLS"], {
        "owner_id": owner, "visibility": "private", "state": "pending",
        "queued": 0, "created_at": now, "error": "",
        "lease_until": facts._iso(facts._now() - dt.timedelta(hours=2)),
        "lease_token": "dead"})
    claimed = facts.claim_backfill()
    assert claimed is not None
    doc = _raw_es("GET", "/%s/_doc/bf1" % ISOLATED["BACKFILLS"])["_source"]
    delta = (facts._parse_iso(doc["lease_until"]) - facts._now()).total_seconds()
    assert 0 < delta <= facts.BACKFILL_LEASE_SECONDS + 5, delta
    assert delta < facts.LEASE_SECONDS


def test_reales_live_lease_not_stolen(isolated_facts):
    """A running job with a LIVE lease must not be claimable via run_once by id."""
    job_id, owner = "fix2live1", "fix2-owner"
    live = facts._iso(facts._now() + dt.timedelta(hours=1))
    _seed_job(job_id, owner, "running", live)
    assert facts.run_once(owner, job_id) is False
    doc = _raw_es("GET", "/%s/_doc/%s" % (ISOLATED["QUEUE"], job_id))["_source"]
    assert doc["state"] == "running"
    assert doc["lease_token"] == "deadtoken"      # untouched: not stolen


def test_reales_loop_aborts_after_no_progress(monkeypatch):
    """Inner loop: run_once makes no change -> after NO_PROGRESS_LIMIT rounds the
    backfill raises instead of spinning. Pure-ES-shape facade (no network)."""
    owner = "fix2-owner"
    job_id = "fix2stuck1"
    old = facts._iso(facts._now() - dt.timedelta(hours=2))
    stuck_job = {"state": "running", "lease_until": old,
                 "next_at": "2000-01-01T00:00:00Z", "attempts": 3}

    def stuck_es(method, path, body=None):
        if "_search" in path and "episodic" in path:
            return {"hits": {"hits": [
                {"_id": "src1", "sort": ["2024-01-01", "g1"], "_source": {}}]}}
        if method == "GET" and job_id in path:
            return {"_source": dict(stuck_job)}
        return {}

    run_once_calls = {"n": 0}

    def stuck_run_once(owner_id, jid):
        run_once_calls["n"] += 1
        return True          # "ran" but changed nothing on the doc

    monkeypatch.setattr(facts, "es", stuck_es)
    monkeypatch.setattr(facts, "enqueue", lambda o, s: job_id)
    monkeypatch.setattr(facts, "idx", lambda kind: "fix2test_episodic")
    monkeypatch.setattr(facts, "_update_backfill", lambda *a, **k: None)
    monkeypatch.setattr(facts, "run_once", stuck_run_once)
    monkeypatch.setattr(facts.time, "sleep", lambda s: None)   # bounded by N, not time
    with pytest.raises(RuntimeError, match="no progress"):
        facts._run_backfill("bfkey", {"owner_id": owner, "visibility": "private"}, "tok")
    assert run_once_calls["n"] == facts.NO_PROGRESS_LIMIT
