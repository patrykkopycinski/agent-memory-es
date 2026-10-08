"""fix-2 round 2 real-ES recovery tests (H2 a-e, M3, H1). Per-run UUID index
prefix; SKIP (not error) when FIX2_ES is unset or ES is unreachable.
Requires a DEDICATED live ES: set FIX2_ES (no default; never point it at a
shared/production cluster — these tests create, write and delete indices).
  FIX2_ES=http://127.0.0.1:9279 PYTHONPATH=backend python3 -m pytest backend/tests/test_fact_fix2_reales.py
"""
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request
import uuid

import pytest

from app import facts

ES = os.environ.get("FIX2_ES")  # dedicated ES only — no default; unset => skip
PREFIX = f"fix2r2_{uuid.uuid4().hex[:8]}_"
QUEUE = PREFIX + "am_fact_jobs"
BACKFILLS = PREFIX + "am_fact_backfills"


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


def es_reachable():
    if not ES:
        return False
    try:
        with urllib.request.urlopen(ES, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not ES or not es_reachable(),
                                reason="FIX2_ES not set or live ES unreachable")


@pytest.fixture()
def isolated(monkeypatch):
    monkeypatch.setattr(facts, "QUEUE", QUEUE)
    monkeypatch.setattr(facts, "BACKFILLS", BACKFILLS)
    monkeypatch.setattr(facts, "es",
                        lambda method, path, body=None: _raw_es(method, path, body))
    _raw_es("PUT", "/" + QUEUE, facts.JOB_MAPPING)
    _raw_es("PUT", "/" + BACKFILLS, facts.BACKFILL_MAPPING)
    yield
    for idx in (QUEUE, BACKFILLS):
        try:
            _raw_es("DELETE", "/" + idx)
        except RuntimeError:
            pass


def _seed_job(job_id, owner, state, lease_until=None, attempts=0, next_at=None,
              token="deadtoken"):
    now = facts._iso(facts._now())
    _raw_es("PUT", f"/{QUEUE}/_doc/{job_id}?refresh=true", {
        "owner_id": owner, "source_id": "src", "state": state, "attempts": attempts,
        "next_at": next_at or now, "created_at": now,
        "lease_until": lease_until, "lease_token": token})


def _seed_backfill(owner, state, lease_until=None, token="deadbf"):
    now = facts._iso(facts._now())
    _raw_es("PUT", f"/{BACKFILLS}/_doc/bf?refresh=true", {
        "owner_id": owner, "visibility": "private", "state": state, "queued": 0,
        "created_at": now, "error": "", "lease_until": lease_until,
        "lease_token": token})


def _episodic_facade(inner_es):
    """Raw ES wrapper: the FIRST episodic scan returns one doc; later scans
    (after the cursor advanced past it) return none; updates are no-ops."""
    served = {"n": 0}

    def facade(method, path, body=None):
        if "episodic" not in path:
            return inner_es(method, path, body)
        if "_search" in path and method == "POST":
            if (body or {}).get("search_after"):
                return {"hits": {"hits": []}}
            served["n"] += 1
            return {"hits": {"hits": [
                {"_id": "src1", "sort": ["2024-01-01", "g1"], "_source": {}}]}}
        if "_update" in path:
            return {}
        return inner_es(method, path, body)
    return facade


# ---------------------------------------------------------------- H2 (a)
def test_H2a_orphaned_backfill_plus_expired_job_completes(monkeypatch, isolated):
    """RED on 5f479ab (no fix B): backfill + inner job both running w/ expired
    leases; claim_backfill + _run_backfill complete the job within a bound."""
    owner = "fix2-owner"
    _seed_backfill(owner, "running",
                   lease_until=facts._iso(facts._now() - dt.timedelta(hours=1)))
    job_id = "jobA"
    _seed_job(job_id, owner, "running",
              lease_until=facts._iso(facts._now() - dt.timedelta(hours=2)))
    monkeypatch.setattr(facts, "enqueue", lambda o, s: job_id)
    monkeypatch.setattr(facts, "idx", lambda kind: f"{PREFIX}episodic")
    real_es = facts.es
    monkeypatch.setattr(facts, "es", _episodic_facade(real_es))

    def fake_process(source_id, owner_id, abort=None):
        doc = real_es("GET", f"/{QUEUE}/_doc/{job_id}")
        _raw_es("POST", f"/{QUEUE}/_update/{job_id}"
                        f"?if_seq_no={doc['_seq_no']}&if_primary_term={doc['_primary_term']}&refresh=true",
                {"doc": {"state": "completed", "finished_at": facts._iso(facts._now())}})

    monkeypatch.setattr(facts, "process", fake_process)
    monkeypatch.setattr(facts.time, "sleep", lambda s: None)
    deadline = time.monotonic() + 30
    done = False
    while time.monotonic() < deadline and not done:
        if facts.run_backfill_once():
            done = True
    assert done, "backfill not claimed within bound"
    job = _raw_es("GET", f"/{QUEUE}/_doc/{job_id}")["_source"]
    assert job["state"] == "completed"


# ---------------------------------------------------------------- H2 (b)
def test_H2b_stale_fresh_1h_job_lease_becomes_visible_failure(monkeypatch, isolated):
    """RED on 1935023 (H1): inner job running with a FRESH 1h lease whose owner
    is DEAD. Old code slept until lease expiry (silent hour-long stall); new HEAD
    treats a job that never advances as no-progress and FAILS the backfill
    visibly (bounded by NO_PROGRESS_LIMIT), never silently waits 1h."""
    owner = "fix2-owner"
    job_id = "jobB"
    _seed_backfill(owner, "running", token="tok")
    _seed_job(job_id, owner, "running",
              lease_until=facts._iso(facts._now() + dt.timedelta(hours=1)))
    monkeypatch.setattr(facts, "enqueue", lambda o, s: job_id)
    monkeypatch.setattr(facts, "idx", lambda kind: f"{PREFIX}episodic")
    real_es = facts.es
    monkeypatch.setattr(facts, "es", _episodic_facade(real_es))
    monkeypatch.setattr(facts, "NO_PROGRESS_LIMIT", 3)
    monkeypatch.setattr(facts.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="no progress"):
        facts._run_backfill("bf", {"owner_id": owner, "visibility": "private"}, "tok")


# ---------------------------------------------------------------- H2 (c)
def test_H2c_no_progress_marks_backfill_failed(monkeypatch, isolated):
    owner = "fix2-owner"
    _seed_backfill(owner, "running",
                   lease_until=facts._iso(facts._now() - dt.timedelta(hours=1)), token="tok")
    job_id = "jobC"
    _seed_job(job_id, owner, "running",
              lease_until=facts._iso(facts._now() - dt.timedelta(hours=2)))
    monkeypatch.setattr(facts, "NO_PROGRESS_LIMIT", 3)
    monkeypatch.setattr(facts.time, "sleep", lambda s: None)
    monkeypatch.setattr(facts, "enqueue", lambda o, s: job_id)
    monkeypatch.setattr(facts, "idx", lambda kind: f"{PREFIX}episodic")
    real_es = facts.es
    monkeypatch.setattr(facts, "es", _episodic_facade(real_es))
    monkeypatch.setattr(facts, "run_once", lambda *a, **kw: True)   # changes nothing
    with pytest.raises(RuntimeError, match="no progress"):
        facts._run_backfill("bf", {"owner_id": owner, "visibility": "private"}, "tok")


# ---------------------------------------------------------------- H2 (d)
def test_H2d_running_job_without_lease_not_stolen_counts_to_abort(monkeypatch, isolated):
    owner = "fix2-owner"
    job_id = "jobD"
    _seed_backfill(owner, "running", token="tok")
    _seed_job(job_id, owner, "running", lease_until=None)
    monkeypatch.setattr(facts, "NO_PROGRESS_LIMIT", 3)
    monkeypatch.setattr(facts.time, "sleep", lambda s: None)
    stolen = []
    monkeypatch.setattr(facts, "run_once", lambda o, j, **kw: stolen.append(j) or True)
    monkeypatch.setattr(facts, "enqueue", lambda o, s: job_id)
    monkeypatch.setattr(facts, "idx", lambda kind: f"{PREFIX}episodic")
    real_es = facts.es
    monkeypatch.setattr(facts, "es", _episodic_facade(real_es))
    with pytest.raises(RuntimeError, match="no progress"):
        facts._run_backfill("bf", {"owner_id": owner, "visibility": "private"}, "tok")
    assert not stolen, "unleased running job must never be claimed"
    job = _raw_es("GET", f"/{QUEUE}/_doc/{job_id}")["_source"]
    assert job["state"] == "running"


# ---------------------------------------------------------------- H2 (e)
def test_H2e_dead_backfill_holder_reclaimed_after_backfill_lease(isolated):
    owner = "fix2-owner"
    _seed_backfill(owner, "running",
                   lease_until=facts._iso(facts._now() - dt.timedelta(
                       seconds=facts.BACKFILL_LEASE_SECONDS + 5)))
    claimed = facts.claim_backfill()
    assert claimed is not None
    bf = _raw_es("GET", f"/{BACKFILLS}/_doc/bf")["_source"]
    delta = (facts._parse_iso(bf["lease_until"]) - facts._now()).total_seconds()
    assert 0 < delta <= facts.BACKFILL_LEASE_SECONDS + 5


# ---------------------------------------------------------------- M3
def test_M3_stale_token_write_rejected_and_heartbeat_aborts(monkeypatch, isolated):
    """_update_backfill rejects a stale holder's cursor write (lease_token CAS);
    heartbeat retries renewal 2 more times then sets abort."""
    owner = "fix2-owner"
    _seed_backfill(owner, "running",
                   lease_until=facts._iso(facts._now() + dt.timedelta(minutes=5)),
                   token="realholder")
    # stale holder (wrong token) must NOT be able to write the cursor
    with pytest.raises(RuntimeError, match="lease lost"):
        facts._update_backfill("bf", "stale", cursor="[]")

    import threading
    stop = threading.Event()
    abort = threading.Event()
    monkeypatch.setattr(facts, "_update_backfill",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(facts, "HEARTBEAT_SECONDS", 0.01)
    facts._heartbeat("bf", "tok", stop, abort)      # returns after 3 failures
    assert abort.is_set(), "abort must be set after 3 renewal failures"


# ---------------------------------------------------------------- H1 mechanism
def test_H1_heartbeat_renews_tracked_job_lease(isolated):
    owner = "fix2-owner"
    job_id = "jobE"
    _seed_job(job_id, owner, "running",
              lease_until=facts._iso(facts._now() + dt.timedelta(hours=1)),
              token="mytoken")
    assert facts._renew_job_lease(job_id, "mytoken") is True
    job = _raw_es("GET", f"/{QUEUE}/_doc/{job_id}")["_source"]
    delta = (facts._parse_iso(job["lease_until"]) - facts._now()).total_seconds()
    assert 0 < delta <= facts.BACKFILL_LEASE_SECONDS + 5, delta
    assert facts._renew_job_lease(job_id, "othertoken") is False
