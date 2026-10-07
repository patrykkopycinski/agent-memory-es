"""Round 3 lease integration checks. All ES writes use a per-run UUID index prefix."""
import datetime as dt
import threading
import time
import uuid

import pytest

from app import facts
from test_fact_fix2_reales import _http, _raw_es, _episodic_facade, es_reachable

pytestmark = pytest.mark.skipif(not es_reachable(), reason="live ES unreachable")
PREFIX = f"fix2r3_{uuid.uuid4().hex}_"
QUEUE = PREFIX + "am_fact_jobs"
BACKFILLS = PREFIX + "am_fact_backfills"


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(facts, "QUEUE", QUEUE)
    monkeypatch.setattr(facts, "BACKFILLS", BACKFILLS)
    monkeypatch.setattr(facts, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(facts, "BACKFILL_LEASE_SECONDS", 0.3)
    monkeypatch.setattr(facts, "idx", lambda kind: PREFIX + kind)
    monkeypatch.setattr(facts, "es", _raw_es)
    _raw_es("PUT", "/" + QUEUE, facts.JOB_MAPPING)
    _raw_es("PUT", "/" + BACKFILLS, facts.BACKFILL_MAPPING)
    yield
    for index in (QUEUE, BACKFILLS):
        _raw_es("DELETE", "/" + index)


def seed(state="pending", job_state="pending", job_lease=None, job_token="foreign", bf_lease=None):
    now = facts._iso(facts._now())
    _raw_es("PUT", f"/{BACKFILLS}/_doc/bf?refresh=true", {
        "owner_id": "owner", "visibility": "private", "state": state, "queued": 0,
        "created_at": now, "error": "", "lease_token": "oldbf",
        "lease_until": bf_lease})
    _raw_es("PUT", f"/{QUEUE}/_doc/job?refresh=true", {
        "owner_id": "owner", "source_id": "src", "state": job_state,
        "attempts": 0, "created_at": now, "next_at": now,
        "lease_token": job_token, "lease_until": job_lease})


def doc(index, key):
    return _raw_es("GET", f"/{index}/_doc/{key}")["_source"]


def runner(monkeypatch, process):
    monkeypatch.setattr(facts, "enqueue", lambda *_: "job")
    monkeypatch.setattr(facts, "process", process)
    monkeypatch.setattr(facts, "es", _episodic_facade(_raw_es))


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_T1_minted_token_renews_during_process(monkeypatch, setup):
    seed()
    entered, release = threading.Event(), threading.Event()
    runner(monkeypatch, lambda *a, **kw: (entered.set(), release.wait(3)))
    thread = threading.Thread(target=facts.run_backfill_once)
    thread.start()
    try:
        assert entered.wait(2)
        first = doc(QUEUE, "job")
        assert first["lease_token"] != "foreign"
        assert first["state"] == "running"
        horizon = (facts._parse_iso(first["lease_until"]) - facts._now()).total_seconds()
        assert 0 < horizon <= 0.4
        assert wait_for(lambda: doc(QUEUE, "job")["lease_until"] > first["lease_until"], 2)
        later = doc(QUEUE, "job")
        assert later["lease_token"] == first["lease_token"]
        assert (facts._parse_iso(later["lease_until"]) - facts._now()).total_seconds() <= 0.4
        # Stay blocked for at least five heartbeat periods.
        time.sleep(0.26)
        assert doc(QUEUE, "job")["lease_until"] > later["lease_until"]
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert doc(BACKFILLS, "bf")["state"] == "completed"


def test_T2_dead_holder_reclaims_job_and_scope(monkeypatch, setup):
    seed()
    entered, release, stop_seen = threading.Event(), threading.Event(), threading.Event()
    calls = []
    original = facts._heartbeat

    def heartbeat(key, token, stop, *args):
        stop_seen.stop = stop
        stop_seen.set()
        return original(key, token, stop, *args)

    monkeypatch.setattr(facts, "_heartbeat", heartbeat)

    def process(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            release.wait(4)

    runner(monkeypatch, process)
    first = threading.Thread(target=facts.run_backfill_once)
    first.start()
    try:
        assert entered.wait(2) and stop_seen.wait(2)
        stop_seen.stop.set()  # holder dies while process is still blocked
        assert wait_for(lambda: facts._parse_iso(doc(QUEUE, "job")["lease_until"]) <= facts._now(), 2)
        assert facts.run_backfill_once()
        assert calls == [1, 1]
        assert doc(BACKFILLS, "bf")["state"] == "completed"
        assert doc(QUEUE, "job")["state"] == "completed"
    finally:
        release.set()
        first.join(3)
    assert not first.is_alive()


def test_T3_reboot_old_hour_job_fails_bounded_and_becomes_claimable(monkeypatch, setup):
    seed(state="running", job_state="running",
         job_lease=facts._iso(facts._now() + dt.timedelta(hours=1)),
         bf_lease=facts._iso(facts._now() - dt.timedelta(seconds=1)))
    calls = []
    runner(monkeypatch, lambda *a, **kw: calls.append(1))
    monkeypatch.setattr(facts, "NO_PROGRESS_LIMIT", 2)
    monkeypatch.setattr(facts.time, "sleep", lambda _: None)
    start = time.monotonic()
    assert facts.run_backfill_once()
    assert time.monotonic() - start < 5
    assert doc(BACKFILLS, "bf")["state"] == "failed"
    assert calls == []
    assert (facts._parse_iso(doc(QUEUE, "job")["lease_until"]) - facts._now()).total_seconds() > 3500
    # Documented fallback: original long lease remains until expiry, then retry.
    job = _raw_es("GET", f"/{QUEUE}/_doc/job")
    _raw_es("POST", f"/{QUEUE}/_update/job?if_seq_no={job['_seq_no']}"
            f"&if_primary_term={job['_primary_term']}&refresh=true",
            {"doc": {"lease_until": facts._iso(facts._now() - dt.timedelta(seconds=1))}})
    bf = _raw_es("GET", f"/{BACKFILLS}/_doc/bf")
    _raw_es("POST", f"/{BACKFILLS}/_update/bf?if_seq_no={bf['_seq_no']}"
            f"&if_primary_term={bf['_primary_term']}&refresh=true",
            {"doc": {"state": "pending"}})
    assert facts.run_backfill_once()
    assert calls == [1]
    assert doc(BACKFILLS, "bf")["state"] == "completed"


def test_T4_foreign_lease_never_renewed(monkeypatch, setup):
    future = facts._iso(facts._now() + dt.timedelta(seconds=5))
    seed(job_state="running", job_lease=future)
    runner(monkeypatch, lambda *a, **kw: pytest.fail("foreign process stolen"))
    monkeypatch.setattr(facts, "NO_PROGRESS_LIMIT", 3)
    monkeypatch.setattr(facts.time, "sleep", lambda _: None)
    assert facts.run_backfill_once()
    assert doc(QUEUE, "job")["lease_until"] == future
    assert doc(QUEUE, "job")["lease_token"] == "foreign"


def test_T5_expired_race_does_not_revive_foreign_token(monkeypatch, setup):
    past = facts._iso(facts._now() - dt.timedelta(seconds=1))
    seed(job_state="running", job_lease=past)
    assert facts._renew_job_lease("job", "other") is False
    assert doc(QUEUE, "job")["lease_until"] == past
    calls = []
    monkeypatch.setattr(facts, "process", lambda *a, **kw: calls.append(1))
    assert facts.run_once("owner", "job")
    assert calls == [1]
    assert doc(QUEUE, "job")["state"] == "completed"


def test_T6_abort_before_scope_expiry(monkeypatch, setup):
    seed()
    entered = threading.Event()
    renews = []
    real_update = facts._update_backfill

    def update(key, token, **fields):
        if "lease_until" in fields:
            renews.append(1)
            raise RuntimeError("transient scope failure")
        return real_update(key, token, **fields)

    monkeypatch.setattr(facts, "_update_backfill", update)

    def process(*a, **kw):
        entered.set()
        abort = kw["abort"]
        assert wait_for(abort.is_set, 2)

    runner(monkeypatch, process)
    assert facts.run_backfill_once()
    bf = doc(BACKFILLS, "bf")
    assert entered.is_set() and len(renews) == 3
    assert bf["state"] == "failed" and not bf.get("cursor")
    assert facts._parse_iso(bf["lease_until"]) > facts._now()


def test_job_renew_failure_does_not_abort_scope(monkeypatch, setup):
    seed()
    entered, release = threading.Event(), threading.Event()
    runner(monkeypatch, lambda *a, **kw: (entered.set(), release.wait(2)))
    monkeypatch.setattr(facts, "_renew_job_lease", lambda *a: (_ for _ in ()).throw(RuntimeError("404")))
    thread = threading.Thread(target=facts.run_backfill_once)
    thread.start()
    try:
        assert entered.wait(2)
        assert wait_for(lambda: facts._parse_iso(doc(BACKFILLS, "bf")["lease_until"]) >
                        facts._now() + dt.timedelta(seconds=0.25), 1)
    finally:
        release.set()
        thread.join(3)
    assert doc(BACKFILLS, "bf")["state"] == "completed"


def test_renew_requires_running_state(setup):
    seed(job_state="completed", job_token="mine")
    assert facts._renew_job_lease("job", "mine") is False
    assert doc(QUEUE, "job")["lease_until"] is None
