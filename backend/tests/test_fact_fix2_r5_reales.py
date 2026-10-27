"""Round 5 backoff and retry-budget tests on dedicated FIX2_ES."""
import datetime as dt
import threading
import urllib.error

import pytest

from app import facts
from test_fact_fix2_reales import _raw_es, es_reachable
from test_fact_fix2_r3_reales import BACKFILLS, QUEUE, doc, runner, seed, setup

pytestmark = pytest.mark.skipif(not es_reachable(), reason="FIX2_ES not set or unreachable")


def fake_clock(monkeypatch):
    now = [facts._now()]
    sleeps = []
    monkeypatch.setattr(facts, "_now", lambda: now[0])

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += dt.timedelta(seconds=seconds)

    monkeypatch.setattr(facts.time, "sleep", sleep)
    return now, sleeps


def test_pending_120s_backoff_survives(monkeypatch, setup, caplog):
    seed()
    now, sleeps = fake_clock(monkeypatch)
    _raw_es("POST", f"/{QUEUE}/_update/job?refresh=true", {
        "doc": {"next_at": facts._iso(now[0] + dt.timedelta(seconds=120))}})
    calls = []
    runner(monkeypatch, lambda *a, **kw: calls.append(facts._now()))
    assert facts.run_backfill_once()
    assert doc(BACKFILLS, "bf")["state"] == "completed"
    assert doc(QUEUE, "job")["state"] == "completed"
    assert len(calls) == 1
    assert sum(sleeps) >= 120
    assert all(0.5 <= seconds <= 15 for seconds in sleeps)
    assert "pending without" not in caplog.text


@pytest.mark.parametrize("error", [urllib.error.URLError("offline"), ValueError("boom")],
                         ids=["transient", "value-error"])
def test_abort_with_exception_preserves_retry_budgets(monkeypatch, setup, error):
    seed()
    _raw_es("POST", f"/{QUEUE}/_update/job?refresh=true", {
        "doc": {"transient_retries": 3}})
    abort = threading.Event()

    def process(*a, **kw):
        abort.set()
        raise error

    monkeypatch.setattr(facts, "process", process)
    assert facts.run_once("owner", "job", on_claim=lambda *_: None, abort=abort)
    job = doc(QUEUE, "job")
    assert job["state"] == "pending"
    assert job["attempts"] == 0
    assert job["transient_retries"] == 3


def test_completion_conflict_exhaustion_preserves_attempts(monkeypatch, setup):
    seed()
    monkeypatch.setattr(facts, "process", lambda *a, **kw: None)
    transport = facts.es
    conflicts = []

    def conflicting_es(method, path, body=None):
        if (method == "POST" and f"/{QUEUE}/_update/job?" in path
                and body and body.get("doc", {}).get("state") == "completed"):
            conflicts.append(facts._renew_job_lease("job", doc(QUEUE, "job")["lease_token"]))
        return transport(method, path, body)

    monkeypatch.setattr(facts, "es", conflicting_es)
    finished = []
    with pytest.raises(RuntimeError, match="-> 409:"):
        facts.run_once("owner", "job", on_claim=lambda *_: None,
                       on_finish=lambda: finished.append(True))
    assert conflicts == [True] * 5
    assert finished == [True]
    job = doc(QUEUE, "job")
    assert job["state"] == "running"
    assert job["attempts"] == 0
    monkeypatch.setattr(facts, "es", transport)
    monkeypatch.setattr(facts, "_now", lambda: facts._parse_iso(job["lease_until"]) + dt.timedelta(seconds=1))
    assert facts.run_once("owner", "job", on_claim=lambda *_: None)
    assert doc(QUEUE, "job")["state"] == "completed"
    assert doc(QUEUE, "job")["attempts"] == 0
