"""Own orphaned job (run_once raised after claiming) at production constants.

After a run_once failure the job stays `running` with a lease up to
BACKFILL_LEASE_SECONDS out and the heartbeat stops renewing it. The scope must
wait out that lease (poll budget spans it) instead of failing as "stuck under a
foreign lease", and a final failure must carry the last run_once error.
Offline: in-memory ES fake and a fake clock, no live cluster needed.
"""
import datetime as dt

import pytest

from app import facts


class FakeEs:
    def __init__(self, clock):
        self.clock = clock
        self.job = {"state": "running", "attempts": 1, "next_at": None,
                    "lease_until": facts._iso(clock[0] + dt.timedelta(seconds=facts.BACKFILL_LEASE_SECONDS))}
        self.searches = 0

    def __call__(self, method, path, body=None):
        if path.endswith("/_search"):
            self.searches += 1
            hits = [{"_id": "src", "sort": [1, "g"]}] if self.searches == 1 else []
            return {"hits": {"hits": hits}}
        if "/_update/src" in path:
            return {}
        if path.startswith(f"/{facts.QUEUE}/_doc/"):
            return {"_source": dict(self.job)}
        if "/_update/" in path:
            return {}
        raise AssertionError((method, path))


@pytest.fixture
def harness(monkeypatch):
    now = [facts._now()]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += dt.timedelta(seconds=seconds)

    fake = FakeEs(now)
    monkeypatch.setattr(facts, "_now", lambda: now[0])
    monkeypatch.setattr(facts.time, "sleep", sleep)
    monkeypatch.setattr(facts, "es", fake)
    monkeypatch.setattr(facts, "idx", lambda kind: "episodic")
    monkeypatch.setattr(facts, "enqueue", lambda owner, source: "j")
    monkeypatch.setattr(facts, "_update_backfill", lambda *a, **kw: None)
    return now, sleeps, fake


def _fail_once_then_reclaim(monkeypatch, now, fake, ran):
    def run_once(owner, job_id, on_claim=None, on_finish=None, abort=None):
        ran.append(now[0])
        if len(ran) == 1:
            raise OSError("finalize transport error")
        fake.job.update(state="completed")
        return True

    monkeypatch.setattr(facts, "run_once", run_once)


DOC = {"owner_id": "o", "visibility": "private"}


def test_orphaned_own_job_waits_for_lease_then_is_reclaimed(monkeypatch, harness):
    now, sleeps, fake = harness
    ran = []
    # Job pending+due: first attempt raises, job is left running under a 90s lease.
    fake.job.update(state="pending", lease_until=None, next_at=facts._iso(now[0]))

    def run_once(owner, job_id, on_claim=None, on_finish=None, abort=None):
        ran.append(now[0])
        if len(ran) == 1:
            fake.job.update(state="running", lease_until=facts._iso(
                now[0] + dt.timedelta(seconds=facts.BACKFILL_LEASE_SECONDS)))
            raise OSError("finalize transport error")
        fake.job.update(state="completed")
        return True

    monkeypatch.setattr(facts, "run_once", run_once)
    facts._run_backfill("k", DOC, "tok")          # must not raise
    assert len(ran) == 2
    assert (ran[1] - ran[0]).total_seconds() >= facts.BACKFILL_LEASE_SECONDS


def test_persistent_foreign_lease_error_names_last_run_error(monkeypatch, harness):
    now, sleeps, fake = harness
    fake.job.update(state="pending", lease_until=None, next_at=facts._iso(now[0]))

    def run_once(owner, job_id, on_claim=None, on_finish=None, abort=None):
        # Every attempt fails and re-arms a 1h lease: scope must still fail
        # visibly, naming the underlying error.
        fake.job.update(state="running", lease_until=facts._iso(now[0] + dt.timedelta(hours=1)))
        raise OSError("finalize transport error")

    monkeypatch.setattr(facts, "run_once", run_once)
    with pytest.raises(RuntimeError, match="finalize transport error"):
        facts._run_backfill("k", DOC, "tok")


def test_retry_failed_job_resets_transient_retries(monkeypatch):
    sent = []

    def fake_es(method, path, body=None):
        if method == "GET":
            return {"_source": {"owner_id": "o", "state": "failed"}, "_seq_no": 1, "_primary_term": 1}
        sent.append(body)
        return {}

    monkeypatch.setattr(facts, "es", fake_es)
    facts.retry_failed_job("o", "j")
    assert sent[0]["doc"]["transient_retries"] == 0


def test_incomplete_read_is_transient():
    import http.client
    from app import http_retry
    assert http_retry.is_transient(http.client.IncompleteRead(b"par"))
