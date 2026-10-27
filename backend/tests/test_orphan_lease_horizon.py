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


def test_deterministic_run_once_failure_aborts_instead_of_looping(monkeypatch, harness):
    """Every run re-arms a 90s lease and raises: the changed lease_until used to
    look like progress and reset the counter, looping (and re-running the LLM)
    forever. Now fails visibly after MAX_CONSECUTIVE_RUN_FAILURES runs."""
    now, sleeps, fake = harness
    fake.job.update(state="pending", lease_until=None, next_at=facts._iso(now[0]))
    runs = []

    def run_once(owner, job_id, on_claim=None, on_finish=None, abort=None):
        runs.append(now[0])
        if len(runs) > 50:
            raise _StillLooping("still looping")
        fake.job.update(state="running", lease_until=facts._iso(
            now[0] + dt.timedelta(seconds=facts.BACKFILL_LEASE_SECONDS)))
        raise OSError("extraction boom")

    monkeypatch.setattr(facts, "run_once", run_once)
    with pytest.raises(RuntimeError, match="extraction boom"):
        facts._run_backfill("k", DOC, "tok")
    assert len(runs) == facts.MAX_CONSECUTIVE_RUN_FAILURES


class _StillLooping(BaseException):
    """Escapes `except Exception` but is a normal test failure, not a session abort."""


def test_failure_count_is_per_job_not_per_scope(monkeypatch, harness):
    """A job that reaches `completed` although run_once raised (finalize write
    committed, response lost) must not count toward the next job's cap."""
    now, sleeps, fake = harness
    jobs = iter(range(5))
    monkeypatch.setattr(facts, "enqueue", lambda owner, source: f"j{next(jobs)}")
    searches = []
    states = {}

    def es(method, path, body=None):
        if path.endswith("/_search"):
            searches.append(1)
            return {"hits": {"hits": [{"_id": f"s{len(searches)}", "sort": [len(searches), "g"]}]
                             if len(searches) <= 5 else []}}
        if path.startswith(f"/{facts.QUEUE}/_doc/"):
            job_id = path.rsplit("/", 1)[1]
            return {"_source": {"state": states.get(job_id, "pending"), "attempts": 0,
                                "next_at": facts._iso(now[0]), "lease_until": None}}
        return {}

    monkeypatch.setattr(facts, "es", es)

    def run_once(owner, job_id, on_claim=None, on_finish=None, abort=None):
        states[job_id] = "completed"     # write committed, response lost
        raise OSError("finalize read timeout (write committed)")

    monkeypatch.setattr(facts, "run_once", run_once)
    facts._run_backfill("k", DOC, "tok")          # must not raise
