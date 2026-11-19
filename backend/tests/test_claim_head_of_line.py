"""Generic claim() must not stall behind a backfill-owned owner's oldest jobs.

Runs against the throwaway ES from the suite env, in private per-test queue and
backfill indices so other tests' jobs cannot be claimed here."""
import datetime as dt
import uuid

import pytest

from app import facts
from app.store import es


@pytest.fixture()
def queues(monkeypatch):
    tag = uuid.uuid4().hex[:8]
    queue = f"{facts.PREFIX}hol_{tag}_am_fact_jobs"
    backfills = f"{facts.PREFIX}hol_{tag}_am_fact_backfills"
    monkeypatch.setattr(facts, "QUEUE", queue)
    monkeypatch.setattr(facts, "BACKFILLS", backfills)
    facts.ensure_queue()
    yield queue, backfills
    for index in (queue, backfills):
        es("DELETE", f"/{index}")


def _job(queue, owner, n, age_minutes):
    stamp = facts._iso(facts._now() - dt.timedelta(minutes=age_minutes))
    es("PUT", f"/{queue}/_doc/{owner}-{n}", {
        "owner_id": owner, "source_id": f"src-{n}", "state": "pending",
        "attempts": 0, "next_at": stamp, "created_at": stamp})


def test_generic_claim_skips_backfill_owner_beyond_first_page(queues):
    queue, backfills = queues
    # 25 > the claim page size (20): the oldest page is all owner A.
    for n in range(25):
        _job(queue, "owner-a", n, age_minutes=120)
    _job(queue, "owner-b", 0, age_minutes=1)
    es("PUT", f"/{backfills}/_doc/bf-a?refresh=true", {
        "owner_id": "owner-a", "visibility": "private", "state": "running",
        "queued": 0, "created_at": facts._iso(facts._now())})
    es("POST", f"/{queue}/_refresh")

    claimed = facts.claim()

    assert claimed is not None, "generic worker starved behind backfill-owned jobs"
    job_id, source, _lease = claimed
    assert job_id == "owner-b-0" and source["owner_id"] == "owner-b"


def test_job_id_claim_reaches_backfill_owner_and_owner_claim_stays_gated(queues):
    queue, backfills = queues
    _job(queue, "owner-a", 0, age_minutes=120)
    _job(queue, "owner-a", 1, age_minutes=119)
    es("PUT", f"/{backfills}/_doc/bf-a?refresh=true", {
        "owner_id": "owner-a", "visibility": "private", "state": "pending",
        "queued": 0, "created_at": facts._iso(facts._now())})
    es("POST", f"/{queue}/_refresh")

    assert facts.claim() is None
    assert facts.claim(job_id="owner-a-1")[0] == "owner-a-1"
    # unchanged: only the exact-id path (the backfill runner) may take these jobs
    assert facts.claim(owner_id="owner-a") is None


def test_backfill_owner_cap_overflow_logs_warning(monkeypatch, caplog):
    reply = {"aggregations": {"owners": {"sum_other_doc_count": 7,
                                         "buckets": [{"key": "owner-a"}]}}}
    monkeypatch.setattr(facts, "es", lambda *args, **kwargs: reply)
    with caplog.at_level("WARNING", logger=facts.log.name):
        assert facts._backfill_owners() == ["owner-a"]
    assert "truncated" in caplog.text and "7" in caplog.text
