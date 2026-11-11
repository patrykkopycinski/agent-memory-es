"""Anti-ephemeral retain guard (Cortex-style durable-only policy): a `semantic` memory that
reads as a single-run report — an event tally, a precise timestamp, a past-tense duration —
is refused 422 on both retain surfaces; `episodic` accepts the same text unchanged, and
durable semantic text (including schedules with clock times and rate/size limits) is
untouched. Pattern rules live in tests first: a new rejection or an allowable case is added
here before the pattern changes.

Runs against the throwaway test ES only (tests/_safety.py). An autouse fixture clears this
owner's slice before and after every test so no writes leak into the shared test index.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest
from fastapi.testclient import TestClient

from app import auth, ephemeral_guard, memory
from app.main import app
from app.store import PREFIX, es, idx

OWNER = "p4ephemeral"
client = TestClient(app)
H = {"X-API-Key": auth.create_key(OWNER, person=OWNER)}

INCIDENT_REPORTS = [
    # event tallies: the number is a property of one run
    "the ingest worker dropped 5944 errors during the night",
    "23 restarts were observed on the collector",
    "about 97 percent of requests failed",           # counted noun 'requests'
    "we saw 12 timeouts across the fleet",
    "roughly 40 pods could not schedule",
    # the rate limit frames the first number; the later tally is one run's and still rejects
    "the gateway handles 5000 requests per second yet dropped 5944 requests tonight",
    # precise timestamps: full dates, month-day dates, clock moments
    "the check ran on 2026-11-10 and failed",
    "incident started 2026-11-10T10:42:15Z",
    "the run finished at 10:42",                     # completed-event verb pins the clock time
    "the node went down at 22:10",
    "the collector crashed at 07:45 on Nov 9",       # month-day date, not ISO
    "the node went down on September 21",
    # past-tense durations: how long one finished thing lasted
    "the pipeline took 18 minutes to drain",
    "the pipeline took twenty minutes to drain",     # spelled-out tens, not just 1-12
    "es was down for 2 hours",
    "it lasted six minutes",
]

DURABLE_FACTS = [
    "the ingest worker drops errors repeatedly when the queue backs up",
    "restart loops recur on the collector after config reloads",
    "the auth migration finished; token estimator docs live in the wiki",
    "the ferry timetable changed to a 6 o'clock departure",      # present-tense schedule
    "since 2026 the team standard is trunk-based development",   # bare year is allowed
    "two approvals are required to override a decision",         # configuration, not a tally
    "the nightly job takes 2 hours every night",                 # habitual present tense
    "kibana service accounts land in 9.6",
    # schedules may carry clock times (bare HH:MM or HH:MM am/pm) on a present-tense verb
    "the nightly batch job starts at 02:00",
    "the ferry departs at 6:30 pm daily",
    # tense-ambiguous verbs carry schedules too; only completed-event verbs pin moments
    "the maintenance window was at 02:00 every night",
    "the sync started at 02:00 nightly",
    # rate and size limits are durable configuration, not one-run tallies
    "the api accepts at most 100 requests per second",
    "the ingest index is capped at 5000 documents",
    # possession/retention statements of present state stay writable
    "kafka retains 7 days of messages",
    "the index has 3 shards",
    "the archive holds 2000 records",
]

TWO_LIMITS = ("the gateway handles 5000 requests per second; "
              "the ingest index is capped at 5000 documents")


def _clean():
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true", {"query": {"term": {"owner_id": OWNER}}})
    try:
        es("POST", f"/{PREFIX}am_mutations/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": OWNER}}})
    except RuntimeError:
        pass


@pytest.fixture(autouse=True)
def _clean_slice():
    _clean()
    yield
    _clean()


def test_pattern_rules_accept_every_durable_fact():
    for text in DURABLE_FACTS:
        ephemeral_guard.reject_ephemeral(text)  # no raise


def test_pattern_rules_reject_every_incident_report():
    for text in INCIDENT_REPORTS:
        with pytest.raises(ValueError):
            ephemeral_guard.reject_ephemeral(text)


def test_two_framed_limits_in_one_text_stay_writable():
    ephemeral_guard.reject_ephemeral(TWO_LIMITS)  # no raise


def test_every_counted_noun_rejects_when_unframed():
    # the noun list is closed but long; a noun that stops rejecting would silently
    # widen the durable path, so every noun is exercised mechanically
    import re as _re
    nouns = _re.sub(r"\?.*", "", "|".join(ephemeral_guard._COUNTED_NOUNS)).split("|")
    for noun in nouns:
        base = _re.sub(r"\?.*", "", noun)
        with pytest.raises(ValueError):
            ephemeral_guard.reject_ephemeral(f"we observed 30 {base} during the incident")


@pytest.mark.parametrize("text", INCIDENT_REPORTS)
def test_semantic_incident_report_is_422_on_both_retain_surfaces(text):
    direct = client.post("/memory/retain", headers=H, json={"kind": "semantic", "text": text})
    assert direct.status_code == 422, direct.text
    via_mcp = client.post("/mcp/tools/retain", headers=H, json={"kind": "semantic", "text": text})
    assert via_mcp.status_code == 422, via_mcp.text


@pytest.mark.parametrize("text", INCIDENT_REPORTS)
def test_episodic_accepts_the_same_text(text):
    r = client.post("/memory/retain", headers=H, json={"kind": "episodic", "text": text})
    assert r.status_code == 200, r.text


def test_durable_semantic_fact_still_writes_clean():
    r = client.post("/memory/retain", headers=H,
                    json={"kind": "semantic", "text": DURABLE_FACTS[0]})
    assert r.status_code == 200, r.text
    assert "too_long" not in r.json()


def test_message_names_the_pattern_not_the_match():
    with pytest.raises(ValueError) as exc:
        ephemeral_guard.reject_ephemeral("5944 errors happened")
    assert "5944" not in str(exc.value)
    assert "count" in str(exc.value)
