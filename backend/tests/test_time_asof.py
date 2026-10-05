"""Cause 3 fix contract: time is resolved against the instant the query is ASKED
(`as_of`), not against the wall clock, and recall surfaces `occurred_at`.

Two failures this pins:
  * recency decay used `datetime.now()`, so when the benchmark replays 2023 sessions in
    2026 every memory is ~3 years old and the arm's ordering is meaningless;
  * `parse_window` resolved "last two weeks" against today, so a 2023 question produced
    a 2026 window and the temporal arm matched nothing.

Integration tests run against live ES in the backend image, like the rest of the
suite:  docker compose ... run --rm --no-deps -v <repo>:/srv -w /srv/backend \
         ames-backend python -m pytest tests/test_time_asof.py -v
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, temporal  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qasof"
QDATE = "2023-05-30"


def _wipe():
    for kind in ("episodic", "semantic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": OWNER}}})


# ------------------------------------------------------------------ pure units
def test_parse_as_of_accepts_date_and_datetime_and_z():
    d = memory._parse_as_of("2023-05-30")
    assert d.date() == datetime.date(2023, 5, 30) and d.tzinfo is not None
    assert d.hour == 23, "a bare date must cover the whole day (same-day sessions)"
    dt = memory._parse_as_of("2023-05-30T08:15:00Z")
    assert (dt.hour, dt.minute) == (8, 15)
    naive = memory._parse_as_of("2023-05-30T08:15:00")
    assert naive.tzinfo == datetime.timezone.utc


def test_parse_window_resolves_against_as_of_not_today():
    now = datetime.datetime(2023, 5, 30, tzinfo=datetime.timezone.utc)
    start, end = temporal.parse_window("what did I do in the last two weeks", now)
    assert start.startswith("2023-05-16"), start
    assert end.startswith("2023-05-30"), end
    s2, _ = temporal.parse_window("what did I do in the last two weeks")
    assert not s2.startswith("2023"), s2


def test_parse_window_number_words_and_digits_agree():
    now = datetime.datetime(2023, 5, 30, tzinfo=datetime.timezone.utc)
    assert temporal.parse_window("in the last two weeks", now) == \
        temporal.parse_window("in the last 2 weeks", now)
    c = temporal.parse_window("in the last six days", now)
    assert c[0].startswith("2023-05-24"), c


def test_parse_window_year_and_month():
    now = datetime.datetime(2024, 1, 5, tzinfo=datetime.timezone.utc)
    s, e = temporal.parse_window("what did we plan in 2023", now)
    assert s.startswith("2023-01-01") and e.startswith("2023-12-31")
    s, e = temporal.parse_window("the budget for 2023-07", now)
    assert s.startswith("2023-07-01") and e.startswith("2023-07-31"), (s, e)


def test_recency_decay_uses_the_supplied_now():
    now = datetime.datetime(2023, 5, 30, tzinfo=datetime.timezone.utc)
    near = memory.recency_decay("2023-05-29T10:00:00", now)
    far = memory.recency_decay("2022-05-29T10:00:00", now)
    assert near > 0.5 > far, (near, far)
    wall = memory.recency_decay("2023-05-29T10:00:00")   # old behaviour: today
    assert wall < 0.01, wall


# ------------------------------------------------------------------ integration
def test_recall_as_of_recomputes_recency_against_the_question_date():
    """Recency is a tie-break arm (weight 0.15 x one RRF step), so the contract is the
    DECAY VALUE, not a guaranteed flip: the same session must look recent when the
    question date is near it and stale under the wall clock."""
    _wipe()
    memory.retain(OWNER, "episodic", "user: the spare key is in the blue tin by the door",
                  occurred_at="2022-04-01T09:00:00", doc_group="s-old")
    memory.retain(OWNER, "episodic", "user: the spare key was moved to the tin by the door",
                  occurred_at="2023-05-29T09:00:00", doc_group="s-new")
    r = memory.recall(OWNER, "where is the spare key kept?", kinds=["episodic"], size=8,
                      as_of=QDATE)
    by_group = {}
    for h in r["results"]:
        by_group.setdefault(h.get("doc_group"), h)
    assert {"s-old", "s-new"} <= set(by_group), by_group.keys()
    # `recency` is the BOOSTED tie-break value (max = 0.15/61 ~ 0.00246), not the
    # raw decay: near the question date it is at the ceiling, a year away it is ~0.
    assert by_group["s-new"]["recency"] > 0.0024, by_group["s-new"]["recency"]
    assert by_group["s-old"]["recency"] < 1e-4, by_group["s-old"]["recency"]
    assert all(h.get("occurred_at") for h in r["results"])


def test_recall_as_of_two_weeks_window_matches_the_replayed_period():
    """'last two weeks' asked on 2023-05-30 must hit a 2023-05-20 session and NOT the
    same-topic session from a year earlier (the wall-clock version matched neither)."""
    _wipe()
    memory.retain(OWNER, "episodic", "user: I bought a red road bike for the triathlon",
                  occurred_at="2022-05-20T09:00:00", doc_group="bike-2022")
    memory.retain(OWNER, "episodic", "user: I bought a blue road bike for the triathlon",
                  occurred_at="2023-05-20T09:00:00", doc_group="bike-2023")
    r = memory.recall(OWNER, "what did I buy in the last two weeks?",
                      kinds=["episodic"], size=8, as_of=QDATE)
    by_group = {}
    for h in r["results"]:
        by_group.setdefault(h.get("doc_group"), h)
    assert {"bike-2022", "bike-2023"} <= set(by_group), by_group.keys()
    window = r.get("temporal", {}).get("window") or []
    assert window and window[0].startswith("2023-05-16"), window
    assert by_group["bike-2023"].get("temporal_hit") is True, by_group["bike-2023"]
    assert not by_group["bike-2022"].get("temporal_hit"), by_group["bike-2022"]


def test_recall_without_as_of_keeps_wall_clock_behaviour():
    _wipe()
    memory.retain(OWNER, "episodic", "user: the backup disk is in the top drawer",
                  occurred_at="2020-01-01T09:00:00", doc_group="old")
    r = memory.recall(OWNER, "where is the backup disk?", kinds=["episodic"], size=8)
    assert r["results"], r
    assert r["results"][0]["recency"] < 1e-4, "no as_of -> decay from today (unchanged)"
