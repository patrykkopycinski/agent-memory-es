"""Round 6 fixes (review6 findings 1 and 3): bounded scheduled wait and
finalize errors that must not fail the backfill scope.
Requires a DEDICATED live ES: set FIX2_ES (it has no default).
"""
import datetime as dt
import urllib.error

import pytest

from app import facts
from test_fact_fix2_reales import _raw_es, es_reachable
from test_fact_fix2_r3_reales import BACKFILLS, QUEUE, doc, runner, seed, setup

pytestmark = pytest.mark.skipif(not es_reachable(), reason="FIX2_ES not set or unreachable")
from test_fact_fix2_r5_reales import fake_clock


def test_far_future_next_at_is_bounded(monkeypatch, setup, caplog):
    """F1: next_at = now + 1 day (clock skew / bad value). The scheduled wait
    must warn once, count no_progress, and fall into the existing abort path
    instead of sleeping forever while generic workers skip the owner."""
    seed()
    now, sleeps = fake_clock(monkeypatch)
    _raw_es("POST", f"/{QUEUE}/_update/job?refresh=true", {
        "doc": {"next_at": facts._iso(now[0] + dt.timedelta(days=1))}})
    calls = []
    runner(monkeypatch, lambda *a, **kw: calls.append(1))
    with caplog.at_level("WARNING"):
        assert facts.run_backfill_once()
    assert calls == []                      # the far-future job never ran
    assert doc(BACKFILLS, "bf")["state"] == "failed"
    assert doc(QUEUE, "job")["state"] == "pending"
    horizon_warnings = [r for r in caplog.records
                        if "next_at" in r.getMessage()
                        and "beyond" in r.getMessage()]
    assert horizon_warnings                 # warned at least once
    assert all(0.5 <= s <= 15 for s in sleeps)


def test_finalize_URLError_scope_still_completes(monkeypatch, setup, caplog):
    """F3: run_once raises when job finalization hits a transport error (all
    5 finalize retries fail). _run_backfill must log it, count no_progress,
    and keep going: the job is reclaimed via lease expiry and the scope ends
    'completed', never 'failed'."""
    seed()
    calls = []
    runner(monkeypatch, lambda *a, **kw: calls.append(1))
    transport = facts.es
    failures = []

    def flaky_es(method, path, body=None):
        if (method == "POST" and f"/{QUEUE}/_update/job?" in path
                and body and body.get("doc", {}).get("state") == "completed"):
            if not failures:
                failures.append(1)
                raise urllib.error.URLError("es down")
        return transport(method, path, body)

    monkeypatch.setattr(facts, "es", flaky_es)
    with caplog.at_level("WARNING"):
        assert facts.run_backfill_once()
    monkeypatch.setattr(facts, "es", transport)
    assert failures == [1]                  # first round's finalize hit the transport error
    assert calls == [1, 1]                  # job reprocessed after lease expiry
    assert doc(BACKFILLS, "bf")["state"] == "completed"
    assert doc(BACKFILLS, "bf")["error"] == ""
    assert doc(QUEUE, "job")["state"] == "completed"
    assert doc(QUEUE, "job")["attempts"] == 0
    assert any("run_once failed after processing" in r.getMessage()
               or "finalize" in r.getMessage() for r in caplog.records)
