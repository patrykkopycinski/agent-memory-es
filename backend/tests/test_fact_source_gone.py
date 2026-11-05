"""A deleted, moved or missing source is a terminal skip: the job completes with a skip note and
is NOT charged an attempt (it used to fail after MAX_ATTEMPTS). Genuine extraction errors still
consume attempts (negative control)."""
import pytest

from app import facts


def _job(attempts=1):
    return ("job", {"source_id": "src", "owner_id": "alice", "attempts": attempts}, "token")


@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(facts, "_finish", lambda *a, **kw: seen.update({"args": a, **kw}))
    return seen


def _fake_es(source):
    def es(method, path, body=None):
        assert method == "GET"
        if isinstance(source, Exception):
            raise source
        return {"_source": source}
    return es


@pytest.mark.parametrize("source", [
    {"owner_id": "alice", "active": False, "occurred_at": "2024-01-01T00:00:00Z"},
    {"owner_id": "bob", "active": True, "occurred_at": "2024-01-01T00:00:00Z"},
    RuntimeError('ES GET /am_episodic/_doc/src -> 404: {"_index":"am_episodic","_id":"src","found":false}'),
], ids=["soft-deleted", "owner-mismatch", "missing-404"])
def test_gone_source_completes_as_skipped_without_charging_an_attempt(monkeypatch, captured, source):
    monkeypatch.setattr(facts, "claim", lambda: _job(attempts=2))
    monkeypatch.setattr(facts, "es", _fake_es(source))
    assert facts.run_once()
    assert captured["args"][2] == "completed"
    assert captured["error"] == facts.SKIP_MESSAGE
    assert "attempts" not in captured and "next_at" not in captured


def test_skip_happens_even_on_the_last_attempt(monkeypatch, captured):
    monkeypatch.setattr(facts, "claim", lambda: _job(attempts=facts.MAX_ATTEMPTS - 1))
    monkeypatch.setattr(facts, "es", _fake_es({"owner_id": "alice", "active": False}))
    facts.run_once()
    assert captured["args"][2] == "completed"


def test_non_404_source_read_error_still_consumes_an_attempt(monkeypatch, captured):
    monkeypatch.setattr(facts, "claim", lambda: _job(attempts=1))
    monkeypatch.setattr(facts, "es", _fake_es(RuntimeError("ES GET -> 400: bad request")))
    facts.run_once()
    assert captured["args"][2] == "pending" and captured["attempts"] == 2


def test_genuine_extraction_error_still_consumes_an_attempt(monkeypatch, captured):
    monkeypatch.setattr(facts, "claim", lambda: _job(attempts=1))
    monkeypatch.setattr(facts, "process", lambda *a: (_ for _ in ()).throw(ValueError("bad response")))
    facts.run_once()
    assert captured["args"][2] == "pending" and captured["attempts"] == 2
    assert "skipped" not in captured["error"]


def test_index_level_404_is_a_charged_failure_not_a_skip(monkeypatch, captured):
    """index_not_found_exception means the store is broken: every job would silently 'complete'."""
    monkeypatch.setattr(facts, "claim", lambda: _job(attempts=1))
    monkeypatch.setattr(facts, "es", _fake_es(RuntimeError(
        'ES GET /am_episodic/_doc/src -> 404: {"error":{"root_cause":[{"type":"index_not_found_exception",'
        '"reason":"no such index [am_episodic]"}],"type":"index_not_found_exception"},"status":404}')))
    facts.run_once()
    assert captured["args"][2] == "pending" and captured["attempts"] == 2
    assert "skipped" not in captured["error"]
