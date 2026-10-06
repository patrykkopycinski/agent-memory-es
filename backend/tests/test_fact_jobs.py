"""Unit tests for fact queue, validation, supersession and fail-open writes."""
from unittest.mock import patch
import pytest

from app import facts, memory


def test_extract_date_dedup_and_validation():
    with patch.object(facts.llm, "chat_json", return_value={"facts": ["Alice moved to Paris.", "Alice moved to Paris."]}):
        assert facts._extract("Alice moved to Paris", "2020-01-02") == ["[2020-01-02] Alice moved to Paris."]
    with patch.object(facts.llm, "chat_json", return_value={"facts": "wrong"}):
        with pytest.raises(ValueError):
            facts._extract("x", "2020-01-02")
    with pytest.raises(ValueError):
        facts._date("not-a-date")


def test_enqueue_idempotent_and_owner_status():
    calls = []
    def fake_es(method, path, body=None):
        calls.append((method, path, body))
        if "_create" in path and len([c for c in calls if "_create" in c[1]]) > 1:
            raise RuntimeError("ES PUT -> 409: conflict")
        if "_search" in path:
            return {"aggregations": {"states": {"buckets": [{"key": "failed", "doc_count": 1}]}}}
        return {}
    with patch.object(facts, "es", fake_es):
        assert facts.enqueue("alice", "src") == facts.enqueue("alice", "src")
        s = facts.status("alice")
    assert s["failed"] == 1 and not s["drained"]
    assert calls[-1][2]["query"] == {"term": {"owner_id": "alice"}}


def test_replacement_only_owner_scope_and_older_dates():
    body = {}
    def fake_es(method, path, data=None):
        body.update(data)
        return {"hits": {"hits": [{"_id": "old", "_source": {"text": "Alice lives in Rome", "valid_from": "2020-01-01"}},
                                  {"_id": "future", "_source": {"text": "Alice lives in Paris", "valid_from": "2030-01-01"}}]}}
    with patch.object(facts, "es", fake_es), patch.object(facts.llm, "chat_json", return_value={"replace_ids": ["old"]}):
        assert facts._replacement_ids("alice", "private", [0.1], "Alice lives in Paris", "2021-01-01") == ["old"]
    assert {"term": {"owner_id": "alice"}} in body["knn"]["filter"]
    assert {"term": {"visibility": "private"}} in body["knn"]["filter"]
    with patch.object(facts, "es", fake_es), patch.object(facts.llm, "chat_json", return_value={"replace_ids": ["future"]}):
        with pytest.raises(ValueError):
            facts._replacement_ids("alice", "private", [0.1], "x", "2021-01-01")


def test_fail_open_write_and_opt_out():
    def fake_es(method, path, body=None):
        if "_search" in path:
            return {"hits": {"hits": []}}
        return {"_id": "passage"}
    with patch.object(memory, "es", fake_es), patch.object(memory, "chunk_text", return_value=["Alice moved to Paris"]), \
         patch.object(memory.embeddings, "embed", side_effect=RuntimeError("offline")), \
         patch.object(facts, "enqueue", side_effect=RuntimeError("queue offline")):
        out = memory.retain("alice", "episodic", "Alice moved to Paris")
        assert out["_id"] == "passage" and out["fact_enqueue_errors"]
        out = memory.retain("alice", "episodic", "Bob moved to Rome", extract_facts=False)
        assert "fact_jobs" not in out


def test_drain_endpoint_uses_authenticated_owner(monkeypatch):
    from app import main
    monkeypatch.setattr(main.facts, "status", lambda owner: {"owner_id": owner, "drained": True})
    assert main.fact_drain({"owner_id": "alice"}) == {"owner_id": "alice", "drained": True}


def test_retry_terminal_failure(monkeypatch):
    monkeypatch.setattr(facts, "claim", lambda: ("job", {"source_id": "src", "owner_id": "alice", "attempts": 3}, "token"))
    monkeypatch.setattr(facts, "process", lambda *a: (_ for _ in ()).throw(ValueError("bad response")))
    captured = {}
    monkeypatch.setattr(facts, "_finish", lambda *args, **kwargs: captured.update({"args": args, **kwargs}))
    assert facts.run_once()
    assert captured["args"][2] == "failed" and captured["attempts"] == 4
