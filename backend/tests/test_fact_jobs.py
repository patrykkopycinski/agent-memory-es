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
        if "_count" in path:
            return {"count": 2}
        if "_search" in path:
            return {"aggregations": {"states": {"buckets": [{"key": "failed", "doc_count": 1}]}}}
        return {}
    with patch.object(facts, "es", fake_es):
        assert facts.enqueue("alice", "src") == facts.enqueue("alice", "src")
        s = facts.status("alice")
    assert s["failed"] == 1 and not s["drained"]
    assert calls[-2][2]["query"] == {"term": {"owner_id": "alice"}}
    assert s["missing_jobs"] == 1


def test_replacement_only_owner_scope_and_older_dates():
    body = {}
    def fake_es(method, path, data=None):
        body.update(data)
        return {"hits": {"hits": [{"_id": "old", "_score": 0.91, "_source": {"text": "Alice lives in Rome", "valid_from": "2020-01-01"}},
                                  {"_id": "future", "_score": 0.90, "_source": {"text": "Alice lives in Paris", "valid_from": "2030-01-01"}}]}}
    with patch.object(facts, "es", fake_es), patch.object(facts.llm, "chat_json", return_value={"replace_ids": ["old"]}):
        assert facts._replacement_ids("alice", "private", [0.1], "Alice lives in Paris", "2021-01-01") == ["old"]
    assert {"term": {"owner_id": "alice"}} in body["knn"]["filter"]
    assert {"term": {"visibility": "private"}} in body["knn"]["filter"]
    with patch.object(facts, "es", fake_es), patch.object(facts.llm, "chat_json", return_value={"replace_ids": ["future"]}):
        # v4: a future-dated (or hallucinated) id is DROPPED with a logged warning,
        # not a hard failure — the document's other facts still supersede normally.
        assert facts._replacement_ids("alice", "private", [0.1], "x", "2021-01-01") == []


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
    monkeypatch.setattr(main.facts, "backfill_status", lambda owner, vis: None)
    assert main.fact_drain({"owner_id": "alice"}) == {"owner_id": "alice", "drained": True,
                                                      "facts_dropped_invalid": 0,
                                                      "facts_dropped_overflow": 0}


def test_retry_terminal_failure(monkeypatch):
    monkeypatch.setattr(facts, "claim", lambda: ("job", {"source_id": "src", "owner_id": "alice", "attempts": 3}, "token"))
    monkeypatch.setattr(facts, "process", lambda *a: (_ for _ in ()).throw(ValueError("bad response")))
    captured = {}
    monkeypatch.setattr(facts, "_finish", lambda *args, **kwargs: captured.update({"args": args, **kwargs}))
    assert facts.run_once()
    assert captured["args"][2] == "failed" and captured["attempts"] == 4


def test_document_cache_and_window_extraction():
    seen = {}
    calls = []
    def fake_es(method, path, body=None):
        if method == "GET":
            if path not in seen:
                raise RuntimeError("ES GET -> 404: not found")
            return {"_source": seen[path]}
        key = path.split("/_create/")[1]
        seen[f"/{facts.CACHE}/_doc/{key}"] = body
        return {}
    def extract(text, date):
        calls.append(text)
        return [f"[{date}] Alice lives in Paris."]
    with patch.object(facts, "es", fake_es), patch.object(facts, "_extract", extract):
        passages = [("one", "Alice lives"), ("two", " in Paris")]
        assert len(facts._document_facts(passages, "2020-01-01T12:03:00Z")) == 1
        assert len(calls) == 1
        facts._document_facts(passages, "2020-01-01T12:03:00Z")
        assert len(calls) == 1
        with patch.object(facts, "PROMPT_VERSION", "new"):
            facts._document_facts(passages, "2020-01-01T12:03:00Z")
        assert len(calls) == 2


def test_similarity_gate_skips_judge():
    with patch.object(facts, "es", return_value={"hits": {"hits": [{"_score": 0.84,
            "_id": "old", "_source": {"text": "Alice lives in Rome", "valid_from": "2020-01-01"}}]}}), \
         patch.object(facts.llm, "chat_json", side_effect=AssertionError("judge invoked")):
        assert facts._replacement_ids("alice", "private", [0.1], "Alice moved", "2021-01-01") == []


def test_recall_validity_filters_both_retrievers():
    bodies = []
    def search(method, path, body=None):
        bodies.append(body)
        return {"hits": {"hits": []}}
    with patch.object(memory, "es", search), patch.object(memory.embeddings, "embed", return_value=[[0.1]]):
        memory.recall("alice", "where", kinds=["semantic"], as_of="2020-01-01T12:00:00Z")
    body = next(b for b in bodies if b and "retriever" in b)
    retrievers = body["retriever"]["rrf"]["retrievers"]
    lexical = retrievers[0]["standard"]["query"]["bool"]["filter"]
    knn = retrievers[1]["knn"]["filter"]
    assert lexical == knn
    assert any("valid_from" in str(c) and "2020-01-01T12:00:00+00:00" in str(c) for c in lexical)
    assert any("valid_to" in str(c) and "gt" in str(c) for c in lexical)


def test_status_unqueued_blocks_drain():
    def fake_es(method, path, body=None):
        if path.endswith("_count"):
            assert {"term": {"fact_requested": True}} in body["query"]["bool"]["filter"]
            return {"count": 3}
        return {"aggregations": {"states": {"buckets": [{"key": "completed", "doc_count": 2}]}}}
    with patch.object(facts, "es", fake_es):
        assert facts.status("alice")["missing_jobs"] == 1
        assert not facts.status("alice")["drained"]


def test_timestamp_precision():
    assert facts._date("2022-02-01T18:15:00-05:00") == "2022-02-01T23:15:00Z"
    assert facts._date("2022-02-01T08:00:00Z") != facts._date("2022-02-01T18:00:00Z")


def test_process_retry_uses_cached_facts_and_best_passage():
    docs = {}
    extraction = []
    source = {"owner_id": "alice", "visibility": "private", "active": True,
              "occurred_at": "2021-04-02T16:30:00Z", "doc_group": "group", "passages_total": 2}
    def fake_es(method, path, body=None):
        if path.endswith("_search"):
            return {"hits": {"hits": [
                {"_id": "p1", "_source": {"text": "A different sentence."}},
                {"_id": "p2", "_source": {"text": "Alice lives in Paris."}}]}}
        if path.endswith("/_doc/p1"):
            return {"_source": source}
        if method == "GET":
            if path not in docs:
                raise RuntimeError("ES GET -> 404: missing")
            return {"_source": docs[path]}
        if "/_create/" in path:
            key = path.split("/_create/")[1].split("?")[0]
            index = path.split("/_create/")[0]
            docs[f"{index}/_doc/{key}"] = body
            return {}
        raise AssertionError(path)
    def extract(text, date):
        extraction.append(text)
        return [f"[{date}] Alice lives in Paris."]
    with patch.object(facts, "es", fake_es), patch.object(facts, "_extract", extract), \
         patch.object(facts.embeddings, "embed", return_value=[[0.1]]), \
         patch.object(facts, "_replacement_ids", return_value=[]):
        facts.process("p1", "alice")
        facts.process("p1", "alice")
    assert len(extraction) == 1
    semantic = [v for k, v in docs.items() if k.startswith(f"/{facts.idx('semantic')}/_doc/")]
    assert len(semantic) == 1
    assert semantic[0]["source_id"] == "p2"
    assert semantic[0]["doc_group"] == "group"
    assert semantic[0]["valid_from"] == "2021-04-02T16:30:00Z"
