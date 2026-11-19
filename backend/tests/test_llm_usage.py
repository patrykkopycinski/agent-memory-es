"""llm usage recording: every chat call's token usage is accumulated per
(model, purpose) and the last call's delta is exposed for extraction records.
OpenAI and es_inference paths both covered; no Elasticsearch needed."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import llm  # noqa: E402

from test_llm_es_inference import Fake, sse  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_usage():
    llm.USAGE.clear()
    llm.LAST_CALL.clear()
    llm.CURRENT_PURPOSE[0] = "chat"
    yield
    llm.USAGE.clear()
    llm.LAST_CALL.clear()
    llm.CURRENT_PURPOSE[0] = "chat"


def test_openai_usage_recorded_with_served_model(monkeypatch):
    class Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"},
                                            "finish_reason": "stop"}],
                               "model": "zai/glm-5.3-flash-x",
                               "usage": {"prompt_tokens": 11, "completion_tokens": 3}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        return Resp()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    llm.chat_json("sys", "user")
    assert llm.LAST_CALL["model"] == "zai/glm-5.3-flash-x"
    assert llm.LAST_CALL["prompt_tokens"] == 11
    assert llm.LAST_CALL["completion_tokens"] == 3
    key = ("zai/glm-5.3-flash-x", "chat")
    assert llm.USAGE[key] == {"calls": 1, "prompt_tokens": 11, "completion_tokens": 3}


def test_purpose_label_attributes_usage(monkeypatch):
    class Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"},
                                            "finish_reason": "stop"}],
                               "model": "m", "usage": {"prompt_tokens": 5,
                                                       "completion_tokens": 1}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    from app.facts import _Purpose
    with _Purpose("fact_extraction"):
        llm.chat_json("sys", "user")
    assert ("m", "fact_extraction") in llm.USAGE
    assert llm.LAST_CALL["purpose"] == "fact_extraction"


def test_es_inference_usage_recorded_from_final_chunk(monkeypatch):
    monkeypatch.setenv("AMES_LLM_PROVIDER", "es_inference")
    monkeypatch.setenv("AMES_LLM_INFERENCE_ID", "ep-1")
    monkeypatch.setenv("AMES_LLM_MODEL", "my-model")
    stream = sse(_chunked_with_usage("ok", "stop", {"prompt_tokens": 7, "completion_tokens": 2}))
    fake = Fake([(200, stream)])
    monkeypatch.setattr(llm.store, "_es_url", lambda: fake.url)
    monkeypatch.setattr(llm.store, "_es_headers", lambda: {"Authorization": "x"})
    out = llm.chat("q", [{"id": "am_1", "text": "t"}])
    assert out == "ok"
    assert llm.LAST_CALL["model"] == "my-model"
    assert llm.LAST_CALL["prompt_tokens"] == 7
    assert llm.USAGE[("my-model", "chat")]["completion_tokens"] == 2


def test_sse_usage_deltas_are_summed(monkeypatch):
    monkeypatch.setenv("AMES_LLM_PROVIDER", "es_inference")
    monkeypatch.setenv("AMES_LLM_INFERENCE_ID", "ep-1")
    monkeypatch.setenv("AMES_LLM_MODEL", "m2")
    c1 = json.dumps({"chat_completion": {"choices": [{"index": 0, "delta": {"content": "a"}}],
                                         "usage": {"prompt_tokens": 0, "completion_tokens": 1}}})
    c2 = json.dumps({"chat_completion": {"choices": [{"index": 0, "delta": {"content": "b"},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 0, "completion_tokens": 1}}})
    fake = Fake([(200, sse(c1, c2))])
    monkeypatch.setattr(llm.store, "_es_url", lambda: fake.url)
    monkeypatch.setattr(llm.store, "_es_headers", lambda: {})
    assert llm.chat("q", []) == "ab"
    assert llm.LAST_CALL["completion_tokens"] == 2


def test_missing_usage_is_not_recorded(monkeypatch):
    class Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"},
                                            "finish_reason": "stop"}], "model": "m"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    llm.chat_json("s", "u")
    assert llm.USAGE == {}
    assert llm.LAST_CALL == {}


def test_malformed_usage_does_not_fail_the_call(monkeypatch):
    class Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"},
                                            "finish_reason": "stop"}], "model": "m",
                               "usage": {"prompt_tokens": "n/a"}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda req, timeout=None: Resp())
    assert llm.chat_json("s", "u") == {"ok": True}
    assert llm.USAGE == {}


def test_sse_nested_usage_details_are_ignored(monkeypatch):
    monkeypatch.setenv("AMES_LLM_PROVIDER", "es_inference")
    monkeypatch.setenv("AMES_LLM_INFERENCE_ID", "ep-1")
    monkeypatch.setenv("AMES_LLM_MODEL", "m3")
    c1 = json.dumps({"chat_completion": {"choices": [{"index": 0, "delta": {"content": "a"},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 0, "completion_tokens": 4,
                                                   "prompt_tokens_details": {"cached_tokens": 0}}}})
    fake = Fake([(200, sse(c1))])
    monkeypatch.setattr(llm.store, "_es_url", lambda: fake.url)
    monkeypatch.setattr(llm.store, "_es_headers", lambda: {})
    assert llm.chat("q", []) == "a"
    assert llm.LAST_CALL["completion_tokens"] == 4


def _cache_fake_es(writes):
    def fake_es(method, path, body=None):
        if method == "GET" and "/_doc/" in path:
            raise RuntimeError(f'GET {path} -> 404: {{"found": false}}')
        if method == "PUT" and "/_create/" in path:
            writes.append(body)
            return {"result": "created"}
        raise AssertionError(f"unexpected {method} {path}")
    return fake_es


def test_cache_write_carries_this_calls_usage(monkeypatch):
    from app import facts

    def fake_extract(window, date):
        llm.record_usage("m", llm.CURRENT_PURPOSE[0], {"prompt_tokens": 9, "completion_tokens": 2})
        return ["fact one"]

    writes = []
    monkeypatch.setattr(facts, "es", _cache_fake_es(writes))
    monkeypatch.setattr(facts, "_extract", fake_extract)
    assert facts._document_facts([("p1", "text")], "2026-01-01") == ["fact one"]
    assert writes[0]["usage"] == {"model": "m", "purpose": "fact_extraction",
                                  "prompt_tokens": 9, "completion_tokens": 2}


def test_failed_extraction_usage_is_not_written_to_the_next_doc(monkeypatch):
    """Job A's call records usage then raises (bad JSON); job B's response has no
    usage. B's cache doc must not carry A's numbers."""
    from app import facts

    def failing_extract(window, date):
        llm.record_usage("m", llm.CURRENT_PURPOSE[0], {"prompt_tokens": 50, "completion_tokens": 5})
        raise ValueError("invalid JSON reply")

    writes = []
    monkeypatch.setattr(facts, "es", _cache_fake_es(writes))
    monkeypatch.setattr(facts, "_extract", failing_extract)
    with pytest.raises(ValueError):
        facts._document_facts([("a1", "doc A")], "2026-01-01")
    monkeypatch.setattr(facts, "_extract", lambda window, date: ["fact b"])
    assert facts._document_facts([("b1", "doc B")], "2026-01-01") == ["fact b"]
    assert writes == [{"facts": ["fact b"], "usage": {}, "created_at": writes[0]["created_at"]}]


def test_ensure_queue_cache_mapping_failure_is_loud(monkeypatch):
    from app import facts

    def fake_es(method, path, body=None):
        if method == "GET":
            return {"ok": True}
        if path == f"/{facts.CACHE}/_mapping":
            raise RuntimeError(f"PUT {path} -> 403: forbidden")
        return {"acknowledged": True}

    monkeypatch.setattr(facts, "es", fake_es)
    with pytest.raises(RuntimeError, match="403"):
        facts.ensure_queue()


def test_job_span_usage_is_a_per_job_delta():
    from app import facts
    llm.record_usage("m", "fact_extraction", {"prompt_tokens": 100, "completion_tokens": 10})
    start = facts._snapshot_usage()
    llm.record_usage("m", "fact_extraction", {"prompt_tokens": 7, "completion_tokens": 1})
    assert facts._usage_since(start) == {"m::fact_extraction": {
        "calls": 1, "prompt_tokens": 7, "completion_tokens": 1}}


def _chunked_with_usage(text, finish, usage):
    return json.dumps({"chat_completion": {"choices": [{"index": 0,
                                                        "delta": {"content": text},
                                                        "finish_reason": finish}],
                                           "usage": usage}})
