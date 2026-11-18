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


def _chunked_with_usage(text, finish, usage):
    return json.dumps({"chat_completion": {"choices": [{"index": 0,
                                                        "delta": {"content": text},
                                                        "finish_reason": finish}],
                                           "usage": usage}})
