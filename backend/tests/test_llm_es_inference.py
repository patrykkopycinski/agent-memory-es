"""AMES_LLM_PROVIDER=es_inference: SSE assembly against a fake _inference endpoint, retry
semantics, chat_json parsing, and the unchanged openai path. No Elasticsearch needed."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import http_retry, llm, store  # noqa: E402


def _chunk(text=None, finish=None):
    choice = {"index": 0, "delta": {"content": text} if text is not None else {}}
    if finish:
        choice["finish_reason"] = finish
    return json.dumps({"id": "c1", "choices": [choice], "object": "chat.completion.chunk"})


class Fake:
    """Local HTTP server; `script` is a list of (status, body-bytes) answers served in order."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.requests.append({"path": self.path, "headers": dict(self.headers),
                                       "body": json.loads(self.rfile.read(n))})
                status, payload = outer.script.pop(0) if outer.script else (500, b"exhausted")
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if status == 200 else "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def sse(*chunks, done=True, event=False):
    lines = []
    for c in chunks:
        lines.append(("event: message\n" if event else "") + f"data: {c}\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


@pytest.fixture
def es_inference(monkeypatch):
    made = []

    def start(script, **env):
        fake = Fake(script)
        made.append(fake)
        monkeypatch.setenv("AMES_LLM_PROVIDER", "es_inference")
        monkeypatch.setenv("AMES_ES_URL", fake.url)
        monkeypatch.setattr(store, "_es_headers", lambda: {
            "Content-Type": "application/json", "Authorization": "ApiKey es-key"})
        monkeypatch.setenv("AMES_LLM_INFERENCE_ID", ".rainbow-sprinkles-elastic")
        monkeypatch.delenv("AMES_LLM_MODEL", raising=False)
        monkeypatch.delenv("AMES_LLM_TEMPERATURE", raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(http_retry, "_sleep_seconds", lambda *_: 0.0)
        return fake
    yield start
    for f in made:
        f.close()


def test_sse_chunks_are_assembled_and_done_ends_the_stream(es_inference):
    fake = es_inference([(200, sse(_chunk("Hel"), _chunk("lo "), _chunk("world"), _chunk(None, "stop")))])
    assert llm.chat("q?", [{"id": "am_1", "text": "fact"}]) == "Hello world"
    req = fake.requests[0]
    assert req["path"] == "/_inference/chat_completion/.rainbow-sprinkles-elastic/_stream"
    assert req["headers"]["Authorization"] == "ApiKey es-key"
    assert "model" not in req["body"] and req["body"]["temperature"] == 0
    assert req["body"]["messages"][0]["role"] == "system"
    assert "response_format" not in req["body"]


def test_event_lines_and_wrapped_chunks_are_tolerated(es_inference):
    wrapped = json.dumps({"chat_completion": json.loads(_chunk("ok"))})
    es_inference([(200, sse(wrapped, _chunk("!"), event=True))])
    assert llm.chat("q", []) == "ok!"


def test_explicit_model_and_temperature_omit(es_inference):
    fake = es_inference([(200, sse(_chunk("x")))], AMES_LLM_MODEL="gpt-x", AMES_LLM_TEMPERATURE="omit")
    llm.chat("q", [])
    assert fake.requests[0]["body"]["model"] == "gpt-x"
    assert "temperature" not in fake.requests[0]["body"]


def test_4xx_is_not_retried_and_is_raised(es_inference):
    fake = es_inference([(400, b'{"error":"bad"}'), (200, sse(_chunk("late")))])
    with pytest.raises(Exception) as exc:
        llm.chat("q", [])
    assert getattr(exc.value, "code", None) == 400
    assert len(fake.requests) == 1


def test_5xx_and_429_are_retried(es_inference):
    fake = es_inference([(503, b"{}"), (429, b"{}"), (200, sse(_chunk("fine")))])
    assert llm.chat("q", []) == "fine"
    assert len(fake.requests) == 3


def test_truncated_stream_is_retried(es_inference):
    fake = es_inference([(200, sse(_chunk("par"), done=False)), (200, sse(_chunk("whole")))])
    assert llm.chat("q", []) == "whole"
    assert len(fake.requests) == 2


def test_error_event_raises_without_echoing_the_body(es_inference):
    es_inference([(200, sse(json.dumps({"error": {"reason": "PROMPT-CANARY"}})))])
    with pytest.raises(ValueError) as exc:
        llm.chat("q", [])
    assert "PROMPT-CANARY" not in str(exc.value)


def test_chat_json_parses_via_es_inference_without_response_format(es_inference):
    fake = es_inference([(200, sse(_chunk('{"a":'), _chunk(' 1}'), _chunk(None, "stop")))])
    assert llm.chat_json("sys", "user text") == {"a": 1}
    body = fake.requests[0]["body"]
    assert "response_format" not in body and "chat_template_kwargs" not in body
    assert "json" in body["messages"][1]["content"].lower()


def test_chat_json_tolerates_fence_and_keeps_safe_errors(es_inference):
    es_inference([(200, sse(_chunk('```json\n{"a": 2}\n```')))])
    assert llm.chat_json("s", "u") == {"a": 2}
    es_inference([(200, sse(_chunk('{"secret":"CANARY"'), _chunk(None, "length")))])
    with pytest.raises(ValueError) as exc:
        llm.chat_json("s", "u")
    assert "CANARY" not in str(exc.value) and "finish_reason='length'" in str(exc.value)
    assert exc.value.__cause__ is None


def test_rewrite_queries_via_es_inference(es_inference):
    es_inference([(200, sse(_chunk("1. first query\n2. second"), _chunk(" query")))])
    assert llm.rewrite_queries("q") == ["first query", "second query"]


def test_missing_inference_id_and_bad_provider(es_inference, monkeypatch):
    es_inference([], AMES_LLM_INFERENCE_ID=" ")
    with pytest.raises(RuntimeError, match="misconfigured") as missing:
        llm.chat("q", [])
    assert "AMES_LLM" not in str(missing.value)
    monkeypatch.setenv("AMES_LLM_PROVIDER", "bedrock")
    with pytest.raises(ValueError, match="misconfigured") as bad:
        llm.chat("q", [])
    assert "bedrock" not in str(bad.value)


def test_validate_config_fails_fast_without_echoing_values(monkeypatch):
    monkeypatch.delenv("AMES_LLM_PROVIDER", raising=False)
    llm.validate_config()
    monkeypatch.setenv("AMES_LLM_PROVIDER", "bedrock")
    with pytest.raises(ValueError, match="AMES_LLM_PROVIDER") as bad:
        llm.validate_config()
    assert "bedrock" not in str(bad.value)
    monkeypatch.setenv("AMES_LLM_PROVIDER", "es_inference")
    monkeypatch.setenv("AMES_LLM_INFERENCE_ID", " ")
    with pytest.raises(ValueError, match="AMES_LLM_INFERENCE_ID"):
        llm.validate_config()
    monkeypatch.setenv("AMES_LLM_INFERENCE_ID", ".some-endpoint")
    llm.validate_config()


def test_app_startup_refuses_a_bad_llm_provider(monkeypatch, caplog):
    from app import main
    monkeypatch.setenv("AMES_LLM_PROVIDER", "bedrock")
    with pytest.raises(ValueError):
        main._startup()
    assert "invalid configuration" in caplog.text


def test_openai_null_content_returns_empty_string(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            data = json.dumps({"choices": [{"finish_reason": "length", "message": {"content": None}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.delenv("AMES_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(llm, "BASE", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(llm, "KEY", "k")
    try:
        assert llm.chat("q", []) == ""
    finally:
        server.shutdown()
        server.server_close()


def test_openai_path_is_unchanged(monkeypatch):
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}]}
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen["path"], seen["auth"] = self.path, self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(n))
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.delenv("AMES_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AMES_LLM_TEMPERATURE", raising=False)
    monkeypatch.setattr(llm, "BASE", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(llm, "KEY", "k")
    monkeypatch.setattr(llm, "MODEL", "m")
    try:
        assert llm.chat_json("sys", "u") == {"ok": True}
    finally:
        server.shutdown()
        server.server_close()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["temperature"] == 0
