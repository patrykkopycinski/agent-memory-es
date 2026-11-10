"""Reflect LLM wiring: OmniRoute chat completion synthesizes an answer from recalled
evidence. Cites only retrieved ids. Abstains (atlas finding) when evidence is thin."""
import json
import os
import urllib.parse
import urllib.request

from . import http_retry, store

BASE = os.environ.get("AMES_LLM_BASE", "http://localhost:20128/v1")
MODEL = os.environ.get("AMES_LLM_MODEL", "auto/best-chat")
KEY = os.environ.get("AMES_LLM_KEY", os.environ.get("OMNIROUTE_API_KEY", ""))


def _sampling(default):
    """AMES_LLM_TEMPERATURE=omit (or empty) => send no temperature key at all
    (gh/gpt-6-luna rejects the parameter); otherwise the configured value."""
    value = os.environ.get("AMES_LLM_TEMPERATURE")
    if value is None:
        return {"temperature": default}
    value = value.strip()
    if not value or value.lower() == "omit":
        return {}
    return {"temperature": float(value)}


def _template_kwargs():
    """AMES_LLM_DISABLE_THINKING=1 => send chat_template_kwargs.enable_thinking=false.
    For models whose chat template thinks by default and can spend the whole
    max_tokens budget on reasoning, returning empty content. Leave unset for
    providers that reject unknown request fields."""
    if os.environ.get("AMES_LLM_DISABLE_THINKING") == "1":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


SYSTEM = (
    "You answer strictly from the EVIDENCE memories provided. "
    "Cite the memory ids you used like [am_xxx]. If the evidence does not "
    "contain the answer, reply exactly: INSUFFICIENT_EVIDENCE."
)


def _provider() -> str:
    """AMES_LLM_PROVIDER=openai (default: OpenAI-compatible gateway at AMES_LLM_BASE) or
    es_inference (the Elasticsearch _inference chat_completion API on AMES_ES_URL with the same
    auth as store.es, endpoint id from AMES_LLM_INFERENCE_ID)."""
    provider = os.environ.get("AMES_LLM_PROVIDER", "openai").strip().lower() or "openai"
    if provider not in ("openai", "es_inference"):
        raise ValueError("llm provider misconfigured")    # per-call: no env values to clients
    return provider


def validate_config() -> None:
    """Fail fast at startup (app and worker) with a log-only explanation; values are not echoed."""
    provider = os.environ.get("AMES_LLM_PROVIDER", "openai").strip().lower() or "openai"
    if provider not in ("openai", "es_inference"):
        raise ValueError("AMES_LLM_PROVIDER must be 'openai' or 'es_inference'")
    if provider == "es_inference" and not os.environ.get("AMES_LLM_INFERENCE_ID", "").strip():
        raise ValueError("AMES_LLM_INFERENCE_ID is required when AMES_LLM_PROVIDER=es_inference")


def _parse_sse(response) -> tuple:
    """Assemble (content, finish_reason) from a chat_completion SSE stream. Accepts the bare chunk
    (`choices[].delta.content`) and the `{"chat_completion": {...}}` wrapper; skips `event:` lines.
    A stream that ends without `[DONE]` was cut off: ConnectionResetError, which http_retry retries."""
    parts, finish, done = [], None, False
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        if "error" in chunk:
            raise ValueError("inference stream returned an error event")   # body may echo the prompt
        for choice in chunk.get("chat_completion", chunk).get("choices") or []:
            parts.append((choice.get("delta") or {}).get("content") or "")
            finish = choice.get("finish_reason") or finish
    if not done:
        raise ConnectionResetError("inference stream ended before [DONE]")
    return "".join(parts), finish


def _complete(messages: list, sampling: dict, json_mode: bool, timeout: float) -> tuple:
    """One chat completion on the configured provider -> (content, finish_reason).
    json_mode asks for a JSON object: on openai via response_format (+ the optional
    disable-thinking template kwarg); on es_inference by prompt alone, since the _inference API
    rejects unknown request fields."""
    if _provider() == "es_inference":
        inference_id = os.environ.get("AMES_LLM_INFERENCE_ID", "").strip()
        if not inference_id:
            raise RuntimeError("llm provider misconfigured")
        body = {"messages": messages, **sampling}
        model = os.environ.get("AMES_LLM_MODEL")
        if model:
            body["model"] = model
        url = (store._es_url() + "/_inference/chat_completion/"
               + urllib.parse.quote(inference_id, safe="") + "/_stream")
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={**store._es_headers(), "Accept": "text/event-stream"})
        return http_retry.request_with(req, _parse_sse, timeout=timeout)
    body = {"model": MODEL, **sampling, "messages": messages}
    if json_mode:
        body.update(_template_kwargs())
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    choice = http_retry.request_json(req, timeout=timeout)["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason")


def chat(question: str, evidence: list) -> str:
    content, _ = _complete([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "QUESTION: " + question + "\n\nEVIDENCE:\n" +
         "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)},
    ], _sampling(0), False, 120)
    return content


REWRITE_SYSTEM = (
    "Rewrite the question as 2 alternative search queries for a memory bank. "
    "Use different vocabulary than the original. Reply with one query per line, "
    "nothing else."
)


def rewrite_queries(question: str, n: int = 2) -> list:
    """Generate alternative recall queries for reflect's multi-round loop."""
    content, _ = _complete([
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": question},
    ], _sampling(0.3), False, 60)
    lines = [l.strip().lstrip("0123456789.-) ") for l in
             content.splitlines() if l.strip()]
    return lines[:n]


def chat_json(system: str, user: str, timeout: int = 120) -> dict:
    """One temperature-0 chat completion that must return a JSON object (label extraction).
    Raises on transport errors, empty content or a non-object body: the caller decides policy."""
    raw, reason = _complete([
        {"role": "system", "content": system},
        {"role": "user", "content": "Return a json object.\n\n" + user}],
        _sampling(0), True, timeout)
    content = raw.strip()
    if content.startswith("```"):                      # tolerate a fenced reply
        content = content.strip("`")
        if content[:4].lower() == "json":
            content = content[4:]
    reason = reason if reason in ("stop", "length", "content_filter", "tool_calls") else "other"
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        # Do not include the completion or the parser exception (which carries it).
        raise ValueError(f"invalid JSON reply: finish_reason={reason!r} "
                         f"raw_length={len(raw)} parsed_length={len(content)} "
                         f"error_position={exc.pos}") from None
    if not isinstance(obj, dict):
        raise ValueError(f"extractor reply is not a JSON object: "
                         f"finish_reason={reason!r} "
                         f"raw_length={len(raw)} parsed_length={len(content)}")
    return obj
