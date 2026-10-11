"""Reflect LLM wiring: OmniRoute chat completion synthesizes an answer from recalled
evidence. Cites only retrieved ids. Abstains (atlas finding) when evidence is thin."""
import json
import os
import urllib.request

from . import http_retry

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


SYSTEM = (
    "You answer strictly from the EVIDENCE memories provided. "
    "Cite the memory ids you used like [am_xxx]. If the evidence does not "
    "contain the answer, reply exactly: INSUFFICIENT_EVIDENCE."
)


def chat(question: str, evidence: list) -> str:
    body = json.dumps({
        "model": MODEL,
        **_sampling(0),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "QUESTION: " + question + "\n\nEVIDENCE:\n" +
             "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)},
        ],
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    out = http_retry.request_json(req, timeout=120)
    return out["choices"][0]["message"]["content"]


REWRITE_SYSTEM = (
    "Rewrite the question as 2 alternative search queries for a memory bank. "
    "Use different vocabulary than the original. Reply with one query per line, "
    "nothing else."
)


def rewrite_queries(question: str, n: int = 2) -> list:
    """Generate alternative recall queries for reflect's multi-round loop."""
    body = json.dumps({
        "model": MODEL, **_sampling(0.3),
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": question},
        ],
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    out = http_retry.request_json(req, timeout=60)
    lines = [l.strip().lstrip("0123456789.-) ") for l in
             out["choices"][0]["message"]["content"].splitlines() if l.strip()]
    return lines[:n]


def chat_json(system: str, user: str, timeout: int = 120) -> dict:
    """One temperature-0 chat completion that must return a JSON object (label extraction).
    Raises on transport errors, empty content or a non-object body: the caller decides policy."""
    body = json.dumps({
        "model": MODEL, **_sampling(0),
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": "Return a json object.\n\n" + user}],
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    out = http_retry.request_json(req, timeout=timeout)
    choice = out["choices"][0]
    raw = choice["message"].get("content") or ""
    content = raw.strip()
    if content.startswith("```"):                      # tolerate a fenced reply
        content = content.strip("`")
        if content[:4].lower() == "json":
            content = content[4:]
    reason = choice.get("finish_reason")
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
