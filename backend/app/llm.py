"""Reflect LLM wiring: OmniRoute chat completion synthesizes an answer from recalled
evidence. Cites only retrieved ids. Abstains (atlas finding) when evidence is thin."""
import json
import os
import urllib.request

BASE = os.environ.get("AMES_LLM_BASE", "http://localhost:20128/v1")
MODEL = os.environ.get("AMES_LLM_MODEL", "auto/best-chat")
KEY = os.environ.get("AMES_LLM_KEY", os.environ.get("OMNIROUTE_API_KEY", ""))

SYSTEM = (
    "You answer strictly from the EVIDENCE memories provided. "
    "Cite the memory ids you used like [am_xxx]. If the evidence does not "
    "contain the answer, reply exactly: INSUFFICIENT_EVIDENCE."
)


def chat(question: str, evidence: list) -> str:
    body = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "QUESTION: " + question + "\n\nEVIDENCE:\n" +
             "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)},
        ],
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"]


REWRITE_SYSTEM = (
    "Rewrite the question as 2 alternative search queries for a memory bank. "
    "Use different vocabulary than the original. Reply with one query per line, "
    "nothing else."
)


def rewrite_queries(question: str, n: int = 2) -> list:
    """Generate alternative recall queries for reflect's multi-round loop."""
    body = json.dumps({
        "model": MODEL, "temperature": 0.3,
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": question},
        ],
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    lines = [l.strip().lstrip("0123456789.-) ") for l in
             out["choices"][0]["message"]["content"].splitlines() if l.strip()]
    return lines[:n]


def chat_json(system: str, user: str, timeout: int = 120) -> dict:
    """One temperature-0 chat completion that must return a JSON object (label extraction).
    Raises on transport errors, empty content or a non-object body: the caller decides policy."""
    body = json.dumps({
        "model": MODEL, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    content = (out["choices"][0]["message"].get("content") or "").strip()
    if content.startswith("```"):                      # tolerate a fenced reply
        content = content.strip("`")
        if content[:4].lower() == "json":
            content = content[4:]
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError("extractor reply is not a JSON object")
    return obj
