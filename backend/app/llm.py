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
