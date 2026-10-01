"""Embeddings via OmniRoute gateway (OpenAI-compatible /v1/embeddings).
Model: gemini-embedding-2 via openrouter. dim discovered on first call."""
import json
import os
import urllib.request
from typing import Optional

BASE = os.environ.get("AMES_EMBED_BASE", "http://localhost:20128/v1")
MODEL = os.environ.get("AMES_EMBED_MODEL", "openrouter/google/gemini-embedding-2")
KEY = os.environ.get("AMES_EMBED_KEY", os.environ.get("OMNIROUTE_API_KEY", ""))
DIM = int(os.environ.get("AMES_EMBED_DIM", "0"))  # 0 = discover

_cache_dim = None


def api_key() -> str:
    global KEY
    if not KEY:
        KEY = os.environ.get("OMNIROUTE_API_KEY", "")
    return KEY


def embed(texts: list) -> list:
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    req = urllib.request.Request(BASE + "/embeddings", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    return [d["embedding"] for d in out["data"]]


def dim() -> int:
    global _cache_dim
    if _cache_dim:
        return _cache_dim
    _cache_dim = DIM or len(embed(["probe"])[0])
    return _cache_dim
