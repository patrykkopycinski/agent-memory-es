"""Embeddings: ES-native inference first (`_inference/text_embedding`), OmniRoute fallback.

Default endpoint .multilingual-e5-small-elasticsearch (384-dim, cluster-side, no external
gateway). AMES_EMBED_BACKEND=es|omniroute pins one. Dims differ per backend — indices must
be rebuilt when switching backends.
"""
import json
import logging
import os
import threading
import time
import urllib.request

from . import http_retry
from .store import _es_url

log = logging.getLogger("ames.embed")
ES_ENDPOINT = os.environ.get("AMES_ES_INFERENCE", ".multilingual-e5-small-elasticsearch")
OR_BASE = os.environ.get("AMES_EMBED_BASE", "http://localhost:20128/v1")
OR_MODEL = os.environ.get("AMES_EMBED_MODEL", "openrouter/google/gemini-embedding-2")

_cache_dim = None
_backend = None


def _post(url, body, headers=None, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    return http_retry.request_json(req, timeout=timeout)


def _es_embed(texts):
    out = _post(f"{_es_url()}/_inference/text_embedding/{ES_ENDPOINT}", {"input": texts})
    return [d["embedding"] for d in out["text_embedding"]]


def _or_embed(texts):
    key = os.environ.get("AMES_EMBED_KEY", os.environ.get("OMNIROUTE_API_KEY", ""))
    out = _post(f"{OR_BASE}/embeddings", {"model": OR_MODEL, "input": texts},
                headers={"Authorization": f"Bearer {key}"})
    return [d["embedding"] for d in out["data"]]


def backend() -> str:
    global _backend
    if _backend:
        return _backend
    pinned = os.environ.get("AMES_EMBED_BACKEND", "")
    if pinned in ("es", "omniroute"):
        _backend = pinned
    else:
        try:
            _es_embed(["probe"])
            _backend = "es"
        except Exception:
            _backend = "omniroute"
    return _backend


def embed(texts: list) -> list:
    return _es_embed(texts) if backend() == "es" else _or_embed(texts)


# Process-wide embed failure counter. embed() failures on the retain/recall hot path used to be
# swallowed silently (a license expiry left retain writing vector-less docs for 5 days); every
# swallowed failure now lands here and is surfaced by /health and /memory/stats.
FAILURES = {"count": 0, "last_error": None, "last_at": None}
_failures_lock = threading.Lock()


def describe_error(exc: BaseException) -> str:
    """Short, single-line error string safe to return in an API response."""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"


def embed_safe(texts: list) -> tuple:
    """embed() that never raises: (vectors, None) on success, (None, error_string) on failure.
    Failures are counted in FAILURES and logged. Looks up `embed` at call time so tests can
    monkeypatch it."""
    try:
        return embed(texts), None
    except Exception as exc:
        err = describe_error(exc)
        with _failures_lock:
            FAILURES["count"] += 1
            FAILURES["last_error"] = err
            FAILURES["last_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log.warning("ames.embed failed (writes land vector-less, recall loses its vector arm): %s", err)
        return None, err


def dim() -> int:
    global _cache_dim
    if _cache_dim:
        return _cache_dim
    _cache_dim = int(os.environ.get("AMES_EMBED_DIM", "0")) or len(embed(["probe"])[0])
    return _cache_dim
