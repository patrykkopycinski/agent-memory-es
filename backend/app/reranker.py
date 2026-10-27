"""Reranker: ES-native _inference rerank (ES 8.14+ rerank API) over the fused list.
License-gated: falls back to identity ordering when the rerank endpoint 4xx's,
returning {'reranked': False, 'fallback_reason': ...} so callers never fake a rerank."""
import json
import os
import socket
import time
import urllib.error
import urllib.request

from .store import _es_url

# Cross-encoder call budget (seconds). Recall is an interactive call, so rerank gets a short
# leash: past this the fused order is served and the response says why (fallback_reason).
# A slower deployment can raise it explicitly; the default is NOT sized for a cold CPU node.
RERANK_TIMEOUT = float(os.environ.get("AMES_RERANK_TIMEOUT", "2"))
# Hard ceiling on passages sent per call (cross-encoder cost is linear in this).
MAX_INPUT = 50
# Fallback reasons surfaced to callers: "timeout" (budget exceeded), "unavailable" (endpoint
# missing / license-gated / unreachable), "error" (server error or unusable response).
_UNAVAILABLE_HTTP = (400, 401, 402, 403, 404)


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback(hits: list, top_n: int, reason: str, t0: float) -> dict:
    return {"hits": hits[:top_n], "reranked": False, "fallback_reason": reason,
            "ms": int((time.monotonic() - t0) * 1000)}


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (socket.timeout, TimeoutError))


def rerank(query: str, hits: list, top_n: int = 5,
           model: str = None, timeout: float = None) -> dict:
    """hits: [{'id','text',...}]. Returns {'hits': [...], 'reranked': bool, 'ms': int}; when not
    reranked also 'fallback_reason' in {timeout, unavailable, error}."""
    model = model or os.environ.get("AMES_RERANK_MODEL", ".rerank-v1-elasticsearch")
    t0 = time.monotonic()
    if not hits:
        return {"hits": hits, "reranked": False, "fallback_reason": "error", "ms": 0}
    body = json.dumps({
        "query": query,
        "top_n": min(top_n, len(hits)),
        "input": [h["text"] for h in hits[:MAX_INPUT]],  # plain strings; response keyed by index
    }).encode()
    req = urllib.request.Request(
        _es_url() + f"/_inference/rerank/{model}", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=RERANK_TIMEOUT if timeout is None else timeout) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        # license/mapping/model unavailable → honest fallback, no fake rerank
        return _fallback(hits, top_n, "unavailable" if e.code in _UNAVAILABLE_HTTP else "error", t0)
    except (urllib.error.URLError, RuntimeError, OSError, ValueError) as e:
        if _is_timeout(e):
            return _fallback(hits, top_n, "timeout", t0)
        unreachable = isinstance(e, (urllib.error.URLError, ConnectionError))
        return _fallback(hits, top_n, "unavailable" if unreachable else "error", t0)
    if not isinstance(out, dict):
        return _fallback(hits, top_n, "error", t0)
    raw = out.get("rerank")
    if not isinstance(raw, list):
        return _fallback(hits, top_n, "error", t0)
    cap = min(len(hits), MAX_INPUT)
    ranked, seen = [], set()
    for item in raw[:top_n]:
        # a partial or malformed response is DROPPED item by item (never raises, never invents a
        # score): the caller keeps the fused order for everything the endpoint did not cover.
        if not isinstance(item, dict):
            continue
        i = _as_int(item.get("index"))
        if i is None or not (0 <= i < cap) or i in seen:
            continue
        with_score = dict(hits[i])
        with_score["rerank_score"] = _as_float(item.get("relevance_score"))
        ranked.append(with_score)
        seen.add(i)
    if not ranked:
        return _fallback(hits, top_n, "error", t0)
    return {"hits": ranked, "reranked": True, "ms": int((time.monotonic() - t0) * 1000)}
