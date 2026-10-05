"""Reranker: ES-native _inference rerank (ES 8.14+ rerank API) over the fused list.
License-gated: falls back to identity ordering when the rerank endpoint 4xx's,
returning {'reranked': False} so callers never fake a rerank."""
import json
import os
import urllib.error
import urllib.request

from .store import ES_URL

# Cross-encoder call timeout (seconds). Configurable: a cold, CPU-only ES node needs several
# seconds per 50-passage batch, so this is where a deployment trades latency for recall quality.
RERANK_TIMEOUT = int(os.environ.get("AMES_RERANK_TIMEOUT", "30"))


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


def rerank(query: str, hits: list, top_n: int = 5,
           model: str = None) -> dict:
    """hits: [{'id','text',...}]. Returns {'hits': [...], 'reranked': bool}."""
    model = model or os.environ.get("AMES_RERANK_MODEL", ".rerank-v1-elasticsearch")
    if not hits:
        return {"hits": hits, "reranked": False}
    body = json.dumps({
        "query": query,
        "top_n": min(top_n, len(hits)),
        "input": [h["text"] for h in hits[:50]],  # plain strings; response keyed by index
    }).encode()
    req = urllib.request.Request(
        ES_URL + f"/_inference/rerank/{model}", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=RERANK_TIMEOUT) as r:
            out = json.load(r)
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, OSError, ValueError):
        # license/mapping/model unavailable → honest fallback, no fake rerank
        return {"hits": hits[:top_n], "reranked": False}
    if not isinstance(out, dict):
        return {"hits": hits[:top_n], "reranked": False}
    raw = out.get("rerank")
    if not isinstance(raw, list):
        return {"hits": hits[:top_n], "reranked": False}
    cap = min(len(hits), 50)
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
        return {"hits": hits[:top_n], "reranked": False}
    return {"hits": ranked, "reranked": True}
