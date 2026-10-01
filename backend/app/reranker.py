"""Reranker: ES-native _inference rerank (ES 8.14+ rerank API) over the fused list.
License-gated: falls back to identity ordering when the rerank endpoint 4xx's,
returning {'reranked': False} so callers never fake a rerank."""
import json
import os
import urllib.error
import urllib.request

from .store import ES_URL


def rerank(query: str, hits: list, top_n: int = 5,
           model: str = None) -> dict:
    """hits: [{'id','text',...}]. Returns {'hits': [...], 'reranked': bool}."""
    model = model or os.environ.get("AMES_RERANK_MODEL", ".rerank-v1-elasticsearch")
    if not hits:
        return {"hits": hits, "reranked": False}
    body = json.dumps({
        "model": model,
        "query": query,
        "top_n": min(top_n, len(hits)),
        "input": [{"id_": h["id"], "text": h["text"]} for h in hits[:50]],
        "inference_id": True,
    }).encode()
    req = urllib.request.Request(
        ES_URL + "/_inference/rerank", data=body, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.load(r)
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError):
        # license/mapping/model unavailable → honest fallback, no fake rerank
        return {"hits": hits[:top_n], "reranked": False}
    by_id = {h["id"]: h for h in hits}
    ranked = []
    for item in out.get("rerank", [])[:top_n]:
        hid = item.get("id_") or item.get("index")
        if hid in by_id:
            h = dict(by_id[hid])
            h["rerank_score"] = item.get("relevance_score")
            ranked.append(h)
    if not ranked:
        return {"hits": hits[:top_n], "reranked": False}
    return {"hits": ranked, "reranked": True}
