"""Graph retrieval arm: entity-keyed two-hop expansion fused into recall.

Query entities are matched against the `entities` keyword field (hop 1); facts
sharing entities with hop-1 facts come back even when they share no query words
(hop 2). Fused by RRF rank with lexical/kNN arms in memory.recall.
"""
from .store import es, idx


def graph_expand(owner_id: str, query: str, size: int = 10,
                 vis_filter: dict = None) -> list:
    """Return hop-2 facts for the query's entities, visibility-filtered."""
    from .memory import extract_entities  # late import: shared tokenizer
    ents = extract_entities(query, limit=6)
    if not ents or vis_filter is None:
        return []
    filt = [{"term": {"active": True}}, vis_filter]
    # hop 1: facts mentioning any query entity
    r1 = es("POST", f"/{idx('semantic')}/_search", {
        "size": 20,
        "query": {"bool": {
            "must": [{"terms": {"entities": ents}}],
            "filter": filt,
        }},
        "_source": ["entities"],
    })
    hop1_entities = set()
    for h in r1["hits"]["hits"]:
        hop1_entities.update(h["_source"].get("entities", []))
    hop1_entities -= set(ents)
    if not hop1_entities:
        return []
    # hop 2: active facts sharing hop-1 entities, excluding hop-1 docs themselves
    r2 = es("POST", f"/{idx('semantic')}/_search", {
        "size": size,
        "query": {"bool": {
            "must": [{"terms": {"entities": sorted(hop1_entities)[:64]}}],
            "must_not": [{"terms": {"_id": [h["_id"] for h in r1["hits"]["hits"]]}}],
            "filter": filt,
        }},
    })
    return [{"id": h["_id"], "score": h["_score"],
             **{k: v for k, v in h["_source"].items() if k != "embedding"}}
            for h in r2["hits"]["hits"]]
