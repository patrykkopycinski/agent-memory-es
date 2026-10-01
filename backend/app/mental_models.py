"""Mental models: user-curated summaries for frequent queries, consulted FIRST
by recall/reflect (priority tier above raw facts). Stored in am_models."""
from typing import Optional

from .store import es

MODEL_MAPPINGS = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "owner_id": {"type": "keyword"},
            "visibility": {"type": "keyword"},
            "question_pattern": {"type": "text"},   # matched loosely vs the query
            "summary": {"type": "text"},             # the curated answer
            "updated_at": {"type": "date"},
        },
    },
}


def ensure_models_index() -> None:
    try:
        es("GET", "/am_models")
    except RuntimeError:
        es("PUT", "/am_models", MODEL_MAPPINGS)


def upsert_model(owner_id: str, question_pattern: str, summary: str,
                 visibility: str = "private") -> dict:
    ensure_models_index()
    import datetime
    doc = {
        "owner_id": owner_id, "visibility": visibility,
        "question_pattern": question_pattern, "summary": summary,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r = es("POST", "/am_models/_doc?refresh=true", doc)
    return {"id": r["_id"], **doc}


def match_model(owner_id: str, query: str, size: int = 1) -> Optional[dict]:
    """Best mental model whose question_pattern matches the query (BM25, top-1).
    Returns None when nothing scores."""
    ensure_models_index()
    r = es("POST", "/am_models/_search", {
        "size": size,
        "query": {"bool": {
            "must": [{"match": {"question_pattern": {"query": query}}}],
            "filter": [{"bool": {"should": [
                {"term": {"owner_id": owner_id}},
                {"bool": {"must_not": {"term": {"visibility": "private"}}}},
            ]}}],
        }},
    })
    hits = r["hits"]["hits"]
    if not hits:
        return None
    return {"id": hits[0]["_id"], "score": hits[0]["_score"], **hits[0]["_source"]}
