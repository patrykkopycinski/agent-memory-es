"""Mental models: user-curated summaries for frequent queries, consulted FIRST
by recall/reflect (priority tier above raw facts). Stored in am_models.

Drafts (worker-proposed, never auto-promoted) carry status=draft; only an
explicit upsert via the owner's key creates status=active. Staleness: a model
is stale when its owner has active semantic facts newer than the model that
BM25-match its question pattern."""
import datetime
from typing import Optional

from .store import es, PREFIX

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
            "status": {"type": "keyword"},           # active | draft
            "source_ids": {"type": "keyword"},       # draft: evidence fact ids
        },
    },
}


def ensure_models_index() -> None:
    try:
        es("GET", f"/{PREFIX}am_models")
    except RuntimeError:
        es("PUT", f"/{PREFIX}am_models", MODEL_MAPPINGS)


def upsert_model(owner_id: str, question_pattern: str, summary: str,
                 visibility: str = "private") -> dict:
    """Explicit owner-curated model — the ONLY path to status=active."""
    ensure_models_index()
    doc = {
        "owner_id": owner_id, "visibility": visibility,
        "question_pattern": question_pattern, "summary": summary,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "active",
    }
    r = es("POST", f"/{PREFIX}am_models/_doc?refresh=true", doc)
    return {"id": r["_id"], **doc}


def propose_draft(owner_id: str, question_pattern: str, summary: str,
                  source_ids: list, visibility: str = "private") -> dict:
    """Worker-proposed draft — NEVER enters the priority tier; promotion is
    an explicit upsert_model by the owner."""
    ensure_models_index()
    doc = {
        "owner_id": owner_id, "visibility": visibility,
        "question_pattern": question_pattern, "summary": summary,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "draft", "source_ids": source_ids,
    }
    r = es("POST", f"/{PREFIX}am_models/_doc?refresh=true", doc)
    return {"id": r["_id"], **doc}


def list_drafts(owner_id: str) -> list:
    ensure_models_index()
    r = es("POST", f"/{PREFIX}am_models/_search", {
        "size": 50,
        "query": {"bool": {"filter": [{"term": {"owner_id": owner_id}},
                                      {"term": {"status": "draft"}}]}},
        "sort": [{"updated_at": {"order": "desc"}}]})
    return [{"id": h["_id"], **h["_source"]} for h in r["hits"]["hits"]]


def staleness(owner_id: str, model: dict) -> bool:
    """True when the owner has ACTIVE semantic facts newer than the model whose
    text matches the model's question_pattern (BM25 min_score 1.0)."""
    from .store import idx
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": 1,
        "query": {"bool": {
            "must": [{"match": {"text": {"query": model["question_pattern"]}}}],
            "filter": [{"term": {"owner_id": owner_id}},
                       {"term": {"active": True}},
                       {"range": {"occurred_at": {"gt": model["updated_at"]}}}],
        }},
        "min_score": 1.0,
    })
    return r["hits"]["total"]["value"] > 0


def match_model(owner_id: str, query: str, size: int = 1) -> Optional[dict]:
    """Best mental model whose question_pattern matches the query (BM25, top-1).
    Returns None when nothing scores."""
    ensure_models_index()
    r = es("POST", f"/{PREFIX}am_models/_search", {
        "size": size,
        "query": {"bool": {
            "must": [{"match": {"question_pattern": {"query": query}}}],
            "filter": [{"bool": {"should": [
                {"term": {"owner_id": owner_id}},
                {"bool": {"must_not": {"term": {"visibility": "private"}}}},
            ]}}, {"term": {"status": "active"}}],
        }},
    })
    hits = r["hits"]["hits"]
    if not hits:
        return None
    m = {"id": hits[0]["_id"], "score": hits[0]["_score"], **hits[0]["_source"]}
    try:
        m["stale"] = staleness(owner_id, m)
    except Exception:
        pass  # staleness is advisory; never block recall on it
    return m


def list_models(owner_id: str, status: str = "active", size: int = 50) -> list:
    """All models of a status the owner can see (own + shared), newest first."""
    ensure_models_index()
    r = es("POST", f"/{PREFIX}am_models/_search", {
        "size": size,
        "query": {"bool": {"filter": [
            {"term": {"status": status}},
            {"bool": {"should": [
                {"term": {"owner_id": owner_id}},
                {"bool": {"must_not": {"term": {"visibility": "private"}}}},
            ]}},
        ]}},
        "sort": [{"updated_at": {"order": "desc"}}]})
    return [{"id": h["_id"], **h["_source"]} for h in r["hits"]["hits"]]
