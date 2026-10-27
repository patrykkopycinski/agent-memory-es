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


def normalize_pattern(pattern: str) -> str:
    """Canonical form for exact draft/cover matching: lowercase,
    whitespace-collapsed."""
    return " ".join(pattern.lower().split())


def find_models(owner_id: str, pattern: str,
                status: Optional[str] = None) -> list:
    """Exact normalized question_pattern matches among the owner's models;
    optional status filter. Unlike match_model (BM25, loose), this is exact —
    worker cover checks need it because every generated pattern shares the
    'facts and conventions' tail."""
    ensure_models_index()
    q: dict = {"term": {"owner_id": owner_id}}
    if status:
        q = {"bool": {"filter": [{"term": {"owner_id": owner_id}},
                                 {"term": {"status": status}}]}}
    r = es("POST", f"/{PREFIX}am_models/_search", {
        "size": 20,
        "query": {"bool": {"filter": [q],
                           "must": [{"match_phrase": {"question_pattern":
                                                      pattern}}]}}})
    return [h["_source"] for h in r["hits"]["hits"]
            if normalize_pattern(h["_source"]["question_pattern"])
            == normalize_pattern(pattern)]


def propose_draft(owner_id: str, question_pattern: str, summary: str,
                  source_ids: list, visibility: str = "private") -> dict:
    """Worker-proposed draft — NEVER enters the priority tier; promotion is
    an explicit upsert_model by the owner. Idempotent per (owner,
    question_pattern): an existing draft is updated in place (summary,
    source_ids, visibility) instead of creating a duplicate — the worker
    loop re-proposes every cluster each AMES_CONSOLIDATE_INTERVAL, and
    match_model() only counts active models as coverage."""
    ensure_models_index()
    doc = {
        "owner_id": owner_id, "visibility": visibility,
        "question_pattern": question_pattern, "summary": summary,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "draft", "source_ids": source_ids,
    }
    existing = _find_existing_draft(owner_id, question_pattern)
    if existing is not None:
        es("POST", f"/{PREFIX}am_models/_update/{existing['id']}?refresh=true",
           {"doc": doc})
        return {"id": existing["id"], "created": False, **doc}
    r = es("POST", f"/{PREFIX}am_models/_doc?refresh=true", doc)
    return {"id": r["_id"], "created": True, **doc}


def _find_existing_draft(owner_id: str, question_pattern: str) -> dict | None:
    """Newest draft with this exact owner + question_pattern, or None."""
    ensure_models_index()
    # question_pattern is a text field (loose matching elsewhere), so filter
    # by match_phrase then verify the exact normalized string client-side.
    r = es("POST", f"/{PREFIX}am_models/_search", {
        "size": 20,
        "query": {"bool": {"filter": [
            {"term": {"owner_id": owner_id}},
            {"term": {"status": "draft"}},
        ], "must": [{"match_phrase": {"question_pattern": question_pattern}}]}},
        "sort": [{"updated_at": {"order": "desc"}}],
    })
    for h in r["hits"]["hits"]:
        if normalize_pattern(h["_source"]["question_pattern"]) \
                == normalize_pattern(question_pattern):
            return {"id": h["_id"], **h["_source"]}
    return None


def list_drafts(owner_id: str) -> list:
    ensure_models_index()
    r = es("POST", f"/{PREFIX}am_models/_search", {
        "size": 50,
        "query": {"bool": {"filter": [{"term": {"owner_id": owner_id}},
                                      {"term": {"status": "draft"}}]}},
        "sort": [{"updated_at": {"order": "desc"}}]})
    return [{"id": h["_id"], **h["_source"]} for h in r["hits"]["hits"]]


def staleness(owner_id: str, model: dict) -> bool:
    """True when the owner has ACTIVE semantic facts newer than the model that
    share an entity with its question_pattern. Keyword match, not BM25
    min_score: BM25 scores scale with index-wide IDF, so the same fact would
    flip stale/not-stale as unrelated owners grow the index."""
    from .memory import extract_entities
    from .store import idx
    from .worker import _DRAFT_ENTITY_DENYLIST
    ents = [e for e in extract_entities(model["question_pattern"])
            if e not in _DRAFT_ENTITY_DENYLIST]
    if not ents:
        return False
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": 1,
        "query": {"bool": {
            "filter": [{"term": {"owner_id": owner_id}},
                       {"term": {"active": True}},
                       {"terms": {"entities": ents}},
                       {"range": {"occurred_at": {"gt": model["updated_at"]}}}],
        }},
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
