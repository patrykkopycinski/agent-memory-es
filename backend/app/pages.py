"""Knowledge pages: living documents the bank writes about itself.

A page is scoped by an entity/topic (e.g. 'omniroute', 'evals'), synthesized from
that owner's ACTIVE semantic facts mentioning the scope, never from another page
(no self-citation feedback loop). Stored in am_pages, rewritten incrementally by
the consolidation worker. Text-synthesis is deterministic (grouped fact listing);
an LLM rewrite hook is optional.
"""
import hashlib
from typing import Optional

from .store import es, ensure_indices

PAGE_MAPPINGS = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "owner_id": {"type": "keyword"},
            "visibility": {"type": "keyword"},
            "scope": {"type": "keyword"},          # topic key, e.g. "omniroute"
            "title": {"type": "text"},
            "body": {"type": "text"},               # markdown
            "source_ids": {"type": "keyword"},      # evidence: memory ids (never page ids)
            "fact_count": {"type": "integer"},
            "updated_at": {"type": "date"},
        },
    },
}


def page_id(owner_id: str, scope: str) -> str:
    return hashlib.sha1(f"{owner_id}/{scope}".encode()).hexdigest()


def ensure_pages_index() -> None:
    try:
        es("GET", "/am_pages")
    except RuntimeError:
        es("PUT", "/am_pages", PAGE_MAPPINGS)


def refresh_page(owner_id: str, scope: str, facts: list, visibility: str = "private") -> dict:
    """Rewrite one page from its facts. `facts` = [{'id','text','occurred_at'}, ...]
    Deterministic body: bullets in chronological order. Returns page doc."""
    ensure_pages_index()
    body_lines = [f"# {scope.replace('_',' ').title()}", ""]
    for f in sorted(facts, key=lambda x: x.get("occurred_at") or ""):
        body_lines.append(f"- {f['text']}")
    doc = {
        "owner_id": owner_id,
        "visibility": visibility,
        "scope": scope,
        "title": scope.replace("_", " ").title(),
        "body": "\n".join(body_lines),
        "source_ids": [f["id"] for f in facts],
        "fact_count": len(facts),
        "updated_at": _now(),
    }
    es("PUT", f"/am_pages/_doc/{page_id(owner_id, scope)}?refresh=true", doc)
    return {"id": page_id(owner_id, scope), **doc}


def get_page(owner_id: str, scope: str) -> Optional[dict]:
    try:
        r = es("GET", f"/am_pages/_doc/{page_id(owner_id, scope)}")
        return {"id": r["_id"], **r["_source"]}
    except RuntimeError:
        return None


def list_pages(owner_id: str, include_shared: bool = True) -> list:
    ensure_pages_index()
    filt = [{"term": {"owner_id": owner_id}}]
    if include_shared:
        filt = [{"bool": {"should": [
            {"term": {"owner_id": owner_id}},
            {"bool": {"must_not": {"term": {"visibility": "private"}}}},
        ]}}]
    r = es("POST", "/am_pages/_search", {"size": 100, "query": {"bool": {"filter": filt}},
                                         "sort": [{"updated_at": {"order": "desc"}}]})
    return [{"id": h["_id"], **h["_source"]} for h in r["hits"]["hits"]]


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
