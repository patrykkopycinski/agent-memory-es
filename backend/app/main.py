"""FastAPI app: REST surface + MCP-style tools endpoint + auth middleware."""
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import auth, memory, pages as pages_mod, mental_models as mm_mod, reranker
from .store import KINDS, ensure_indices

app = FastAPI(title="agent-memory-es", version="0.2.0")


@app.on_event("startup")
def _startup():
    ensure_indices()
    pages_mod.ensure_pages_index()
    mm_mod.ensure_models_index()


def caller(x_api_key: Optional[str] = Header(None)) -> dict:
    who = auth.owner_of(x_api_key or "")
    if not who:
        raise HTTPException(401, "invalid or missing X-API-Key")
    return who


class RetainIn(BaseModel):
    kind: str
    text: str
    visibility: str = "private"
    occurred_at: Optional[str] = None


class RecallIn(BaseModel):
    query: str
    kinds: Optional[list] = None
    size: int = 8
    min_score: Optional[float] = None


class PromoteIn(BaseModel):
    kind: str
    doc_id: str
    to_visibility: str


class ReflectIn(BaseModel):
    question: str


@app.post("/admin/keys")
def mint_key(owner_id: str, role: str = "member", _admin: str = Header(None, alias="X-Admin-Token")):
    if _admin != os.environ.get("AMES_ADMIN_TOKEN", "dev-admin"):
        raise HTTPException(403, "bad admin token")
    return {"api_key": auth.create_key(owner_id, role), "owner_id": owner_id}


@app.post("/memory/retain")
def retain(body: RetainIn, who: dict = Depends(caller)):
    try:
        return memory.retain(who["owner_id"], body.kind, body.text, body.visibility, body.occurred_at)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/memory/recall")
def recall(body: RecallIn, who: dict = Depends(caller)):
    return memory.recall(who["owner_id"], body.query, body.kinds, body.size, body.min_score)


@app.post("/memory/promote")
def promote(body: PromoteIn, who: dict = Depends(caller)):
    try:
        return memory.promote(who["owner_id"], body.kind, body.doc_id, body.to_visibility)
    except (ValueError, PermissionError) as e:
        raise HTTPException(422 if isinstance(e, ValueError) else 403, str(e))


@app.post("/memory/consolidate")
def consolidate(who: dict = Depends(caller)):
    return memory.consolidate(who["owner_id"])


@app.post("/memory/reflect")
def reflect(body: ReflectIn, who: dict = Depends(caller)):
    return memory.reflect(who["owner_id"], body.question)


class PageIn(BaseModel):
    scope: str


@app.post("/memory/pages/refresh")
def page_refresh(body: PageIn, who: dict = Depends(caller)):
    """Regenerate one knowledge page from the owner's active semantic facts in scope."""
    r = memory.recall(who["owner_id"], body.scope, kinds=["semantic"], size=50)
    facts = [{"id": h["id"], "text": h["text"], "occurred_at": h.get("occurred_at")}
             for h in r["by_kind"].get("semantic", [])]
    if not facts:
        raise HTTPException(422, "no facts in scope")
    return pages_mod.refresh_page(who["owner_id"], body.scope, facts)


@app.get("/memory/pages")
def page_list(who: dict = Depends(caller)):
    return {"pages": pages_mod.list_pages(who["owner_id"])}


@app.get("/memory/pages/{scope}")
def page_get(scope: str, who: dict = Depends(caller)):
    p = pages_mod.get_page(who["owner_id"], scope)
    if not p:
        raise HTTPException(404, "no page for scope")
    return p


class ModelIn(BaseModel):
    question_pattern: str
    summary: str
    visibility: str = "private"


@app.post("/memory/models")
def model_upsert(body: ModelIn, who: dict = Depends(caller)):
    return mm_mod.upsert_model(who["owner_id"], body.question_pattern, body.summary,
                               body.visibility)


@app.get("/memory/stats")
def memory_stats(who: dict = Depends(caller)):
    """Read-only inspector feed for the desktop plugin/UI: per-index counts the
    owner can see, active mental models (with staleness flags), worker drafts,
    and knowledge pages. No mutations."""
    from .store import es, idx, PREFIX
    owner = who["owner_id"]
    counts = {}
    for kind in ("episodic", "semantic", "procedural"):
        r = es("POST", f"/{idx(kind)}/_search", {
            "size": 0,
            "query": {"bool": {"should": [
                {"term": {"owner_id": owner}},
                {"bool": {"must_not": {"term": {"visibility": "private"}}}},
            ]}},
            "aggs": {"by_vis": {"terms": {"field": "visibility", "size": 5}}}})
        counts[kind] = {
            "total": r["hits"]["total"]["value"],
            "by_visibility": {b["key"]: b["doc_count"]
                              for b in r["aggregations"]["by_vis"]["buckets"]},
        }
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": 5, "sort": [{"occurred_at": {"order": "desc"}}],
        "query": {"bool": {"should": [
            {"term": {"owner_id": owner}},
            {"bool": {"must_not": {"term": {"visibility": "private"}}}},
        ]}}})
    recent = [{"id": h["_id"], "text": h["_source"]["text"][:140],
               "visibility": h["_source"]["visibility"],
               "occurred_at": h["_source"].get("occurred_at")}
              for h in r["hits"]["hits"]]

    drafts = mm_mod.list_drafts(owner)
    models = mm_mod.list_models(owner, "active", 50)
    for m in models:
        m["stale"] = mm_mod.staleness(owner, m)
    return {"owner_id": owner, "counts": counts, "recent_semantic": recent,
            "models": models, "drafts": drafts,
            "pages": pages_mod.list_pages(owner)}


class RerankIn(BaseModel):
    query: str
    hits: list
    top_n: int = 5


@app.post("/memory/rerank")
def rerank_ep(body: RerankIn, who: dict = Depends(caller)):
    out = reranker.rerank(body.query, body.hits, body.top_n)
    if not out["reranked"]:
        return {**out, "note": "rerank unavailable (license/model); identity order returned"}
    return out


@app.post("/mcp/tools/{tool}")
def mcp_tool(tool: str, body: dict, who: dict = Depends(caller)):
    """Minimal MCP-style tools surface (JSON in/out) for coding-farm clients."""
    handlers = {
        "retain": lambda b: memory.retain(who["owner_id"], b["kind"], b["text"], b.get("visibility", "private"), b.get("occurred_at")),
        "recall": lambda b: memory.recall(who["owner_id"], b["query"], b.get("kinds"), b.get("size", 8)),
        "reflect": lambda b: memory.reflect(who["owner_id"], b["question"]),
        "promote": lambda b: memory.promote(who["owner_id"], b["kind"], b["doc_id"], b["to_visibility"]),
    }
    if tool not in handlers:
        raise HTTPException(404, f"unknown tool {tool}; available: {sorted(handlers)}")
    try:
        return {"result": handlers[tool](body)}
    except (ValueError, PermissionError) as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {"ok": True, "kinds": list(KINDS)}
