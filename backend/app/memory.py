"""Core memory operations: retain, recall, reflect-lite, promote, consolidate."""
import re
import time
import uuid
from typing import Any, Optional

from .store import KINDS, es, idx, ensure_indices
from . import embeddings

VISIBILITIES = ("private", "team", "common")

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.+-]{2,}")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "what", "which", "who", "whom", "how", "when", "where", "why", "did",
    "does", "do", "has", "have", "had", "user", "uses", "using", "use",
}


def extract_entities(text: str, limit: int = 12) -> list:
    seen, out = set(), []
    for tok in TOKEN_RE.findall(text):
        t = tok.lower()
        if t not in STOPWORDS and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def retain(owner_id: str, kind: str, text: str, visibility: str = "private",
           occurred_at: Optional[str] = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if visibility not in VISIBILITIES:
        raise ValueError(f"visibility must be one of {VISIBILITIES}")
    vec = None
    try:
        vec = embeddings.embed([text])[0]
    except Exception:
        pass  # embeddings optional: BM25-only degrade
    doc = {
        "kind": kind,
        "owner_id": owner_id,
        "visibility": visibility,
        "text": text,
        "entities": extract_entities(text),
        "occurred_at": occurred_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "active": True,
    }
    if vec is not None:
        doc["embedding"] = vec
    r = es("POST", f"/{idx(kind)}/_doc?refresh=true", doc)
    doc["_id"] = r["_id"]
    return doc


def _visibility_filter(owner_id: str) -> dict:
    # private docs: only creator. team/common docs shared.
    return {"bool": {"should": [
        {"term": {"owner_id": owner_id}},
        {"terms": {"visibility": ["team", "common"]}},
    ]}}


def recall(owner_id: str, query: str, kinds=None, size: int = 8) -> dict:
    """Hybrid recall: BM25 + kNN fused per-kind, visibility isolation, RRF across kinds."""
    kinds = kinds or list(KINDS)
    qvec = None
    try:
        qvec = embeddings.embed([query])[0]
    except Exception:
        pass
    results = {}
    for kind in kinds:
        must = [{"multi_match": {"query": query, "fields": ["text^3", "entities"], "type": "most_fields"}}]
        if qvec is not None:
            body = {
                "size": size,
                "retriever": {"rrf": {
                    "retrievers": [
                        {"standard": {"query": {"bool": {
                            "must": must,
                            "filter": [{"term": {"active": True}}, _visibility_filter(owner_id)],
                        }}}},
                        {"knn": {"field": "embedding", "query_vector": qvec, "num_candidates": 100, "k": size,
                                 "filter": [{"term": {"active": True}}, _visibility_filter(owner_id)]}},
                    ],
                    "rank_constant": 60,
                    "rank_window_size": 50,
                }},
            }
        else:
            body = {
                "size": size,
                "query": {"bool": {
                    "must": must,
                    "filter": [{"term": {"active": True}}, _visibility_filter(owner_id)],
                }},
                "sort": [{"_score": {"order": "desc"}}, {"occurred_at": {"order": "desc"}}],
            }
        r = es("POST", f"/{idx(kind)}/_search", body)
        results[kind] = [
            {"id": h["_id"], "score": h["_score"], **{k: v for k, v in h["_source"].items()
                                                      if k not in ("kind", "embedding")}}
            for h in r["hits"]["hits"]
        ]
    # rank-fusion across kinds (simple RRF, k=60)
    fused = {}
    for kind, hits in results.items():
        for rank, h in enumerate(hits):
            fused.setdefault(h["id"], {**h, "rrf": 0.0})
            fused[h["id"]]["rrf"] += 1.0 / (60 + rank + 1)
    ordered = sorted(fused.values(), key=lambda x: -x["rrf"])
    return {"query": query, "by_kind": results, "fused": ordered[:size]}


def promote(owner_id: str, kind: str, doc_id: str, to_visibility: str) -> dict:
    """Copy-on-promote: private -> team/common creates a promoted copy; original stays private."""
    if to_visibility not in ("team", "common"):
        raise ValueError("promotion target must be team or common")
    src = es("GET", f"/{idx(kind)}/_doc/{doc_id}")
    s = src["_source"]
    if s["owner_id"] != owner_id:
        raise PermissionError("only the creator can promote their memory")
    if s["visibility"] != "private":
        raise ValueError("only private memories can be promoted")
    copy = {k: v for k, v in s.items() if k not in ("superseded_by", "supersedes")}
    copy["visibility"] = to_visibility
    copy["promoted_from"] = doc_id
    r = es("POST", f"/{idx(kind)}/_doc?refresh=true", copy)
    return {"promoted_id": r["_id"], "visibility": to_visibility}


def consolidate(owner_id: str, kind: str = "semantic", similarity_threshold: float = 0.75) -> dict:
    """Dedup/supersede: for each active semantic doc, find highly similar newer active doc
    of same owner+visibility; mark older superseded (history preserved via active=False + chain)."""
    assert kind == "semantic", "consolidation currently targets semantic facts"
    body = {
        "size": 200,
        "query": {"bool": {"filter": [
            {"term": {"active": True}}, {"term": {"owner_id": owner_id}},
        ]}},
        "sort": [{"occurred_at": {"order": "asc"}}],
    }
    r = es("POST", f"/{idx(kind)}/_search", body)
    docs = [(h["_id"], h["_source"]) for h in r["hits"]["hits"]]
    superseded = 0
    for i, (id_a, a) in enumerate(docs):
        if not a.get("active", True):
            continue
        for id_b, b in docs[i + 1:]:
            if not b.get("active", True):
                continue
            ea, eb = set(a.get("entities", [])), set(b.get("entities", []))
            j = len(ea & eb) / len(ea | eb) if ea | eb else 0.0
            ta = set(a["text"].lower().split())
            tb = set(b["text"].lower().split())
            jt = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
            if 0.5 * j + 0.5 * jt >= similarity_threshold:
                es("POST", f"/{idx(kind)}/_update/{id_a}?refresh=true",
                   {"doc": {"active": False, "superseded_by": id_b}})
                superseded += 1
                break
    return {"checked": len(docs), "superseded": superseded}


def reflect(owner_id: str, question: str, llm_answer_fn=None) -> dict:
    """Answer from memory: gather evidence via recall, synthesize if an LLM fn is provided,
    else return the distilled evidence bundle with per-source attribution."""
    evidence = recall(owner_id, question)
    sources = [h for h in evidence["fused"][:5]]
    if llm_answer_fn is None:
        return {
            "question": question,
            "answer": " | ".join(s["text"] for s in sources[:3]) or "(no evidence found)",
            "sources": [s["id"] for s in sources],
            "synthesized": False,
        }
    answer = llm_answer_fn(question, [s["text"] for s in sources])
    return {"question": question, "answer": answer, "sources": [s["id"] for s in sources], "synthesized": True}
