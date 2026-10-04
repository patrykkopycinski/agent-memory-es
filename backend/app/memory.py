"""Core memory operations: retain, recall, reflect-lite, promote, consolidate."""
import datetime as _dt
import math
import re
import time
import uuid
from typing import Any, Optional

from .store import KINDS, es, idx, ensure_indices
from . import embeddings
from . import tombstone

VISIBILITIES = ("private", "team", "common")

# --- recall quality knobs (see docs/quality_decisions) ---------------------
RANK_CONSTANT = 60              # RRF rank constant; must match ES rrf rank_constant
RECENCY_WEIGHT = 0.15           # recency arm weight, as a fraction of one RRF arm's top hit
RECENCY_HALF_LIFE_DAYS = 30.0   # gauss decay half-life applied to occurred_at
PER_DOC_BUDGET = 3              # max fused results from one source doc group
DEDUP_SIM_THRESHOLD = 0.92      # cosine >= this in the same visibility scope => duplicate

import re as _re
_PRIVATE_MARKERS = _re.compile(
    r"(?i)\b(api[_ -]?key|token|password|passwd|secret|credential|ssh[_ -]?key|"
    r"private[_ -]?key|admin[_ -]?token)\b"
)


def guard_promotion(text: str, to_visibility: str) -> tuple:
    """Deterministic write-guard (APPA-style): same text, same decision, no classifier.
    Blocks promotion of private memories carrying sensitive markers to shared visibility."""
    if to_visibility in ("team", "common") and _PRIVATE_MARKERS.search(text):
        return False, "sensitive marker present in private memory; refusing shared promotion"
    return True, "ok"

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


def _parse_ts(value: Optional[str]):
    """Parse AMES occurred_at strings (ISO datetime or date). None when unparseable."""
    if not value:
        return None
    raw = value.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(raw[:19], fmt).replace(tzinfo=_dt.timezone.utc)
        except Exception:
            continue
    try:
        return _dt.datetime.fromisoformat(raw)
    except Exception:
        return None


def recency_decay(occurred_at: Optional[str], now=None,
                  half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """Gaussian recency decay in [0, 1]: 1.0 now, 0.5 at half_life_days, 0.0 when missing."""
    ts = _parse_ts(occurred_at)
    if ts is None:
        return 0.0
    now = now or _dt.datetime.now(_dt.timezone.utc)
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    sigma = half_life_days / math.sqrt(2.0 * math.log(2.0))
    return math.exp(-0.5 * (age_days / sigma) ** 2)


def _recency_boost(occurred_at: Optional[str], now=None) -> float:
    """Recency contribution to the fused score, scaled to one RRF arm's top hit
    (1/(RANK_CONSTANT+1)) so the small weight never outvotes lexical/kNN agreement."""
    return RECENCY_WEIGHT * recency_decay(occurred_at, now) / (RANK_CONSTANT + 1)


def _apply_budgets(ordered: list, size: int, per_kind_budget: Optional[int] = None,
                   per_doc_budget: int = PER_DOC_BUDGET) -> list:
    """Cap how much of the fused window any one kind or source doc can occupy.

    Pass 1 admits at most per_kind_budget per kind and per_doc_budget per doc group
    (doc_group defaults to the doc id), so one kind or one chunked/long retain cannot
    fill the window. Pass 2 backfills any leftover slots with the best remaining hits
    still respecting the per-doc cap, so a query whose matches live in one kind is not
    starved. Returns the (score-ordered) window.
    """
    if per_kind_budget is None:
        n = max(1, len({it.get("kind") for it in ordered}) or 1)
        per_kind_budget = size if n == 1 else max(1, -(-size // n))
    picked, kind_count, doc_count = [], {}, {}
    overflow = []
    for it in ordered:
        kind = it.get("kind")
        doc = it.get("doc_group") or it.get("id")
        if kind_count.get(kind, 0) < per_kind_budget and doc_count.get(doc, 0) < per_doc_budget:
            picked.append(it)
            kind_count[kind] = kind_count.get(kind, 0) + 1
            doc_count[doc] = doc_count.get(doc, 0) + 1
        else:
            overflow.append(it)
    for it in overflow:
        if len(picked) >= size:
            break
        doc = it.get("doc_group") or it.get("id")
        if doc_count.get(doc, 0) < per_doc_budget:
            picked.append(it)
            doc_count[doc] = doc_count.get(doc, 0) + 1
    picked.sort(key=lambda x: -x.get("score", 0.0))
    return picked[:size]


def _dedup_lookup(owner_id: str, kind: str, visibility: str, vec) -> Optional[str]:
    """Existing doc id whose embedding is a near-duplicate (cosine >= DEDUP_SIM_THRESHOLD)
    of vec, in the same kind + same visibility scope. None when no vector or no match.

    Private scope is owner-scoped (a private dupe must not match another owner's doc);
    team/common scope is shared, so cross-owner duplicates collapse onto one doc."""
    if vec is None:
        return None
    filt = [{"term": {"active": True}}, {"term": {"visibility": visibility}}]
    if visibility == "private":
        filt.append({"term": {"owner_id": owner_id}})
    r = es("POST", f"/{idx(kind)}/_search", {
        "size": 1,
        "knn": {"field": "embedding", "query_vector": vec, "k": 1,
                "num_candidates": 50, "filter": filt},
    })
    hits = r["hits"]["hits"]
    if not hits:
        return None
    # ES normalises cosine knn scores to (1 + cosine) / 2; invert to real cosine.
    cosine = 2.0 * hits[0]["_score"] - 1.0
    return hits[0]["_id"] if cosine >= DEDUP_SIM_THRESHOLD else None


def retain(owner_id: str, kind: str, text: str, visibility: str = "private",
           occurred_at: Optional[str] = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if visibility not in VISIBILITIES:
        raise ValueError(f"visibility must be one of {VISIBILITIES}")
    if kind == "semantic":
        rej = tombstone.is_rejected(owner_id, text)
        if rej:
            raise ValueError(f"rejected value: {rej['reason']}")
    vec = None
    try:
        vec = embeddings.embed([text])[0]
    except Exception:
        pass  # embeddings optional: BM25-only degrade
    dup_id = _dedup_lookup(owner_id, kind, visibility, vec)
    if dup_id:
        existing = es("GET", f"/{idx(kind)}/_doc/{dup_id}")["_source"]
        return {"_id": dup_id, "deduped": True, **existing,
                "deduped_against_owner": existing.get("owner_id")}
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
    doc["deduped"] = False
    return doc


def _visibility_filter(owner_id: str) -> dict:
    # private docs: only creator. team/common docs shared.
    return {"bool": {"should": [
        {"term": {"owner_id": owner_id}},
        {"terms": {"visibility": ["team", "common"]}},
    ]}}


def recall(owner_id: str, query: str, kinds=None, size: int = 8,
           min_score: Optional[float] = None) -> dict:
    """Hybrid recall: BM25 + kNN fused per-kind, visibility isolation, RRF across kinds.

    Returns a single flat `results` list (RRF + recency fused, score-ordered) with
    `kind`, `score` and `visibility` on every item. `by_kind` is a deprecated additive
    field kept for compat. `min_score` post-filters the fused list and marks an empty
    result `abstained: true`."""
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
            {"id": h["_id"], "kind": kind, "score": h["_score"],
             **{k: v for k, v in h["_source"].items()
                if k not in ("kind", "embedding")}}
            for h in r["hits"]["hits"]
        ]
    # rank-fusion across kinds (simple RRF, k=RANK_CONSTANT)
    fused = {}
    for kind, hits in results.items():
        for rank, h in enumerate(hits):
            fused.setdefault(h["id"], {**h, "rrf": 0.0})
            fused[h["id"]]["rrf"] += 1.0 / (RANK_CONSTANT + rank + 1)
    GRAPH_RRF_WEIGHT = 0.01  # hop-2 evidence must not outvote lexical+kNN agreement
    # graph arm: entity two-hop expansion, fused by rank (never dominates other arms)
    try:
        from . import graph as _graph
        for rank, h in enumerate(_graph.graph_expand(owner_id, query, size,
                                                     _visibility_filter(owner_id))):
            fused.setdefault(h["id"], {**h, "rrf": 0.0, "arm": "graph"})
            fused[h["id"]]["rrf"] += GRAPH_RRF_WEIGHT / (RANK_CONSTANT + rank + 1)
            fused[h["id"]]["graph_hit"] = True
    except Exception:
        pass
    # temporal arm: window fill spread across the range
    temporal_meta = None
    try:
        from . import temporal as _temporal
        w = _temporal.parse_window(query)
        if w:
            start, end = w
            buckets = _temporal.spread_buckets(start, end)
            per_bucket = max(1, size // len(buckets))
            t_hits = []
            for bs, be in buckets:
                rb = es("POST", f"/{idx('episodic')}/_search", {
                    "size": per_bucket,
                    "query": {"bool": {
                        "must": [{"match_all": {}}],
                        "filter": [{"term": {"active": True}},
                                   _visibility_filter(owner_id),
                                   {"range": {"occurred_at": {"gte": bs, "lte": be}}}],
                    }}, "sort": [{"occurred_at": {"order": "asc"}}]})
                t_hits += [{"id": h["_id"], "kind": "episodic", "score": h["_score"],
                            "arm": "temporal",
                            **{k: v for k, v in h["_source"].items()
                               if k not in ("kind", "embedding")}}
                           for h in rb["hits"]["hits"]]
            for rank, h in enumerate(t_hits):
                fused.setdefault(h["id"], {**h, "rrf": 0.0})
                fused[h["id"]]["rrf"] += 1.0 / (RANK_CONSTANT + rank + 1)
                fused[h["id"]]["temporal_hit"] = True
            temporal_meta = {"window": [start, end], "buckets": len(buckets)}
    except Exception:
        pass
    # recency arm (always on, small weight): gauss decay over occurred_at, scaled to
    # one RRF arm's top hit so it reorders near-ties without outvoting relevance.
    _now = _dt.datetime.now(_dt.timezone.utc)
    for h in fused.values():
        h.setdefault("es_score", h.get("score"))
        h["recency"] = _recency_boost(h.get("occurred_at"), _now)
        h["score"] = h["rrf"] + h["recency"]
    ordered = sorted(fused.values(), key=lambda x: -x["score"])
    if min_score is not None:
        ordered = [h for h in ordered if h["score"] >= min_score]
    window = _apply_budgets(ordered, size)
    out = {
        "query": query,
        "results": window,
        "by_kind": results,   # deprecated: per-kind lists, kept for compat
        "fused": window,      # deprecated alias of `results`
        "abstained": bool(min_score is not None and not window),
    }
    # mental-model priority tier surfaces above facts (like reflect's source order)
    try:
        from . import mental_models as _mm
        m = _mm.match_model(owner_id, query)
        if m:
            out["mental_model"] = m
    except Exception:
        pass
    if temporal_meta:
        out["temporal"] = temporal_meta
    return out


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
    allowed, reason = guard_promotion(s["text"], to_visibility)
    if not allowed:
        raise ValueError(reason)
    copy = {k: v for k, v in s.items() if k not in ("superseded_by", "supersedes")}
    copy["visibility"] = to_visibility
    copy["promoted_from"] = doc_id
    r = es("POST", f"/{idx(kind)}/_doc?refresh=true", copy)
    return {"promoted_id": r["_id"], "visibility": to_visibility}


def _supersede(owner_id: str, old_id: str, new_id: str, kind: str = "semantic") -> None:
    """Owner-checked supersede: both ends must belong to owner_id (atlas: Atlas's
    supersede update is keyed on id alone — we refuse that class of cross-tenant write)."""
    old = es("GET", f"/{idx(kind)}/_doc/{old_id}")["_source"]
    new = es("GET", f"/{idx(kind)}/_doc/{new_id}")["_source"]
    if old["owner_id"] != owner_id or new["owner_id"] != owner_id:
        raise PermissionError("supersede requires both memories to belong to the caller")
    es("POST", f"/{idx(kind)}/_update/{old_id}?refresh=true",
       {"doc": {"active": False, "superseded_by": new_id}})


def consolidate(owner_id: str, kind: str = "semantic", similarity_threshold: float = 0.75,
                dry_run: bool = True) -> dict:
    """Dedup/supersede proposal pass. dry_run=True (default): report pairs + dispositions,
    mutate nothing. dry_run=False: apply supersessions (owner-checked, history preserved).
    disposition 'keep_both' = similarity is coexistence, not contradiction (different times,
    different contexts) — surfaced, never auto-merged."""
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

    def sim(a, b):
        ea, eb = set(a.get("entities", [])), set(b.get("entities", []))
        j = len(ea & eb) / len(ea | eb) if ea | eb else 0.0
        ta = set(a["text"].lower().split())
        tb = set(b["text"].lower().split())
        jt = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
        return 0.5 * j + 0.5 * jt

    proposals, superseded = [], 0
    for i, (id_a, a) in enumerate(docs):
        if not a.get("active", True):
            continue
        for id_b, b in docs[i + 1:]:
            if not b.get("active", True):
                continue
            s = sim(a, b)
            if s >= similarity_threshold:
                disposition = "supersede" if s >= similarity_threshold + 0.1 else "keep_both"
                proposals.append({"older": id_a, "newer": id_b, "similarity": round(s, 3),
                                  "disposition": disposition})
                if not dry_run and disposition == "supersede":
                    _supersede(owner_id, id_a, id_b, kind)
                    superseded += 1
                break
    return {"checked": len(docs), "mutated": 0 if dry_run else superseded,
            "superseded": superseded, "proposals": proposals}


def reflect(owner_id: str, question: str, llm_answer_fn=None, rounds: int = 3) -> dict:
    """Answer from memory via a bounded agentic loop: recall (with mental-model
    priority tier), optionally rewrite the query and recall again when evidence is
    thin, synthesize if an LLM fn is provided. Cites only retrieved ids.
    Abstains when no evidence (atlas: irrelevant memory bends the answer)."""
    from . import mental_models as _mm
    model = _mm.match_model(owner_id, question)
    evidence = recall(owner_id, question)
    sources = [h for h in evidence["fused"][:5]]
    queries_run = [question]
    # multi-round: bounded query rewrites extend evidence whenever the LLM is
    # reachable (Hindsight runs its loop unconditionally; thin-evidence gating
    # never fires on a corpus with shared memories). llm_answer_fn=False means
    # "don't synthesize", not "no LLM at all" — rewrites still run.
    if rounds > 1:
        from . import llm as _llm
        try:
            rewrites = _llm.rewrite_queries(question, rounds - 1)
            for q in rewrites[:rounds - 1]:
                queries_run.append(q)
                more = recall(owner_id, q)
                seen = {s["id"] for s in sources}
                for h in more["fused"]:
                    if h["id"] not in seen and len(sources) < 5:
                        sources.append(h)
        except Exception:
            pass  # rewrite unavailable → single-pass reflect, never fake rounds
    if not sources and model is None:
        return {"question": question, "answer": "INSUFFICIENT_EVIDENCE", "sources": [],
                "synthesized": False, "queries": queries_run}
    if model is not None:
        sources = [{"id": model["id"], "text": model["summary"],
                    "tier": "mental_model", **{k: v for k, v in model.items()
                                              if k in ("owner_id", "visibility", "updated_at")}}] + sources
    if llm_answer_fn is False:
        return {"question": question, "sources": [s["id"] for s in sources],
                "evidence": sources, "synthesized": False, "queries": queries_run}
    if llm_answer_fn is None:
        from . import llm as _llm
        fn = _llm.chat
    else:
        fn = llm_answer_fn
    answer = fn(question, sources)
    cited = [s["id"] for s in sources if s["id"] in (answer or "")]
    return {"question": question, "answer": answer,
            "sources": cited or [s["id"] for s in sources],
            "synthesized": True, "queries": queries_run}
