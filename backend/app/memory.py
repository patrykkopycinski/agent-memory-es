"""Core memory operations: retain, recall, reflect-lite, promote, consolidate."""
import datetime as _dt
import hashlib
import math
import os
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
# Max fused results from one source doc group. Default 3 (round 3). Round 2 tried 2:
# it widened a top-8 from ~4.25 to ~5.3 distinct sessions but did NOT improve answer
# accuracy on the frozen anchor (v3 77 > per_doc=3 75 > per_doc=2 73, McNemar n.s.) and
# temporal-reasoning fell monotonically as the cap tightened (0.72/0.68/0.56). Override per
# request with recall(per_doc=...) / the API `per_doc` field, or AMES_PER_DOC_BUDGET.
PER_DOC_BUDGET = int(os.environ.get("AMES_PER_DOC_BUDGET", "3"))
RECALL_FETCH_FACTOR = 4         # candidates fetched per arm = size * factor (capped)
RECALL_FETCH_CAP = 50
DEDUP_SIM_THRESHOLD = 0.92      # cosine >= this in the same visibility scope => near-dup
# NOTE (cause 1): dedup is exact-normalized-text-hash ONLY. The spec allowed the
# alternative "cosine >= 0.99 over the full text", but e5-small embeds only the head,
# so two sessions sharing an opening line score ~1.0 and the newer one would be
# dropped again — the exact failure being fixed. The threshold is kept as the
# near-duplicate LINK threshold below and referenced by tests.
DEDUP_DROP_SIM = 0.99

# --- chunking (cause 2) ---------------------------------------------------
CHUNK_TARGET_CHARS = 1800       # ~450 tokens: under e5-small's 512-token window
CHUNK_OVERLAP_CHARS = 240       # ~60 tokens of shared tail between passages
CHUNK_MIN_CHARS = 400           # shorter than this is never split

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


def _norm_text(text: str) -> str:
    """Whitespace-collapsed lowercase form used for exact-text identity."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def text_hash(text: str) -> str:
    """sha256 of the normalized text: the ONLY cheap, truncation-proof identity for
    'the same memory'. Embedding similarity is not an identity test (e5-small only
    sees the first ~512 tokens, so same-topic sessions score 0.93-0.97)."""
    return hashlib.sha256(_norm_text(text).encode("utf-8")).hexdigest()


def _parse_as_of(as_of: str) -> _dt.datetime:
    """Parse a caller-supplied 'now' (ISO-8601 date or datetime) into an aware UTC dt.

    Day precision is the useful granularity for the benchmark's question dates; a bare
    date resolves to that day's END so a same-day session is inside the window.
    """
    s = (as_of or "").strip().replace("Z", "+00:00")
    try:
        d = _dt.datetime.fromisoformat(s)
    except ValueError:
        d = _dt.datetime.strptime(s[:10], "%Y-%m-%d")
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    if len(s) == 10:
        d = d + _dt.timedelta(days=1) - _dt.timedelta(seconds=1)
    return d


def _scope_filter(owner_id: str, kind: str, visibility: str) -> dict:
    filt = [{"term": {"active": True}}, {"term": {"visibility": visibility}}]
    if visibility == "private":
        filt.append({"term": {"owner_id": owner_id}})
    return {"bool": {"filter": filt}}


def chunk_text(text: str, target: int = CHUNK_TARGET_CHARS, overlap: int = CHUNK_OVERLAP_CHARS,
               min_chars: int = CHUNK_MIN_CHARS) -> list:
    """Split a long memory into contiguous VERBATIM passages with a shared tail.

    Cause 2: a whole-session doc (median ~10.5k chars) is embedded only at its head,
    so the answer turn at char 3k-6k never entered the vector arm. Passages of
    ~450 tokens keep every turn inside the embedding window; `overlap` chars of the
    previous passage are repeated so a fact straddling a cut is still retrievable.
    Text is never rewritten: every passage is a contiguous slice of the input, cut on
    a line/sentence boundary when one falls in the last 60% of the window. Short
    texts stay a single passage, so nothing changes for them.
    """
    text = text or ""
    if len(text) <= target:
        return [text]
    spans, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + target)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("? "),
                      window.rfind("! "))
            if cut >= int(target * 0.4):
                end = start + cut + 1
        spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [text[a:b] for a, b in spans]


def _dedup_lookup(owner_id: str, kind: str, visibility: str, vec, text_hash_v=None):
    """Exact-text duplicate id, in the same kind + same visibility scope.

    Exact normalized-text hash only. Private scope is owner-scoped (a private dupe
    must not match another owner's doc); team/common scope is shared, so identical
    text collapses onto one doc. Cosine similarity is deliberately NOT an identity
    test here: near-duplicates are written and linked instead (see _nearest_dup)."""
    if not text_hash_v:
        return None
    r = es("POST", f"/{idx(kind)}/_search", {
        "size": 1,
        "query": {"bool": {
            "filter": _scope_filter(owner_id, kind, visibility)["bool"]["filter"] +
                      [{"term": {"text_hash": text_hash_v}}],
        }},
    })
    hits = r["hits"]["hits"]
    return hits[0]["_id"] if hits else None


def _nearest_dup(owner_id: str, kind: str, visibility: str, vec):
    """(id, cosine) of the nearest active doc in scope, or None. Used to LINK a
    near-duplicate write (supersedes/superseded_by), never to drop it."""
    if vec is None:
        return None
    r = es("POST", f"/{idx(kind)}/_search", {
        "size": 1,
        "knn": {"field": "embedding", "query_vector": vec, "k": 1,
                "num_candidates": 50,
                "filter": _scope_filter(owner_id, kind, visibility)["bool"]["filter"]},
    })
    hits = r["hits"]["hits"]
    if not hits:
        return None
    # ES normalises cosine knn scores to (1 + cosine) / 2; invert to real cosine.
    return hits[0]["_id"], 2.0 * hits[0]["_score"] - 1.0


def retain(owner_id: str, kind: str, text: str, visibility: str = "private",
           occurred_at: Optional[str] = None, doc_group: Optional[str] = None) -> dict:
    """Write a memory. Long text is split into verbatim passages (cause 2) that share
    a `doc_group` (the caller's stable id for the source document, e.g. a session id);
    recall returns each passage with its group + occurred_at, so callers can treat a
    group as one document. Identity/dedup is per GROUP: one exact-text-hash match
    collapses the whole write, near-duplicates are linked, never dropped (cause 1)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if visibility not in VISIBILITIES:
        raise ValueError(f"visibility must be one of {VISIBILITIES}")
    if kind == "semantic":
        rej = tombstone.is_rejected(owner_id, text)
        if rej:
            raise ValueError(f"rejected value: {rej['reason']}")
    h = text_hash(text)
    dup_id = _dedup_lookup(owner_id, kind, visibility, None, h)
    if dup_id:
        existing = es("GET", f"/{idx(kind)}/_doc/{dup_id}")["_source"]
        return {"_id": dup_id, "doc_group": existing.get("doc_group") or dup_id,
                "deduped": True, "dedup_reason": "exact_text_hash",
                **existing, "deduped_against_owner": existing.get("owner_id")}
    passages = chunk_text(text)
    group = doc_group or ("dg_%s" % uuid.uuid4().hex)
    occurred = occurred_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    vec = None
    try:
        vec = embeddings.embed([passages[0]])[0]
    except Exception:
        pass  # embeddings optional: BM25-only degrade
    near = _nearest_dup(owner_id, kind, visibility, vec)
    ids = []
    for i, passage in enumerate(passages):
        doc = {
            "kind": kind,
            "owner_id": owner_id,
            "visibility": visibility,
            "text": passage,
            "text_hash": h,          # group identity: identical re-writes collapse
            "doc_group": group,
            "passage_index": i,
            "passages_total": len(passages),
            "entities": extract_entities(passage),
            "occurred_at": occurred,
            "active": True,
        }
        if vec is not None:
            # Each passage gets its OWN embedding (that is the point of chunking);
            # reuse the already-computed head vector for passage 0.
            if i == 0:
                doc["embedding"] = vec
            else:
                try:
                    doc["embedding"] = embeddings.embed([passage])[0]
                except Exception:
                    pass
        r = es("POST", f"/{idx(kind)}/_doc?refresh=true", doc)
        ids.append(r["_id"])
    # `_id` stays a CONCRETE document id (first passage): promote/consolidate/supersede
    # and every existing caller key off a real doc id. The group is separate metadata.
    out = {"_id": ids[0], "ids": ids, "doc_group": group, "passages": len(passages),
           "deduped": False, "dedup_reason": "distinct", "owner_id": owner_id,
           "visibility": visibility, "occurred_at": occurred, "text_hash": h}
    if near and near[1] >= DEDUP_SIM_THRESHOLD:
        # Near-duplicate group (0.92 <= cosine < 0.99): WRITE the newer passages and
        # link both ends. Both stay active: this is coexistence (a later session about
        # the same topic), not a replacement. `_supersede` remains the explicit,
        # owner-checked merge path that deactivates the old doc; deactivating here
        # would re-lose the newer session, which is the entire bug being fixed.
        old_id = near[0]
        es("POST", f"/{idx(kind)}/_update/{ids[0]}?refresh=true",
           {"doc": {"supersedes": [old_id]}})
        es("POST", f"/{idx(kind)}/_update/{old_id}?refresh=true",
           {"doc": {"superseded_by": ids[0]}})
        out["dedup_reason"] = "near_duplicate_linked"
        out["supersedes"] = [old_id]
        out["cosine"] = near[1]
    return out


def _visibility_filter(owner_id: str) -> dict:
    # private docs: only creator. team/common docs shared.
    return {"bool": {"should": [
        {"term": {"owner_id": owner_id}},
        {"terms": {"visibility": ["team", "common"]}},
    ]}}


def recall(owner_id: str, query: str, kinds=None, size: int = 8,
           min_score: Optional[float] = None,
           as_of: Optional[str] = None,
           per_doc: Optional[int] = None) -> dict:
    """Hybrid recall: BM25 + kNN fused per-kind, visibility isolation, RRF across kinds.

    Returns a single flat `results` list (RRF + recency fused, score-ordered) with
    `kind`, `score` and `visibility` on every item. `by_kind` is a deprecated additive
    field kept for compat. `min_score` post-filters the fused list and marks an empty
    result `abstained: true`.

    cause 3: `as_of` (ISO-8601 date/datetime) is the instant the query is asked. Both
    the recency decay and the relative time-window parser resolve against it instead of
    the wall clock, so a benchmark replaying old sessions does not rank by "recent
    relative to today" nor resolve "last two weeks" against the wrong year.

    Round 2: `per_doc` caps how many passages of ONE source doc_group may enter the
    window (default PER_DOC_BUDGET=3). Passages are ranked first, then collapsed to at
    most `per_doc` per group BEFORE the size cut, so the window covers more distinct
    sessions. `doc_groups` in the response reports how many distinct groups it holds.
    """
    now = None
    if as_of:
        now = _parse_as_of(as_of)
    kinds = kinds or list(KINDS)
    qvec = None
    try:
        qvec = embeddings.embed([query])[0]
    except Exception:
        pass
    results = {}
    # Over-fetch: the per-doc collapse runs AFTER ranking and can only shrink the list,
    # so each arm must return more than `size` candidates or a size-8 window would come
    # back under-filled whenever a group hits its cap.
    fetch = max(size, min(RECALL_FETCH_CAP, size * RECALL_FETCH_FACTOR))
    for kind in kinds:
        must = [{"multi_match": {"query": query, "fields": ["text^3", "entities"], "type": "most_fields"}}]
        if qvec is not None:
            body = {
                "size": fetch,
                "retriever": {"rrf": {
                    "retrievers": [
                        {"standard": {"query": {"bool": {
                            "must": must,
                            "filter": [{"term": {"active": True}}, _visibility_filter(owner_id)],
                        }}}},
                        {"knn": {"field": "embedding", "query_vector": qvec, "num_candidates": max(100, fetch * 2), "k": fetch,
                                 "filter": [{"term": {"active": True}}, _visibility_filter(owner_id)]}},
                    ],
                    "rank_constant": 60,
                    "rank_window_size": 50,
                }},
            }
        else:
            body = {
                "size": fetch,
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
        w = _temporal.parse_window(query, now)
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
    _now = now or _dt.datetime.now(_dt.timezone.utc)
    for h in fused.values():
        h.setdefault("es_score", h.get("score"))
        h["recency"] = _recency_boost(h.get("occurred_at"), _now)
        h["score"] = h["rrf"] + h["recency"]
    ordered = sorted(fused.values(), key=lambda x: -x["score"])
    if min_score is not None:
        ordered = [h for h in ordered if h["score"] >= min_score]
    window = _apply_budgets(ordered, size, per_doc_budget=per_doc or PER_DOC_BUDGET)
    out = {
        "query": query,
        "results": window,
        "by_kind": {k: v[:size] for k, v in results.items()},   # deprecated; compat shape
        "fused": window,      # deprecated alias of `results`
        "abstained": bool(min_score is not None and not window),
        "doc_groups": len({it.get("doc_group") or it.get("id") for it in window}),
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
