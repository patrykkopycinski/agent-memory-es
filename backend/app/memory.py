"""Core memory operations: retain, recall, reflect-lite, promote, consolidate."""
import datetime as _dt
import hashlib
import math
import json
import logging
import os
import re
import time
import urllib.parse
import uuid
from typing import Any, Optional

from .store import KINDS, es, idx, is_doc_miss, ensure_indices, ensure_indices_once, knn_num_candidates
from . import embeddings
from . import secrets_guard
from . import tagfilter as _tf
from . import tombstone
from . import governance

_recall_log = logging.getLogger("ames.recall")

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

# The kNN arm always returns `k` hits, so a nonsense query still gets a rank-1 hit in every
# kind and the cross-kind RRF hands each kind's #1 the same 1/(RANK_CONSTANT+1). An optional
# absolute cosine floor drops hits below it. Default OFF: E5-family cosines are compressed into
# a narrow band, so a floor must be calibrated against an eval run before it is enabled.
def _parse_similarity(raw):
    if raw is None or raw == "":
        return None
    # bool is an int subclass: float(True) == 1.0 would silently become a maximal floor
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"min_similarity must be a number, got {raw!r}")
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"min_similarity must be a number, got {raw!r}") from None
    if not -1.0 <= v <= 1.0:
        raise ValueError(f"min_similarity must be within [-1, 1], got {v}")
    return v


KNN_MIN_SIMILARITY = _parse_similarity(os.environ.get("AMES_KNN_MIN_SIMILARITY"))

# `text` uses the standard analyzer (no stopword filter), so "what is the capital of france"
# matches any long English doc on the/of/is, and cross-kind fusion hands that hit a top slot
# the kNN floor cannot remove. Opt-in: strip Lucene's English stop set (ES `_english_`) from
# the BM25 `text` clause only; `entities` keeps the raw query (exact keyword match). Default
# off until an anchor run measures it. An all-stopword query leaves the lexical arm empty.
LEXICAL_STOPWORDS = os.environ.get("AMES_LEXICAL_STOPWORDS", "").lower() in ("1", "true")
_EN_STOP = ("a an and are as at be but by for if in into is it no not of on or such that the "
            "their then there these they this to was will with").split()
_STOP_RE = re.compile(r"(?<![\w'])(?:%s)(?![\w'])" % "|".join(_EN_STOP), re.IGNORECASE)


def _lexical_clause(query: str) -> dict:
    if not LEXICAL_STOPWORDS:
        return {"multi_match": {"query": query, "fields": ["text^3", "entities"], "type": "most_fields"}}
    content = " ".join(_STOP_RE.sub(" ", query).split())
    return {"bool": {"should": [{"match": {"text": {"query": content, "boost": 3}}},
                                {"match": {"entities": query}}],
                     "minimum_should_match": 1}}

# Round 5: ES-native cross-encoder rerank (_inference/rerank) over the fused top-RERANK_DEPTH,
# applied BEFORE the per-doc budget and the size cut (the output budget is unchanged: still
# `size` passages). Default on; AMES_RERANK=0 or per-request rerank=false disables it. If the
# endpoint is unavailable (or slower than AMES_RERANK_TIMEOUT) the fused order is kept and the
# response carries rerank={requested, applied:false, fallback_reason, depth, ms}.
RERANK_DEPTH_CAP = 50           # hard ceiling: the cross-encoder cost is linear in depth
RERANK_DEPTH = min(int(os.environ.get("AMES_RERANK_DEPTH", "50")), RERANK_DEPTH_CAP)
RERANK_DEFAULT = os.environ.get("AMES_RERANK", "0").strip().lower() in ("1", "true", "yes", "on")


def _clamp_depth(depth) -> int:
    """Per-request depth override -> [2, RERANK_DEPTH_CAP]; absent/garbage -> the configured default."""
    try:
        d = int(depth)
    except (TypeError, ValueError):
        return RERANK_DEPTH
    return max(2, min(d, RERANK_DEPTH_CAP))


def _rerank_fused(query: str, ordered: list, depth: int = None) -> tuple:
    """Back-compat wrapper over _rerank_fused_meta: (new_ordered, reranked: bool)."""
    new, info = _rerank_fused_meta(query, ordered, depth)
    return new, info["applied"]


def _rerank_fused_meta(query: str, ordered: list, depth: int = None) -> tuple:
    """Re-order the top `depth` of the fused list by cross-encoder relevance. Returns
    (new_ordered, reranked: bool). `score` is NOT touched (it stays rrf + recency, so the
    documented invariant and `min_score` semantics hold); the cross-encoder verdict is exposed as
    `rerank_score` and the final position as `rerank_rank`, which _apply_budgets sorts by.
    Candidates the endpoint did not return, and everything beyond `depth`, keep their relative
    fused order strictly below the reranked ones.

    Returns (new_ordered, info) with info = {requested, applied, fallback_reason, depth, ms}:
    a rerank that did not happen says so and why (timeout | unavailable | error |
    too_few_candidates) instead of silently serving the fused order."""
    from . import reranker as _rr
    depth = RERANK_DEPTH if depth is None else _clamp_depth(depth)
    head, tail = ordered[:depth], ordered[depth:]
    info = {"requested": True, "applied": False, "fallback_reason": None,
            "depth": len(head), "ms": 0}
    if len(head) < 2:
        info["fallback_reason"] = "too_few_candidates"
        return ordered, info
    out = _rr.rerank(query, head, top_n=len(head))
    info["ms"] = int(out.get("ms") or 0)
    if not out.get("reranked"):
        info["fallback_reason"] = out.get("fallback_reason") or "error"
        return ordered, info
    scored = {h.get("id") for h in out["hits"] if h.get("id") is not None}
    ranked = list(out["hits"])
    ranked += [h for h in head if h.get("id") not in scored]    # never lose a candidate
    new = []
    for i, h in enumerate(ranked + tail):
        h = dict(h)
        h["rerank_rank"] = i
        if h.get("id") not in scored:
            # an item the endpoint did not return must not carry a score it never received
            # (covers both an omitted head item and every item beyond `depth`)
            h.pop("rerank_score", None)
        new.append(h)
    info["applied"] = True
    return new, info
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
# Request bounds. Each passage is one embed call plus one refresh=true write, so
# unbounded text is unbounded ES/embedding work per request. 100k chars is ~65
# passages; the live bank's largest doc is 3 passages.
MAX_TEXT_CHARS = int(os.environ.get("AMES_MAX_TEXT_CHARS", "100000"))
MAX_PASSAGES = 100              # group lookups read at most this many passages
MAX_RECALL_SIZE = 50
MAX_PER_DOC = 50
MAX_ID_CHARS = 512            # caller-supplied ids (unreject source_id); no earlier bound existed, 512 = ES's _id limit

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

    def doc_key(it):
        # groups are per owner: another owner's same-named group must not share the cap
        return (it.get("owner_id"), it.get("doc_group") or it.get("id"))

    for it in ordered:
        kind = it.get("kind")
        doc = doc_key(it)
        if kind_count.get(kind, 0) < per_kind_budget and doc_count.get(doc, 0) < per_doc_budget:
            picked.append(it)
            kind_count[kind] = kind_count.get(kind, 0) + 1
            doc_count[doc] = doc_count.get(doc, 0) + 1
        else:
            overflow.append(it)
    for it in overflow:
        if len(picked) >= size:
            break
        doc = doc_key(it)
        if doc_count.get(doc, 0) < per_doc_budget:
            picked.append(it)
            doc_count[doc] = doc_count.get(doc, 0) + 1
    # reranked lists carry rerank_rank (cross-encoder order); otherwise score-ordered
    picked.sort(key=lambda x: (x["rerank_rank"],) if "rerank_rank" in x else (-x.get("score", 0.0),))
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
    flt = _scope_filter(owner_id, kind, visibility)["bool"]["filter"]
    if kind == "semantic" and visibility == "team":
        # governed decisions are never linked as near-duplicates (either side of the pair)
        flt = {"bool": {"filter": flt, "must_not": governance.exclusion_clauses()}}
    r = es("POST", f"/{idx(kind)}/_search", {
        "size": 1,
        "knn": {"field": "embedding", "query_vector": vec, "k": 1,
                "num_candidates": knn_num_candidates(1), "filter": flt},
    })
    hits = r["hits"]["hits"]
    if not hits:
        return None
    # ES normalises cosine knn scores to (1 + cosine) / 2; invert to real cosine.
    return hits[0]["_id"], 2.0 * hits[0]["_score"] - 1.0


# Soft length hint: durable facts are short and self-contained. Never rejects the write.
RETAIN_SOFT_MAX_CHARS = 1500


def _reject_secret(text, tags) -> None:
    """ValueError (HTTP 422) if the text or any raw tag looks like a credential. The message names
    the pattern, never the match. Must run before any write or delete."""
    hit = secrets_guard.find_secret(text)
    if hit:
        raise ValueError(f"text looks like a secret ({hit}); remove the credential and retry")
    hit = secrets_guard.find_secret_in_tags(tags)
    if hit:
        raise ValueError(f"a tag looks like a secret ({hit}); remove the credential and retry")


def retain(owner_id: str, kind: str, text: str, visibility: str = "private",
           occurred_at: Optional[str] = None, doc_group: Optional[str] = None,
           tags: Optional[list] = None, labels: Optional[list] = None,
           extract_facts: bool = True, supersedes_decision: Optional[str] = None,
           person: Optional[str] = None) -> dict:
    """Write a memory, refusing text that looks like a credential (ValueError -> HTTP 422) and
    flagging over-long text with a `too_long` hint on the result (the write still happens)."""
    _reject_secret(text, tags)
    out = _retain(owner_id, kind, text, visibility, occurred_at, doc_group, tags, labels,
                  extract_facts, supersedes_decision, person=person)
    if isinstance(text, str) and len(text) > RETAIN_SOFT_MAX_CHARS:
        out["too_long"] = True
        out["too_long_hint"] = (f"text is {len(text)} chars (> {RETAIN_SOFT_MAX_CHARS}); "
                                "durable facts should be short and self-contained")
    return out


def _retain(owner_id: str, kind: str, text: str, visibility: str = "private",
            occurred_at: Optional[str] = None, doc_group: Optional[str] = None,
            tags: Optional[list] = None, labels: Optional[list] = None,
            extract_facts: bool = True, supersedes_decision: Optional[str] = None,
            _proposal: Optional[dict] = None, person: Optional[str] = None,
            _group: Optional[str] = None) -> dict:
    """Write a memory. Long text is split into verbatim passages (cause 2) that share
    a `doc_group` (the caller's stable id for the source document, e.g. a session id);
    recall returns each passage with its group + occurred_at, so callers can treat a
    group as one document. Identity/dedup is per GROUP: one exact-text-hash match
    collapses the whole write, near-duplicates are linked, never dropped (cause 1)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if visibility not in VISIBILITIES:
        raise ValueError(f"visibility must be one of {VISIBILITIES}")
    _check_text_size(text)
    ensure_indices_once()
    # Tag filtering (decision record): both inputs are optional; absent => nothing below runs.
    write_tags = _tf.norm_tags(tags)              # FilterError (a ValueError) -> HTTP 422
    label_groups = _tf.validate_labels(labels)
    if kind == "semantic":
        rej = tombstone.is_rejected(owner_id, text)
        if rej and rej.get("capacity"):
            raise ValueError(
                f"text has more than {tombstone.TERMS_CHUNK} key terms and owner has more than "
                f"{tombstone.MAX_PAGES * tombstone.PAGE} candidate tombstones; shorten the text")
        if rej:
            raise ValueError(f"rejected value: {rej['reason']}")
    h = text_hash(text)
    dup_id = _dedup_lookup(owner_id, kind, visibility, None, h)
    if dup_id:
        existing = es("GET", f"/{idx(kind)}/_doc/{dup_id}")["_source"]
        out = {"_id": dup_id, "doc_group": existing.get("doc_group") or dup_id,
               "deduped": True, "dedup_reason": "exact_text_hash",
               **existing, "deduped_against_owner": existing.get("owner_id")}
        if governance.is_governed(existing):
            # identical text of a decision already in the pipeline: never a second proposal
            out["decision_status"] = existing.get("decision_status")
            out["approvals_needed"] = governance.approvals_needed(existing)
        if kind == "episodic" and extract_facts:
            from . import facts
            out["fact_jobs"] = []
            try:
                out["fact_jobs"].append(facts.enqueue(owner_id, dup_id))
            except Exception as exc:
                out["fact_enqueue_errors"] = [{"source_id": dup_id, "error": str(exc)[:200]}]
                out["fact_extraction_complete"] = False
        return out
    extraction = None
    if label_groups:
        # A3: after the exact-duplicate check, so a duplicate costs no LLM call. Fail closed:
        # ExtractionError propagates (HTTP 502) BEFORE anything is written.
        extracted, extraction = _tf.extract_tags(text, label_groups)
        for t in extracted:
            if t not in write_tags:
                write_tags.append(t)
        if len(write_tags) > _tf.MAX_TAGS_PER_WRITE:
            write_tags = write_tags[:_tf.MAX_TAGS_PER_WRITE]
            extraction["capped"] = True
    governed = governance.is_governed_write(kind, visibility, write_tags)
    if supersedes_decision is not None and not governed:
        raise ValueError("supersedes_decision applies only to a team semantic memory tagged "
                         f"{governance.DECISION_TAG}")
    override_of = None
    if governed and _proposal is None:
        # fail closed BEFORE any write: a bad override target must leave nothing behind
        override_of = (governance.validate_override_target(supersedes_decision)
                       if supersedes_decision is not None else None)
    passages = chunk_text(text)
    # A governed write never takes the caller's doc_group: a group is only a label, and status
    # fields are mirrored per group, so a caller-chosen name could pull unrelated docs in.
    # `_group` is the edit path re-using the proposal's own server-generated group.
    group = (_group or "dg_%s" % uuid.uuid4().hex) if governed else (
        doc_group or ("dg_%s" % uuid.uuid4().hex))
    author = person or owner_id
    occurred = occurred_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Embeddings stay optional (BM25-only degrade) but the degrade is reported, never silent.
    vecs, embed_error = embeddings.embed_safe([passages[0]])
    vec = vecs[0] if vecs else None
    near = None if governed else _nearest_dup(owner_id, kind, visibility, vec)
    gov_fields = None
    if governed:
        if _proposal is None:
            gov_fields = governance.proposal_fields(author, override_of)
            governance.audit_propose(author, group, h, override_of)   # BEFORE the first write
        else:
            gov_fields = dict(_proposal)
    ids = []
    unembedded = 1 if vec is None else 0   # passages written without a vector
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
        if kind == "episodic" and extract_facts:
            doc["fact_requested"] = True
        if write_tags:
            doc[_tf.TAG_FIELD] = list(write_tags)
        if gov_fields:
            doc.update(gov_fields)
        if vec is not None:
            # Each passage gets its OWN embedding (that is the point of chunking);
            # reuse the already-computed head vector for passage 0.
            if i == 0:
                doc["embedding"] = vec
            else:
                pvecs, perr = embeddings.embed_safe([passage])
                if pvecs:
                    doc["embedding"] = pvecs[0]
                else:
                    embed_error = embed_error or perr
                    unembedded += 1
        elif i > 0:
            unembedded += 1
        r = es("POST", f"/{idx(kind)}/_doc?refresh=true", doc)
        ids.append(r["_id"])
    # `_id` stays a CONCRETE document id (first passage): promote/consolidate/supersede
    # and every existing caller key off a real doc id. The group is separate metadata.
    out = {"_id": ids[0], "ids": ids, "doc_group": group, "passages": len(passages),
           "deduped": False, "dedup_reason": "distinct", "owner_id": owner_id,
           "visibility": visibility, "occurred_at": occurred, "text_hash": h,
           "embedded": unembedded == 0}
    if unembedded:
        out["embed_error"] = embed_error
    if write_tags:
        out["tags"] = list(write_tags)
    if gov_fields:
        out.update(governance.propose_result(gov_fields))
    if extraction is not None:
        out["extraction"] = extraction
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
    if kind == "episodic" and extract_facts:
        from . import facts
        out["fact_jobs"] = []
        for source_id in ids[:1]:
            try:
                out["fact_jobs"].append(facts.enqueue(owner_id, source_id))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).exception("fact enqueue failed for %s", source_id)
                out.setdefault("fact_enqueue_errors", []).append({"source_id": source_id, "error": str(exc)[:200]})
                out["fact_extraction_complete"] = False
    return out


def _q(doc_id: str) -> str:
    """Percent-encode an id for an ES path segment: a raw '?' or '/' in a
    caller-supplied id would otherwise add query params or path segments."""
    return urllib.parse.quote(str(doc_id), safe="")


def _check_text_size(text) -> None:
    if text is not None and len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text too long: {len(text)} chars (max {MAX_TEXT_CHARS})")


def _get_source(kind: str, doc_id: str) -> dict:
    """Fetch _source or raise LookupError (HTTP 404 at the edge), instead of the
    generic RuntimeError store.es raises on any non-2xx."""
    try:
        return es("GET", f"/{idx(kind)}/_doc/{_q(doc_id)}")["_source"]
    except RuntimeError as e:
        if is_doc_miss(e):          # an index-level 404 is a broken store, not "no such doc"
            raise LookupError(f"no such {kind} doc: {doc_id}") from e
        raise


class SupersededError(Exception):
    """update/delete addressed a memory that has been replaced. HTTP 409 with `live_id`.

    We refuse rather than silently redirect to the successor: a client that holds a stale id
    asked to change/delete *that* text, and applying a destructive delete (or a text overwrite)
    to a different document it never named is the worse failure. 409 + live_id lets the caller
    re-issue against the successor deliberately. `live_id` is None when the chain ends in a doc
    that no longer exists."""

    def __init__(self, doc_id: str, live_id: Optional[str]):
        super().__init__(f"memory {doc_id} was superseded; live memory: {live_id}")
        self.doc_id, self.live_id = doc_id, live_id


def _is_superseded(s: dict) -> bool:
    # retain's near-duplicate link sets superseded_by on a doc that stays active (coexistence);
    # only a doc taken out of service (active:false via _supersede, or valid_to via facts) is stale.
    return bool(s.get("superseded_by")) and (s.get("active") is False or bool(s.get("valid_to")))


MAX_SUPERSEDE_HOPS = 10


def _live_successor(kind: str, s: dict) -> Optional[str]:
    """Follow superseded_by to the first doc still in service. The hop bound is the only guard:
    a cycle or an over-long chain exhausts it and yields None."""
    nxt = s.get("superseded_by")
    for _ in range(MAX_SUPERSEDE_HOPS):
        if not nxt:
            return None
        try:
            cur = _get_source(kind, nxt)
        except LookupError:
            return None
        if not _is_superseded(cur):
            return nxt
        nxt = cur.get("superseded_by")
    return None


def _owned_group_query(owner_id: str, group: str) -> dict:
    # doc_group is a caller-chosen string (often a session id), so two owners can
    # share one. It is not a tenant boundary: every group-level read or delete must
    # also match the owner. Refusing shared names instead would let anyone squat a
    # predictable group and block the real owner's writes.
    return {"bool": {"filter": [{"term": {"doc_group": group}},
                                {"term": {"owner_id": owner_id}}]}}


def _group_hits(owner_id: str, kind: str, doc_id: str) -> list:
    """(id, text) for the doc and every passage of the caller's doc_group, in
    passage order. doc_id may be any passage of the group (retain returns the
    first). PermissionError when it belongs to another owner."""
    s = _get_source(kind, doc_id)
    if s["owner_id"] != owner_id:
        raise PermissionError("only the creator can modify their memory")
    group = s.get("doc_group")
    if not group:
        return [(doc_id, s.get("text", ""))]
    r = es("POST", f"/{idx(kind)}/_search", {
        "size": MAX_PASSAGES,
        "_source": ["text"],
        "query": _owned_group_query(owner_id, group),
        "sort": [{"passage_index": {"order": "asc"}}],
    })
    hits = [(h["_id"], h["_source"].get("text", "")) for h in r["hits"]["hits"]]
    return hits or [(doc_id, s.get("text", ""))]


def _assert_group_mutable(owner_id: str, kind: str, s: dict) -> None:
    """update / delete / promote act on the whole group, so they must clear every member, not
    only the addressed doc: a decision in force (or closed) cannot be changed through a sibling,
    and an open proposal cannot be reached through a doc that is not itself governed."""
    group = s.get("doc_group")
    if not group:
        return
    r = es("POST", f"/{idx(kind)}/_search", {
        "size": MAX_PASSAGES, "_source": ["decision_status"],
        "query": {"bool": {"filter": _owned_group_query(owner_id, group)["bool"]["filter"] + [
            {"exists": {"field": "decision_status"}}]}}})
    for h in r["hits"]["hits"]:
        if not governance.is_governed(s):
            raise governance.GovernedError()
        governance.check_mutable(h["_source"])


def _group_docs(owner_id: str, kind: str, doc_id: str) -> list:
    return [i for i, _ in _group_hits(owner_id, kind, doc_id)]


def _log_mutation(intent: str, owner_id: str, kind: str, doc_id: str, detail: dict,
                  strict: bool = False) -> Optional[str]:
    """Best-effort intent log (who/what/when) written BEFORE the mutation runs.
    A logging failure never blocks the mutation, except with strict=True (decision governance:
    a transition that cannot be audited must not happen)."""
    from .store import PREFIX
    doc = {"intent": intent, "owner_id": owner_id, "kind": kind, "doc_id": doc_id,
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "detail": detail}
    try:
        es("PUT", f"/{PREFIX}am_mutations", {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {
                "intent": {"type": "keyword"}, "owner_id": {"type": "keyword"},
                "kind": {"type": "keyword"}, "doc_id": {"type": "keyword"},
                "at": {"type": "date"},
            }},
        })
    except RuntimeError:
        pass  # already exists
    # Additive mapping migration (same pattern as store.ensure_indices): indices
    # created before doc_id was pinned as keyword would otherwise text-map it.
    try:
        es("PUT", f"/{PREFIX}am_mutations/_mapping", {"properties": {
            "intent": {"type": "keyword"}, "owner_id": {"type": "keyword"},
            "kind": {"type": "keyword"}, "doc_id": {"type": "keyword"},
            "at": {"type": "date"},
        }})
    except RuntimeError:
        pass
    try:
        return es("POST", f"/{PREFIX}am_mutations/_doc?refresh=true", doc).get("_id")
    except RuntimeError:
        if strict:
            raise
    return None


def update(owner_id: str, kind: str, doc_id: str, text=None, tags=None,
           visibility=None, occurred_at=None, person: Optional[str] = None) -> dict:
    """Fix a memory in place: replace text (re-chunked, re-embedded) and/or
    tags/visibility/occurred_at across the whole doc_group. The group id is
    stable. Owner-scoped (same check as promote). The new text goes through
    retain's full path — chunking, per-passage embeddings, tombstone check —
    pinned to the SAME doc_group and occurred_at, so the edit is a true
    in-place fix, not a near-duplicate."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    s = _get_source(kind, doc_id)
    if s["owner_id"] != owner_id:
        raise PermissionError("only the creator can modify their memory")
    if _is_superseded(s):
        raise SupersededError(doc_id, _live_successor(kind, s))
    governance.check_mutable(s)         # active / closed decisions: 409, the creator included
    _assert_group_mutable(owner_id, kind, s)
    author = person or owner_id
    if governance.is_governed(s) and s.get("proposed_by") not in (None, author):
        raise PermissionError("only the creator can modify their memory")
    if visibility is not None and visibility not in VISIBILITIES:
        raise ValueError(f"visibility must be one of {VISIBILITIES}")
    _check_text_size(text)
    _reject_secret(text, tags)  # before _log_mutation and the delete: a 422 must change nothing
    proposal = None
    if governance.is_governed(s):
        # an open proposal edited by its author: stays a proposal, approvals reset below
        if (visibility not in (None, "team") or
                (tags is not None and governance.DECISION_TAG not in _tf.norm_tags(tags))):
            raise ValueError("a proposed decision stays a team decision; delete it to withdraw")
        proposal = {k: s[k] for k in ("decision_status", "proposed_by", "proposal_for") if k in s}
        proposal["approvals"] = []
        if s.get("proposal_expires_at"):
            proposal["proposal_expires_at"] = s["proposal_expires_at"]   # an edit never extends the TTL
        if s.get("objections"):
            proposal["objections"] = s["objections"]
    elif kind == "semantic" and (visibility == "team" or (
            tags is not None and s["visibility"] == "team")):
        # no side door around the proposal flow: update cannot make an existing doc a decision
        will_have = _tf.norm_tags(tags) if tags is not None else (s.get(_tf.TAG_FIELD) or [])
        if governance.DECISION_TAG in will_have and (
                s["visibility"] != "team" or governance.DECISION_TAG not in (s.get(_tf.TAG_FIELD) or [])):
            raise ValueError("a decision cannot be created through update; retain it as a team "
                             "memory tagged " + governance.DECISION_TAG + " to propose it")
    group = s.get("doc_group")
    occurred = s.get("occurred_at")
    old_tags = s.get(_tf.TAG_FIELD)
    old_text = s["text"] if text is not None else None
    hits = _group_hits(owner_id, kind, doc_id)
    if proposal is not None:
        # only the passages of THIS decision: a stray doc sharing the group name is not part of it
        mine = es("POST", f"/{idx(kind)}/_search", {
            "size": MAX_PASSAGES, "_source": False,
            "query": {"bool": {"filter": governance._scope(s)}}})
        mine_ids = {h["_id"] for h in mine["hits"]["hits"]}
        hits = [(i, t) for i, t in hits if i in mine_ids] or hits
    old_ids = [i for i, _ in hits] if text is not None else []
    new_tags = _tf.norm_tags(tags) if tags is not None else None
    target_vis = visibility or s["visibility"]
    if s["visibility"] == "private" and target_vis != "private":
        # Same write-guard as promote(): widening through update must not be a side
        # door for marker-bearing text. Checked against the whole group, not one passage.
        allowed, reason = guard_promotion(
            text if text is not None else "\n".join(t for _, t in hits), target_vis)
        if not allowed:
            raise ValueError(reason)
    if text is not None and text_hash(text) == text_hash(old_text) and new_tags is None \
            and visibility in (None, s["visibility"]):
        return {"_id": doc_id, "doc_group": group or doc_id, "updated": False,
                "note": "no-op: identical normalized text, tags and visibility"}
    _log_mutation("update", owner_id, kind, doc_id,
                  {"old_text": old_text, "new_text": text, "old_ids": old_ids,
                   "old_tags": old_tags, "tags": tags, "visibility": visibility},
                  strict=proposal is not None)
    if text is not None:
        if proposal is not None:
            # write the new passages first, delete the old ones only once the new text is
            # stored: a collision or a failure partway must leave the original proposal intact
            _reject_secret(text, new_tags)
            r = _rewrite_proposal(owner_id, kind, s, text, target_vis, occurred,
                                  new_tags if new_tags is not None else old_tags,
                                  proposal, group, old_ids)
        elif len(old_ids) == 1:
            es("DELETE", f"/{idx(kind)}/_doc/{_q(old_ids[0])}?refresh=true")
        else:
            es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
               {"query": _owned_group_query(owner_id, group)})
        if proposal is None:
            r = retain(owner_id, kind, text, target_vis,
                       occurred_at=occurred, doc_group=group, tags=new_tags)
        out = {"_id": r["_id"], "doc_group": r.get("doc_group", group),
               "passages": r.get("passages"), "updated": True}
        if "embedded" in r:
            out["embedded"] = r["embedded"]
        if r.get("embed_error"):
            out["embed_error"] = r["embed_error"]
        return out
    patch = {}
    if new_tags is not None:
        patch[_tf.TAG_FIELD] = list(new_tags)
    if visibility is not None:
        patch["visibility"] = visibility
    if occurred_at is not None:
        patch["occurred_at"] = occurred_at
    if patch:
        for i, _ in hits:
            es("POST", f"/{idx(kind)}/_update/{_q(i)}?refresh=true", {"doc": patch})
    return {"_id": doc_id, "doc_group": group or doc_id,
            "passages": len(old_ids), "updated": bool(patch)}


def _rewrite_proposal(owner_id, kind, s, text, visibility, occurred, tags, proposal, group,
                      old_ids) -> dict:
    """Replace the text of an open proposal without ever losing it: the new passages are written
    first (same server-generated group, new text hash), the old ones deleted only afterwards.
    A collision with another doc raises before anything is written; a failure partway removes the
    partial new passages and leaves the original untouched."""
    new_hash = text_hash(text)
    if new_hash == s.get("text_hash"):
        # same text, new tags: nothing to rewrite, the approvals stay bound to this text
        for i in old_ids:
            es("POST", f"/{idx(kind)}/_update/{_q(i)}?refresh=true",
               {"doc": {_tf.TAG_FIELD: list(tags or [])}})
        return {"_id": old_ids[0], "doc_group": group, "passages": len(old_ids)}
    try:
        r = _retain(owner_id, kind, text, visibility, occurred_at=occurred, tags=tags,
                    _proposal=proposal, _group=group)
    except Exception:
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true&conflicts=proceed",
           {"query": {"bool": {"filter": governance._scope(
               {"doc_group": group, "owner_id": owner_id, "text_hash": new_hash})}}})
        raise
    if r.get("deduped"):
        raise ValueError("an identical memory already exists; the proposal was left unchanged")
    for old in old_ids:
        es("DELETE", f"/{idx(kind)}/_doc/{_q(old)}?refresh=true")
    return r


def delete(owner_id: str, kind: str, doc_id: str, retract: bool = False,
           person: Optional[str] = None) -> dict:
    """Delete a memory (the whole doc_group) by id. Owner-scoped: another owner's
    doc is a PermissionError (HTTP 403), matching promote. Intent (who/when/what)
    is logged to am_mutations BEFORE the delete runs. A plain delete only removes
    (duplicate, stale wording, test write): the text can be retained again. With
    retract=True the semantic text is also tombstoned ("wrong, never record it
    again": retain() consults is_rejected for semantic kind only, so other kinds
    write no tombstone). tombstone.lift / unreject() undoes a retraction."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not isinstance(retract, bool):
        raise ValueError("retract must be a boolean")
    s = _get_source(kind, doc_id)
    if s["owner_id"] != owner_id:
        raise PermissionError("only the creator can delete their memory")
    if _is_superseded(s):
        raise SupersededError(doc_id, _live_successor(kind, s))
    _assert_group_mutable(owner_id, kind, s)
    if governance.is_governed(s):
        if s.get("decision_status") == governance.PROPOSED:
            return governance.withdraw(person or owner_id, doc_id)   # author's delete = withdrawn
        governance.check_mutable(s)
    group = s.get("doc_group")
    _log_mutation("delete", owner_id, kind, doc_id,
                  {"text": s["text"], "doc_group": group,
                   "visibility": s.get("visibility"), "retract": retract})
    ids = _group_docs(owner_id, kind, doc_id)
    if len(ids) == 1:
        es("DELETE", f"/{idx(kind)}/_doc/{_q(ids[0])}?refresh=true")
    else:
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": _owned_group_query(owner_id, group)})
    tombstoned = retract and kind == "semantic" and tombstone.reject(
        owner_id, s.get("text", ""), reason="deleted by owner", source_id=doc_id) is not None
    # Unified return shape: REST contract (test_delete.py, a96df78) pins
    # deleted=bool + group_deleted + deleted_ids; the POST suite uses
    # deleted_ids for membership. ids enumerated via _group_docs (passage order).
    return {"deleted": True, "doc_group": group or doc_id,
            "kind": kind, "doc_id": doc_id,
            "group_deleted": bool(ids), "deleted_ids": ids,
            "tombstoned": tombstoned}


def unreject(owner_id: str, text: Optional[str] = None, source_id: Optional[str] = None) -> dict:
    """Lift this owner's tombstones so the value can be retained again: every tombstone that
    would block `text` (the same rule as retain's is_rejected check), or those recorded for
    `source_id` (the doc id a retracting delete came from). Exactly one selector. Intent is
    logged to am_mutations BEFORE the lift, with the lifted tombstones' source_ids; lifting
    nothing is a success with lifted=0."""
    if (text is None) == (source_id is None):
        raise ValueError("unreject needs exactly one of text or source_id")
    if text is not None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        _check_text_size(text)
        found = tombstone.blockers(owner_id, text)
    else:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if len(source_id) > MAX_ID_CHARS:
            raise ValueError(f"source_id too long: {len(source_id)} chars (max {MAX_ID_CHARS})")
        found = tombstone.by_source(owner_id, source_id)
    # value_key stands in for tombstones written without a source_id
    lifted_ids = [t.get("source_id") or t["value_key"] for t in found]
    detail = {"by": "text" if text is not None else "source_id", "matched": len(found),
              "source_ids": lifted_ids}
    if text is not None:
        detail["text"] = text
    _log_mutation("unreject", owner_id, "semantic", source_id or "", detail)
    return {"lifted": tombstone.lift(owner_id, [t["_id"] for t in found]),
            "source_ids": lifted_ids, "by": detail["by"]}


MAX_EXISTS_TAGS = 16


def exists(owner_id: str, kind: Optional[str] = None, tags: Optional[list] = None,
           text: Optional[str] = None, size: int = 5) -> dict:
    """Cheap "do I already hold this?" lookup: exact normalized-text hash and/or tags (all must
    match). Term filters only: no embedding, no LLM, no kNN. Scope = this owner's docs plus
    team/common, active only (same visibility rule as recall). `kind` omitted searches all kinds."""
    if text is None and not tags:
        raise ValueError("exists needs text and/or tags")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("text must be a non-empty string")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    _check_text_size(text)
    size = max(1, min(int(size), 20))
    ensure_indices_once()
    flt = [{"term": {"active": True}}, _visibility_filter(owner_id)]
    if text is not None:
        flt.append({"term": {"text_hash": text_hash(text)}})
    if tags:
        norm = _tf.norm_tags(tags)
        if len(norm) > MAX_EXISTS_TAGS:
            raise ValueError(f"at most {MAX_EXISTS_TAGS} tags")
        flt += [{"term": {_tf.TAG_FIELD: t}} for t in norm]
    matches, total = [], 0
    for k in ([kind] if kind else list(KINDS)):
        r = es("POST", f"/{idx(k)}/_search", {
            "size": size, "track_total_hits": True, "_source": ["text", "doc_group", "tags", "occurred_at", "visibility"],
            "query": {"bool": {"filter": flt}}})
        total += r["hits"]["total"]["value"]
        for h in r["hits"]["hits"]:
            src = h["_source"]
            matches.append({"id": h["_id"], "kind": k, "doc_group": src.get("doc_group"),
                            "tags": src.get("tags", []), "occurred_at": src.get("occurred_at"),
                            "visibility": src.get("visibility"), "text": (src.get("text") or "")[:200]})
    return {"exists": total > 0, "count": total, "matches": matches[:size]}


def _visibility_filter(owner_id: str) -> dict:
    # private docs: only creator. team/common docs shared.
    return {"bool": {"should": [
        {"term": {"owner_id": owner_id}},
        {"terms": {"visibility": ["team", "common"]}},
    ]}}


def _abstained(query: str, parsed: dict, resolved: dict, vocab_truncated: bool) -> dict:
    """A filter whose narrow_any resolved to nothing: answer without touching ES at all, so there
    is no code path that could fall back to an unfiltered search."""
    out = {"query": query, "results": [], "by_kind": {}, "fused": [], "abstained": True,
           "abstain_hint": True, "doc_groups": 0, "reranked": False,
           "filter": {"applied": True, "resolved": resolved}}
    if vocab_truncated:
        out["filter"]["vocab_truncated"] = True
    return out


def _tag_vocabulary(owner_id: str, kinds: list, keys: list) -> tuple:
    """Distinct stored tags for `keys`, in this owner's visible scope. -> (tags, truncated)."""
    out, truncated = set(), False
    for kind in kinds:
        body = {"size": 0,
                "query": {"bool": {"filter": [{"term": {"active": True}}, _visibility_filter(owner_id)]}},
                "aggs": {"t": {"terms": {"field": _tf.TAG_FIELD, "size": _tf.FUZZY_VOCAB_CAP,
                                         **({"include": "(" + "|".join(re.escape(k) for k in keys) + "):.*"}
                                            if keys else {})}}}}
        r = es("POST", f"/{idx(kind)}/_search", body)
        agg = r.get("aggregations", {}).get("t", {})
        out.update(b["key"] for b in agg.get("buckets", []))
        if agg.get("sum_other_doc_count", 0) > 0:
            truncated = True
    return out, truncated


RRF_RANK_WINDOW = 50            # per-arm window, same as the ES rrf retriever's rank_window_size


def fusion_mode() -> str:
    """Where BM25 + kNN are fused. `app` (default): two plain searches fused in Python, which
    works on any license (retriever.rrf is paid-tier and 403s once the trial lapses).
    `es`: the single retriever.rrf search. Read per call so it can be flipped without a restart."""
    return "es_rrf" if os.environ.get("AMES_FUSION", "app").strip().lower() == "es" else "app_rrf"


def rrf_fuse(ranked_lists: list, rank_constant: int = RANK_CONSTANT,
             window: int = RRF_RANK_WINDOW) -> list:
    """Reciprocal rank fusion with ES `retriever.rrf` semantics: each input is a best-first list
    of ids; only its top `window` entries count; a doc scores sum(1 / (rank_constant + rank))
    over the lists it appears in, rank being 1-based. Returns [(id, score)] best-first. Ties
    break on the best rank in any list, then on the earlier list, then id. The id step is stable
    only within one index: ES assigns random ids, so it is not a reproducible order across runs."""
    score, best, first_list = {}, {}, {}
    for li, ids in enumerate(ranked_lists):
        for rank, doc_id in enumerate(ids[:window], start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (rank_constant + rank)
            if rank < best.get(doc_id, window + 1):
                best[doc_id] = rank
            first_list.setdefault(doc_id, li)
    order = sorted(score, key=lambda d: (-score[d], best[d], first_list[d], d))
    return [(d, score[d]) for d in order]


def _hybrid_app_fusion(kind: str, lexical_query: dict, knn: dict, fetch: int) -> tuple:
    """BM25 search + kNN search (identical filters), fused here with RRF instead of by
    `retriever.rrf`. Returns (hits, es_took) shaped like an ES hit list whose `_score` is the
    RRF score, best-first, at most `fetch` long."""
    no_vec = {"excludes": ["embedding"]}
    lex = es("POST", f"/{idx(kind)}/_search",
             {"size": RRF_RANK_WINDOW, "query": lexical_query, "_source": no_vec})
    vec = es("POST", f"/{idx(kind)}/_search", {"size": fetch, "knn": knn, "_source": no_vec})
    by_id = {}
    # Keep each arm's raw score on the hit: recall() turns them into `confidence`, and the RRF
    # score that replaces `_score` below carries no absolute relevance.
    for h in vec["hits"]["hits"]:
        by_id[h["_id"]] = {**h, "_cosine": 2.0 * h["_score"] - 1.0}   # ES cosine knn: (1+cos)/2
    for h in lex["hits"]["hits"]:                          # lexical wins on _source: identical docs
        by_id[h["_id"]] = {**h, "_bm25": h["_score"],
                           **({"_cosine": by_id[h["_id"]]["_cosine"]} if h["_id"] in by_id else {})}
    fused = rrf_fuse([[h["_id"] for h in lex["hits"]["hits"]],
                      [h["_id"] for h in vec["hits"]["hits"]]])
    hits = [{**by_id[d], "_score": sc} for d, sc in fused[:fetch]]
    return hits, (lex.get("took") or 0) + (vec.get("took") or 0)


# --- recall confidence ---------------------------------------------------------------------
# RRF scores are rank-only (a nonsense query still gets a 1/61 top hit), so they cannot tell a
# caller "nothing relevant here". `confidence` (0..1) is built from the two raw arm scores:
#   vector: E5 cosines live in a narrow band, so rescale [CONF_COS_LO, CONF_COS_HI] -> [0, 1]
#   lexical: BM25 is unbounded and corpus-dependent, so saturate it: s / (s + CONF_BM25_K)
# and blended CONF_VEC_WEIGHT : (1 - CONF_VEC_WEIGHT). Constants are calibrated by the labelled
# probe in tests/test_recall_contract.py (AUC), not guessed.
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


CONF_COS_LO = _env_float("AMES_CONF_COS_LO", 0.80)
CONF_COS_HI = _env_float("AMES_CONF_COS_HI", 0.90)
CONF_BM25_K = _env_float("AMES_CONF_BM25_K", 10.0)
CONF_VEC_WEIGHT = _env_float("AMES_CONF_VEC_WEIGHT", 0.6)
# 0.35 is deliberately conservative: it never hides an answerable query (0/24 false abstains across
# two independent probe sets, both fusion modes) but catches only ~50-60% of hard negatives
# (on-topic, answer not stored). Raising it trades answerable recall for them (see the sweep in
# the P2 review notes), so deployments that prefer precision set AMES_ABSTAIN_CONFIDENCE.
ABSTAIN_CONFIDENCE = _env_float("AMES_ABSTAIN_CONFIDENCE", 0.35)


def _confidence(bm25: Optional[float], cosine: Optional[float], have_vec: bool) -> float:
    """0..1 relevance estimate for one hit. `bm25`/`cosine` are None when that arm did not return
    the hit (or, for bm25, is unobservable: retriever.rrf hides per-arm scores). With no vector
    arm at all (embed failed) the lexical term stands alone; with no bm25 the vector term does."""
    lex = 0.0 if bm25 is None or bm25 <= 0 else bm25 / (bm25 + CONF_BM25_K)
    vec = 0.0 if cosine is None else min(1.0, max(0.0, (cosine - CONF_COS_LO) / (CONF_COS_HI - CONF_COS_LO)))
    if not have_vec:
        return round(lex, 4)
    if bm25 is None and cosine is not None:      # es_rrf: lexical score hidden
        return round(vec, 4)
    return round(CONF_VEC_WEIGHT * vec + (1.0 - CONF_VEC_WEIGHT) * lex, 4)


def _cosine(a: list, b: list) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else None


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


def _owner_hash(owner_id: str) -> str:
    return hashlib.sha256(owner_id.encode()).hexdigest()[:12]


def _log_recall(owner_id: str, size: int, rerank_requested: bool, filtered: bool,
                abstained: bool, timings: dict, rerank_info: Optional[dict] = None,
                embed_error: Optional[str] = None) -> None:
    """One structured line per recall. Carries a truncated owner hash and timings only: never
    the query text and never the owner id in clear."""
    line = {"owner": _owner_hash(owner_id), "k": size, "rerank": rerank_requested,
            "filtered": filtered, "abstained": abstained, "timings_ms": timings}
    if rerank_info is not None:
        line["rerank_applied"] = rerank_info["applied"]
        if rerank_info["fallback_reason"]:
            line["rerank_fallback"] = rerank_info["fallback_reason"]
    if embed_error:
        line["embed_error"] = embed_error
    _recall_log.info("ames.recall %s", json.dumps(line, separators=(",", ":"), sort_keys=True))


def recall(owner_id: str, query: str, kinds=None, size: int = 8,
           min_score: Optional[float] = None,
           as_of: Optional[str] = None,
           per_doc: Optional[int] = None,
           rerank: Optional[bool] = None,
           filter: Optional[dict] = None,
           rerank_depth: Optional[int] = None,
           min_similarity: Optional[float] = None,
           include_inactive_decisions: bool = False) -> dict:
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

    Tag filtering: `filter` ({all, any, none, narrow_any}, see tagfilter.py) restricts the
    candidate set INSIDE Elasticsearch (BM25 filter + kNN pre-filter + temporal arm). A filter
    that matches nothing returns `results: []` with `abstained: true` and never an unfiltered
    top-k. With no filter this function is unchanged.

    Decision governance: active decisions are returned as before, open proposals carry
    `decision_status: proposed`, and rejected / expired / superseded decisions are hidden unless
    `include_inactive_decisions` is true (they are stored `active: false`).
    """
    if not isinstance(include_inactive_decisions, bool):
        raise ValueError("include_inactive_decisions must be a boolean")
    t_start = time.perf_counter()
    if kinds:
        bad = [k for k in kinds if k not in KINDS]
        if bad:
            raise _tf.FilterError(f"kinds must be drawn from {KINDS}; got {bad[:3]}")
        kinds = list(dict.fromkeys(kinds))   # one ES search per kind: dedupe
    size = max(1, min(int(size), MAX_RECALL_SIZE))
    if per_doc is not None:
        per_doc = max(1, min(int(per_doc), MAX_PER_DOC))
    ensure_indices_once()
    parsed = _tf.parse_filter(filter)             # FilterError (a ValueError) -> HTTP 422
    flt_extra, flt_not, narrow_tags, resolved, vocab_truncated = [], [], None, {}, False
    timings = {}
    rerank_requested = bool(RERANK_DEFAULT if rerank is None else rerank)
    if parsed is not None:
        t_filter = time.perf_counter()
        if parsed["narrow_any"] is not None:
            def _vocab(keys):
                nonlocal vocab_truncated
                vocab, trunc = _tag_vocabulary(owner_id, kinds or list(KINDS), keys)
                vocab_truncated = vocab_truncated or trunc
                return vocab
            narrow_tags, resolved = _tf.resolve_narrow(parsed["narrow_any"], _vocab)
            if not narrow_tags:
                timings["filter_resolve"] = _ms(t_filter)
                timings["total"] = _ms(t_start)
                out = _abstained(query, parsed, resolved, vocab_truncated)
                out["timings_ms"] = timings
                out["context_chars"] = 0
                _log_recall(owner_id, size, rerank_requested, True, True, timings)
                return out
        flt_extra, flt_not = _tf.es_clauses(parsed, narrow_tags)
        timings["filter_resolve"] = _ms(t_filter)
    now = None
    if as_of:
        now = _parse_as_of(as_of)
    kinds = kinds or list(KINDS)
    sim_floor = KNN_MIN_SIMILARITY if min_similarity is None else _parse_similarity(min_similarity)
    t_embed = time.perf_counter()
    qvecs, embed_error = embeddings.embed_safe([query])
    qvec = qvecs[0] if qvecs else None
    timings["embed"] = _ms(t_embed)
    results = {}
    raw_arm = {}     # doc id -> (bm25, cosine) for the confidence signal
    search_by_kind, es_took = {}, 0
    # Over-fetch: the per-doc collapse runs AFTER ranking and can only shrink the list,
    # so each arm must return more than `size` candidates or a size-8 window would come
    # back under-filled whenever a group hits its cap.
    fetch = max(size, min(RECALL_FETCH_CAP, size * RECALL_FETCH_FACTOR))
    for kind in kinds:
        must = [_lexical_clause(query)]
        validity = []
        if now is not None:
            instant = now.isoformat()
            validity = [{"bool": {"should": [{"bool": {"must_not": {"exists": {"field": "valid_from"}}}},
                          {"range": {"valid_from": {"lte": instant}}}], "minimum_should_match": 1}},
                        {"bool": {"should": [{"bool": {"must_not": {"exists": {"field": "valid_to"}}}},
                          {"range": {"valid_to": {"gt": instant}}}], "minimum_should_match": 1}}]
        active_clause = {"term": {"active": True}}
        if include_inactive_decisions and kind == "semantic":
            active_clause = {"bool": {"should": [
                {"term": {"active": True}},
                {"terms": {"decision_status": list(governance.TERMINAL)}}], "minimum_should_match": 1}}
        base_filter = [active_clause, _visibility_filter(owner_id)] + flt_extra + validity
        lexical_query = {"bool": {"must": must, "filter": base_filter,
                                  **({"must_not": flt_not} if flt_not else {})}}
        if qvec is not None:
            knn = {"field": "embedding", "query_vector": qvec, "num_candidates": knn_num_candidates(fetch), "k": fetch,
                   **({"similarity": sim_floor} if sim_floor is not None else {}),
                   "filter": {"bool": {"filter": base_filter, "must_not": flt_not}} if flt_not else base_filter}
        if qvec is not None and fusion_mode() == "es_rrf":
            body = {
                "size": fetch,
                # ES 9.x drops dense_vector from _source by default; confidence needs the stored vector.
                "_source": {"includes": ["*", "embedding"]},
                "retriever": {"rrf": {
                    "retrievers": [{"standard": {"query": lexical_query}}, {"knn": knn}],
                    "rank_constant": RANK_CONSTANT,
                    "rank_window_size": RRF_RANK_WINDOW,
                }},
            }
        elif qvec is not None:
            body = None   # app-side fusion: two searches below
        else:
            body = {
                "size": fetch,
                "query": lexical_query,
                "sort": [{"_score": {"order": "desc"}}, {"occurred_at": {"order": "desc"}}],
            }
        t_search = time.perf_counter()
        if body is None:
            raw, took = _hybrid_app_fusion(kind, lexical_query, knn, fetch)
        else:
            r = es("POST", f"/{idx(kind)}/_search", body)
            raw, took = r["hits"]["hits"], r.get("took") or 0
        search_by_kind[kind] = _ms(t_search)
        es_took += took
        results[kind] = [
            {"id": h["_id"], "kind": kind, "score": h["_score"],
             **{k: v for k, v in h["_source"].items()
                if k not in ("kind", "embedding")}}
            for h in raw
        ]
        for h in raw:
            if body is None:                       # app_rrf: per-arm scores kept by the fusion
                raw_arm[h["_id"]] = (h.get("_bm25"), h.get("_cosine"))
            elif qvec is None:                     # BM25-only: _score is the lexical score
                raw_arm[h["_id"]] = (h["_score"], None)
            else:                                  # es_rrf hides per-arm scores; cosine from the stored vector
                raw_arm[h["_id"]] = (None, _cosine(qvec, h["_source"].get("embedding")))
    timings["search"] = round(sum(search_by_kind.values()), 3)
    timings["by_kind"] = search_by_kind
    timings["es_took"] = es_took
    # rank-fusion across kinds (simple RRF, k=RANK_CONSTANT)
    t_fuse = time.perf_counter()
    fused = {}
    for kind, hits in results.items():
        for rank, h in enumerate(hits):
            fused.setdefault(h["id"], {**h, "rrf": 0.0})
            fused[h["id"]]["rrf"] += 1.0 / (RANK_CONSTANT + rank + 1)
    GRAPH_RRF_WEIGHT = 0.01  # hop-2 evidence must not outvote lexical+kNN agreement
    # graph arm: entity two-hop expansion, fused by rank (never dominates other arms)
    try:
        from . import graph as _graph
        for rank, h in enumerate([] if parsed is not None else
                                 _graph.graph_expand(owner_id, query, size,
                                                     _visibility_filter(owner_id),
                                                     **({"include_inactive_decisions": True}
                                                        if include_inactive_decisions else {}))):
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
                                   {"range": {"occurred_at": {"gte": bs, "lte": be}}}] + flt_extra + validity,
                        **({"must_not": flt_not} if flt_not else {}),
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
        # graph/temporal-only hits have no arm score: confidence 0.0 = "no relevance evidence"
        h["confidence"] = _confidence(*raw_arm.get(h["id"], (None, None)), have_vec=qvec is not None)
    ordered = sorted(fused.values(), key=lambda x: -x["score"])
    if min_score is not None:
        ordered = [h for h in ordered if h["score"] >= min_score]
    reranked, rerank_info = False, None
    rerank_ms = 0.0
    if rerank_requested:
        t_rerank = time.perf_counter()
        ordered, rerank_info = _rerank_fused_meta(query, ordered, rerank_depth)
        reranked = rerank_info["applied"]
        timings["rerank"] = rerank_ms = _ms(t_rerank)
    window = _apply_budgets(ordered, size, per_doc_budget=per_doc or PER_DOC_BUDGET)
    # Extracted facts have a separate evidence lane and do not displace chunks.
    fact_hits = [h for h in results.get("semantic", []) if h.get("source_id")][:24] if "semantic" in kinds else []
    chunks = [h for h in ordered if not h.get("source_id")][:8]
    evidence = fact_hits + chunks
    # `fuse` = everything after the per-kind searches, rerank excluded: RRF, the graph and
    # temporal arms (they issue their own ES calls), recency, budgets.
    timings["fuse"] = round(max(0.0, _ms(t_fuse) - rerank_ms), 3)
    out = {
        "query": query,
        "results": window,
        "evidence": evidence,
        "context_chars": sum(len(h.get("text", "")) for h in evidence),
        "fact_count": len(fact_hits),
        "chunk_count": len(chunks),
        "by_kind": {k: v[:size] for k, v in results.items()},   # deprecated; compat shape
        "fused": window,      # deprecated alias of `results`
        "abstained": bool(min_score is not None and not window),
        "doc_groups": len({it.get("doc_group") or it.get("id") for it in window}),
        "reranked": reranked,
        "vector_arm": qvec is not None,
        "degraded": [] if qvec is not None else ["vector_arm"],
    }
    # True when nothing in the window is confidently relevant: the caller should say "I don't
    # know" instead of presenting the top hit as an answer. A hint, not a filter: results stay.
    out["abstain_hint"] = max((h["confidence"] for h in window), default=0.0) < ABSTAIN_CONFIDENCE
    if embed_error:
        out["embed_error"] = embed_error
    if rerank_info is not None:
        out["rerank"] = rerank_info     # additive: only present when a rerank was requested
    if parsed is not None:
        out["filter"] = {"applied": True, "resolved": resolved}
        if vocab_truncated:
            out["filter"]["vocab_truncated"] = True
        if not window:
            out["abstained"] = True
    # mental-model priority tier surfaces above facts (like reflect's source order);
    # omitted under a filter: it is a separate index of synthesized pages, not tagged memories
    try:
        from . import mental_models as _mm
        m = None if parsed is not None else _mm.match_model(owner_id, query)
        if m:
            out["mental_model"] = m
    except Exception:
        pass
    if temporal_meta:
        out["temporal"] = temporal_meta
    timings["total"] = _ms(t_start)
    out["timings_ms"] = timings
    _log_recall(owner_id, size, rerank_requested, parsed is not None, out["abstained"], timings, rerank_info, embed_error)
    return out


def promote(owner_id: str, kind: str, doc_id: str, to_visibility: str,
            person: Optional[str] = None) -> dict:
    """Copy-on-promote: private -> team/common creates a promoted copy; original stays private."""
    if to_visibility not in ("team", "common"):
        raise ValueError("promotion target must be team or common")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    s = _get_source(kind, doc_id)       # LookupError -> 404 at the edge, not a bare ES 404 -> 500
    if s["owner_id"] != owner_id:
        raise PermissionError("only the creator can promote their memory")
    governance.check_mutable(s)
    _assert_group_mutable(owner_id, kind, s)
    if s["visibility"] != "private":
        raise ValueError("only private memories can be promoted")
    allowed, reason = guard_promotion(s["text"], to_visibility)
    if not allowed:
        raise ValueError(reason)
    copy = {k: v for k, v in s.items() if k not in ("superseded_by", "supersedes")}
    copy["visibility"] = to_visibility
    copy["promoted_from"] = doc_id
    governed = governance.is_governed_write(kind, to_visibility, copy.get(_tf.TAG_FIELD))
    if governed:
        # promoting a private decision to the team bank is a proposal, never an active decision
        author = person or owner_id
        copy["doc_group"] = "dg_%s" % uuid.uuid4().hex     # the proposal gets its own group
        copy["passage_index"] = 0
        copy.update(governance.proposal_fields(author))
        governance.audit_propose(author, copy["doc_group"], copy.get("text_hash"), None)
    r = es("POST", f"/{idx(kind)}/_doc?refresh=true", copy)
    out = {"promoted_id": r["_id"], "visibility": to_visibility}
    if governed:
        out.update(governance.propose_result(copy))
    return out


def _supersede(owner_id: str, old_id: str, new_id: str, kind: str = "semantic") -> None:
    """Owner-checked supersede: both ends must belong to owner_id (atlas: Atlas's
    supersede update is keyed on id alone — we refuse that class of cross-tenant write)."""
    old = es("GET", f"/{idx(kind)}/_doc/{old_id}")["_source"]
    new = es("GET", f"/{idx(kind)}/_doc/{new_id}")["_source"]
    if old["owner_id"] != owner_id or new["owner_id"] != owner_id:
        raise PermissionError("supersede requires both memories to belong to the caller")
    if governance.is_governed(old) or governance.is_governed(new):
        raise governance.GovernedError()
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
        ], "must_not": governance.exclusion_clauses()}},
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
    degraded = list(evidence.get("degraded") or [])
    embed_error = evidence.get("embed_error")

    def _flag(out):
        out["degraded"] = degraded
        if embed_error:
            out["embed_error"] = embed_error
        return out
    sources = evidence.get("evidence") or [h for h in evidence["fused"][:5]]
    has_facts = bool(evidence.get("fact_count"))
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
                for d in more.get("degraded") or []:
                    if d not in degraded:
                        degraded.append(d)
                embed_error = embed_error or more.get("embed_error")
                seen = {s["id"] for s in sources}
                for h in (more.get("evidence", more["fused"]) if has_facts else more["fused"]):
                    if h["id"] not in seen and len(sources) < (32 if has_facts else 5):
                        sources.append(h)
        except Exception:
            pass  # rewrite unavailable → single-pass reflect, never fake rounds
    if not sources and model is None:
        return _flag({"question": question, "answer": "INSUFFICIENT_EVIDENCE", "sources": [],
                      "synthesized": False, "queries": queries_run})
    if model is not None:
        sources = [{"id": model["id"], "text": model["summary"],
                    "tier": "mental_model", **{k: v for k, v in model.items()
                                              if k in ("owner_id", "visibility", "updated_at")}}] + sources
    if llm_answer_fn is False:
        return _flag({"question": question, "sources": [s["id"] for s in sources],
                      "evidence": sources, "synthesized": False, "queries": queries_run})
    if llm_answer_fn is None:
        from . import llm as _llm
        fn = _llm.chat
    else:
        fn = llm_answer_fn
    answer = fn(question, sources)
    cited = [s["id"] for s in sources if s["id"] in (answer or "")]
    return _flag({"question": question, "answer": answer,
                  "sources": cited or [s["id"] for s in sources],
                  "synthesized": True, "queries": queries_run})


def _delete_superseded_note():
    """a96df78's delete() was removed here: it shadowed (was later in the file
    than) the unified delete() above after the feat/memory-mutation-api merge.
    Both call surfaces (REST DELETE /memory/{kind}/{doc_id} and POST
    /memory/delete) now use that single delete() with superset semantics:
    intent log (am_mutations) + optional tombstone (retract=True, semantic only) + group cascade + owner check + 404
    via LookupError. Contracts pinned by test_delete.py and test_update_delete.py."""
    return None
