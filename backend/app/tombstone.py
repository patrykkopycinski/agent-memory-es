"""Rejected-value tombstones (atlas finding #1): value-keyed rejection consulted BEFORE
semantic writes, so a writer cannot recreate something already judged wrong. Also the
'retracted / never true' state: reject() is what 'never recount this' means here.

Key: sha256 of normalized salient terms — survives rewording better than a text hash,
cheap enough to check on every write. Embedding-similarity rejection is future work.
"""
import hashlib
import json
import re
import time

from .store import es, PREFIX

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.+-]{2,}")
STOP = {"the", "and", "was", "for", "with", "not", "never", "always"}


def _terms(text: str) -> set:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOP}


def _key(terms) -> str:
    return hashlib.sha256(" ".join(sorted(terms)).encode()).hexdigest()


def _ensure_index():
    try:
        es("GET", f"/{PREFIX}am_rejected")
    except RuntimeError:
        es("PUT", f"/{PREFIX}am_rejected", {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "value_key": {"type": "keyword"},
                    "terms": {"type": "keyword"},
                    "owner_id": {"type": "keyword"},
                    "reason": {"type": "text"},
                    "source_id": {"type": "keyword"},
                    "rejected_at": {"type": "date"},
                },
            },
        })


def reject(owner_id: str, text: str, reason: str, source_id: str = "") -> dict:
    """Record a value as rejected for this owner (never true / judged wrong)."""
    _ensure_index()
    terms = _terms(text)
    doc = {"value_key": _key(terms), "owner_id": owner_id, "reason": reason,
           "terms": sorted(terms),
           "source_id": source_id,
           "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    es("POST", f"/{PREFIX}am_rejected/_doc?refresh=true", doc)
    return doc


MIN_FUZZY_TERMS = 3   # below this a tombstone is too thin to generalize: exact term-set only
PAGE = 500
TERMS_CHUNK = 4096    # stay far under ES index.max_terms_count (65536) for very long texts


def is_rejected(owner_id: str, text: str) -> dict | None:
    """Return the rejection record if this value was rejected for this owner.
    Matches exact term-set, or (for tombstones with >= MIN_FUZZY_TERMS terms) a
    rewording whose salient terms are a superset of >=80% of the recorded terms
    (blocks 'Microsoft is where Alice works' against a rejection of 'Alice works at
    Microsoft'). Thinner tombstones would over-block, so they need an exact match.

    The candidate filter runs in ES (owner + shares >=1 term, or same value_key for
    legacy records with no terms) and is paged with search_after, so correctness
    never depends on a hit cap. Sorting on value_key alone is safe despite ties:
    records with equal value_key have identical term sets, hence identical verdicts."""
    _ensure_index()
    terms = _terms(text)
    key = _key(terms)
    sorted_terms = sorted(terms)
    chunks = [sorted_terms[i:i + TERMS_CHUNK]
              for i in range(0, len(sorted_terms), TERMS_CHUNK)] or [[]]
    for chunk in chunks:
        should = [{"term": {"value_key": key}}]
        if chunk:
            should.append({"terms": {"terms": chunk}})
        body = {"size": PAGE, "sort": [{"value_key": "asc"}],
                "query": {"bool": {"filter": [{"term": {"owner_id": owner_id}}],
                                   "should": should, "minimum_should_match": 1}}}
        while True:
            hits = es("POST", f"/{PREFIX}am_rejected/_search", body)["hits"]["hits"]
            for h in hits:
                s = h["_source"]
                rterms = set(s.get("terms", []))
                if s["value_key"] == key:
                    return s
                if len(rterms) >= MIN_FUZZY_TERMS and len(terms & rterms) / len(rterms) >= 0.8:
                    return s
            if len(hits) < PAGE:
                break
            body["search_after"] = hits[-1]["sort"]
    return None
