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


def is_rejected(owner_id: str, text: str) -> dict | None:
    """Return the rejection record if this value was rejected for this owner.
    Matches exact term-set, or a rewording whose salient terms are a superset of
    >=80% of a recorded rejection's terms (blocks 'Microsoft is where Alice works'
    against a rejection of 'Alice works at Microsoft')."""
    _ensure_index()
    terms = _terms(text)
    body = {"size": 50, "query": {"bool": {"filter": [{"term": {"owner_id": owner_id}}]}}}
    r = es("POST", f"/{PREFIX}am_rejected/_search", body)
    for h in r["hits"]["hits"]:
        s = h["_source"]
        rterms = set(s.get("terms", []))
        if not rterms:
            if s["value_key"] == _key(terms):
                return s
            continue
        overlap = len(terms & rterms) / len(rterms)
        if overlap >= 0.8:
            return s
    return None
