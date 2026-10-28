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


_index_ready = False


def _ensure_index():
    global _index_ready
    if _index_ready:
        return
    try:
        es("GET", f"/{PREFIX}am_rejected")
    except RuntimeError:
        try:
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
        except RuntimeError as e:
            if "resource_already_exists_exception" not in str(e):  # lost a create race: fine
                raise
    _index_ready = True


def reject(owner_id: str, text: str, reason: str, source_id: str = "") -> dict | None:
    """Record a value as rejected for this owner (never true / judged wrong). Returns None
    without writing when the text has no salient terms: its key (sha256 of "") would block
    every future term-less text."""
    terms = _terms(text)
    if not terms:
        return None
    _ensure_index()
    doc = {"value_key": _key(terms), "owner_id": owner_id, "reason": reason,
           "terms": sorted(terms),
           "source_id": source_id,
           "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    es("POST", f"/{PREFIX}am_rejected/_doc?refresh=true", doc)
    return doc


MIN_FUZZY_TERMS = 3   # below this a tombstone is too thin to generalize: exact term-set only
PAGE = 500
MAX_PAGES = 20        # candidate-scan cap per call; hitting it fails closed
TERMS_CHUNK = 4096    # terms_set clause count is capped by ES (~6.5k); longer texts use the scan path
# ceil(0.8 * n) in int math for tombstones with >= MIN_FUZZY_TERMS terms; thinner ones get an
# unreachable n + 1 so only the value_key clause (exact term set) can match them
_MIN_MATCH = "int n = doc['terms'].size(); return n >= 3 ? (4 * n + 4) / 5 : n + 1;"


def _hit(s: dict, terms: set, key: str) -> bool:
    if s["value_key"] == key:
        return True
    rterms = set(s.get("terms", []))
    return len(rterms) >= MIN_FUZZY_TERMS and len(terms & rterms) / len(rterms) >= 0.8


def is_rejected(owner_id: str, text: str) -> dict | None:
    """Return the rejection record if this value was rejected for this owner.
    Matches exact term-set, or (for tombstones with >= MIN_FUZZY_TERMS terms) a
    rewording whose salient terms are a superset of >=80% of the recorded terms
    (blocks 'Microsoft is where Alice works' against a rejection of 'Alice works at
    Microsoft'). Thinner tombstones would over-block, so they need an exact match.

    The whole rule runs in ES (terms_set for the 80% overlap, value_key term for exact and
    legacy empty-terms records), so the cost tracks real matches, not tombstones that merely
    share a word. Texts above TERMS_CHUNK terms exceed ES's clause cap and fall back to paging
    all of the owner's tombstones through the same Python check. Paging is capped at
    MAX_PAGES and FAILS CLOSED: an unchecked backlog is treated as a rejection."""
    terms = _terms(text)
    if not terms:
        return None
    _ensure_index()
    key = _key(terms)
    should = [{"term": {"value_key": key}}]
    if len(terms) <= TERMS_CHUNK:
        should.append({"terms_set": {"terms": {
            "terms": sorted(terms), "minimum_should_match_script": {"source": _MIN_MATCH}}}})
    body = {"size": PAGE, "sort": [{"value_key": "asc"}, {"_doc": "asc"}],
            "query": {"bool": {"filter": [{"term": {"owner_id": owner_id}}]}}}
    if len(terms) <= TERMS_CHUNK:
        body["query"]["bool"].update(should=should, minimum_should_match=1)
    for _ in range(MAX_PAGES):
        hits = es("POST", f"/{PREFIX}am_rejected/_search", body)["hits"]["hits"]
        for h in hits:
            if _hit(h["_source"], terms, key):
                return h["_source"]
        if len(hits) < PAGE:
            return None
        body["search_after"] = hits[-1]["sort"]
    return {"reason": f"tombstone check exceeded {MAX_PAGES * PAGE} candidates; failing closed",
            "value_key": key, "owner_id": owner_id}
