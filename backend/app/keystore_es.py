"""Elasticsearch key store (AMES_KEY_STORE=es): API keys as documents in `{PREFIX}am_keys`, for
deploys with no persistent volume. Doc id = sha256 hex of the token; the plaintext token is never
stored. The index name matches `am_*`, so an existing least-privilege API key already covers it."""
import os
import threading
import time
from typing import Callable, Optional

from .auth import KeyStoreUnavailable
from .store import PREFIX, es, is_doc_miss

INDEX = f"{PREFIX}am_keys"
MAX_KEYS = 10000    # one search page: the worker's owner view and persons_of read it whole

_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "owner_id": {"type": "keyword"},
            "role": {"type": "keyword"},
            "person": {"type": "keyword"},
            "created_at": {"type": "date"},
        },
    },
}

_lock = threading.Lock()
_index_ready = False
# digest -> (monotonic expiry, entry). Positives only: an unknown key is never remembered as
# valid, and a key minted right after a failed lookup is not locked out.
_cache: dict = {}


def _ttl() -> float:
    try:
        return max(0.0, float(os.environ.get("AMES_KEY_CACHE_TTL_S", "30")))
    except ValueError:
        return 30.0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _unavailable(what: str, exc: Exception) -> KeyStoreUnavailable:
    # the cause stays server-side (chained); the message never carries ES bodies
    return KeyStoreUnavailable(f"key store unavailable: {what} ({type(exc).__name__})")


def _request(what: str, method: str, path: str, body: Optional[dict] = None) -> dict:
    try:
        return es(method, path, body)
    except (RuntimeError, OSError) as exc:
        raise _unavailable(what, exc) from exc


def _is_conflict(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "-> 409" in str(exc)


def ensure_index() -> None:
    global _index_ready
    if _index_ready:
        return
    try:
        es("GET", f"/{INDEX}")
    except RuntimeError as exc:
        if "-> 404" not in str(exc):
            raise _unavailable("index check", exc) from exc
        try:
            es("PUT", f"/{INDEX}", _MAPPING)
        except RuntimeError as put_exc:
            if "resource_already_exists_exception" not in str(put_exc):  # lost a create race: fine
                raise _unavailable("index create", put_exc) from put_exc
        except OSError as put_exc:
            raise _unavailable("index create", put_exc) from put_exc
    except OSError as exc:
        raise _unavailable("index check", exc) from exc
    _index_ready = True


def _entry(source: dict) -> dict:
    entry = {"owner_id": source["owner_id"], "role": source["role"]}
    if source.get("person"):
        entry["person"] = source["person"]
    return entry


def list_all() -> dict:
    """{digest: entry} for every key, shaped like the key file."""
    r = _request("list", "POST", f"/{INDEX}/_search",
                 {"size": MAX_KEYS, "query": {"match_all": {}},
                  "_source": ["owner_id", "role", "person"]})
    return {h["_id"]: _entry(h["_source"]) for h in r["hits"]["hits"]}


def persons_of(owner_id: Optional[str]) -> set:
    r = _request("persons", "POST", f"/{INDEX}/_search",
                 {"size": MAX_KEYS, "query": {"term": {"owner_id": owner_id or ""}},
                  "_source": ["person"]})
    return {h["_source"]["person"].lower() for h in r["hits"]["hits"] if h["_source"].get("person")}


def _create_doc(digest: str, doc: dict) -> bool:
    """op_type=create: an existing id is refused by ES, never overwritten. False on conflict."""
    try:
        es("PUT", f"/{INDEX}/_doc/{digest}?op_type=create&refresh=true", doc)
    except (RuntimeError, OSError) as exc:
        if _is_conflict(exc):
            return False
        raise _unavailable("write", exc) from exc
    return True


def create(entry: dict, mint: Callable[[], tuple]) -> str:
    ensure_index()
    doc = dict(entry, created_at=_now())
    for _ in range(3):
        token, digest = mint()
        if _create_doc(digest, doc):
            return token
    raise KeyStoreUnavailable("key store unavailable: could not mint a unique key")


def put_digest(digest: str, entry: dict) -> bool:
    """Migration write: True when created, False when the digest already exists (left untouched)."""
    ensure_index()
    doc = {"owner_id": entry["owner_id"], "role": entry.get("role", "owner"), "created_at": _now()}
    if entry.get("person"):
        doc["person"] = entry["person"]
    return _create_doc(digest, doc)


def set_person(owner_id: str, person: str) -> int:
    ensure_index()
    r = _request("set person", "POST", f"/{INDEX}/_update_by_query?refresh=true&conflicts=proceed", {
        "query": {"bool": {"filter": [{"term": {"owner_id": owner_id}}],
                           "must_not": [{"exists": {"field": "person"}}]}},
        "script": {"source": "ctx._source.person = params.p", "params": {"p": person}}})
    with _lock:
        _cache.clear()
    return r.get("updated", 0)


def get(digest: str) -> Optional[dict]:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(digest)
        if hit and hit[0] > now:
            return dict(hit[1])
        _cache.pop(digest, None)
    try:
        r = es("GET", f"/{INDEX}/_doc/{digest}")
    except RuntimeError as exc:
        if is_doc_miss(exc):
            return None
        raise _unavailable("lookup", exc) from exc
    except OSError as exc:
        raise _unavailable("lookup", exc) from exc
    if not r.get("found"):
        return None
    entry = _entry(r["_source"])
    ttl = _ttl()
    if ttl:
        with _lock:
            _cache[digest] = (now + ttl, entry)
    return dict(entry)


def delete(digest: str) -> bool:
    with _lock:
        _cache.pop(digest, None)
    try:
        es("DELETE", f"/{INDEX}/_doc/{digest}?refresh=true")
    except RuntimeError as exc:
        if "-> 404" in str(exc) and "not_found" in str(exc) and "index_not_found" not in str(exc):
            return False
        raise _unavailable("delete", exc) from exc
    except OSError as exc:
        raise _unavailable("delete", exc) from exc
    return True


def health() -> bool:
    """Cluster answers and the index exists. Never reads key documents."""
    try:
        es("GET", f"/{INDEX}/_count")
    except (RuntimeError, OSError):
        return False
    return True
