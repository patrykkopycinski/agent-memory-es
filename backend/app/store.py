"""Elasticsearch client + index management for agent-memory-es."""
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

_raw_es_url = os.environ.get("AMES_ES_URL")
if not _raw_es_url:
    # Fail fast, before any request: there is no safe default. The historical
    # default (http://localhost:9268) is an ssh tunnel to the PRODUCTION AMES
    # cluster and has been polluted by accidental test runs before.
    raise RuntimeError(
        "AMES_ES_URL is not set. Refusing to guess an Elasticsearch target "
        "(the old default localhost:9268 is the PROD tunnel). Set it "
        "explicitly — e.g. http://localhost:9200 or http://ames-es:9200.")
ES_URL = _raw_es_url
API_KEYS_FILE = os.environ.get("AMES_API_KEYS_FILE", os.path.join(os.path.dirname(__file__), "..", "data", "api_keys.json"))
# Test isolation: prefix all indices (e.g. amtest_) so suites never read/write
# the live cluster's shared data. Empty in production.
PREFIX = os.environ.get("AMES_INDEX_PREFIX", "")

KINDS = ("episodic", "semantic", "procedural")

MAPPINGS = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "kind": {"type": "keyword"},
            "owner_id": {"type": "keyword"},
            "visibility": {"type": "keyword"},
            "text": {"type": "text"},
            "entities": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine",
            },
            "occurred_at": {"type": "date"},
            "superseded_by": {"type": "keyword"},
            "supersedes": {"type": "keyword"},
            "text_hash": {"type": "keyword"},
            "doc_group": {"type": "keyword"},
            "passage_index": {"type": "integer"},
            "passages_total": {"type": "integer"},
            "active": {"type": "boolean"},
            "promoted_from": {"type": "keyword"},
            "tags": {"type": "keyword"},   # "key:value" labels (caller-supplied or extracted)
        },
    },
}


# Cluster-level APIs the suite actually uses; everything else under /_ is
# refused (bulk APIs like _bulk/_reindex/_aliases/_mget/_delete_by_query can
# hit any index and are not needed under the guard).
_ALLOWED_CLUSTER_APIS = (
    "_cat", "_cluster", "_inference",
    # read-only cross-index ops used by tools/diagnostics:
    "_stats", "_nodes",
)


def _guard_path(path: str, method: str) -> None:
    """Under tests (AMES_TEST_GUARD=1, set by tests/conftest.py and
    tests/_safety.py): refuse any request that could touch non-test data.
    Every comma-separated index name is checked; wildcards and _all are
    refused; cluster-level APIs pass only if allowlisted AND read-only."""
    p = path.split("?", 1)[0].lstrip("/")
    if p == "":
        return
    if p.startswith("_"):
        first = p.split("/", 1)[0]
        if method not in ("GET", "HEAD"):
            raise RuntimeError(
                f"ES test guard: refusing {method} {path} — cluster-level "
                "writes are not allowed under the test guard")
        if not first.startswith(_ALLOWED_CLUSTER_APIS):
            raise RuntimeError(
                f"ES test guard: refusing {method} {path} — cluster API "
                f"{first!r} is not in the test allowlist {_ALLOWED_CLUSTER_APIS}")
        return
    for index in p.split("/", 1)[0].split(","):
        if index == "_all" or "*" in index:
            raise RuntimeError(
                f"ES test guard: refusing {method} {path} — wildcard or "
                f"_all index {index!r} not allowed under the test guard")
        if not index.startswith(PREFIX):
            raise RuntimeError(
                f"ES test guard: refusing {method} {path} — index {index!r} "
                f"does not start with required test prefix {PREFIX!r}. This "
                "would hit non-test data; check AMES_INDEX_PREFIX.")


def es(method: str, path: str, body: Optional[dict] = None) -> dict:
    if os.environ.get("AMES_TEST_GUARD") == "1":
        _guard_path(path, method)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        ES_URL + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        raise RuntimeError(f"ES {method} {path} -> {e.code}: {detail}") from e


def ensure_indices() -> None:
    for kind in KINDS:
        try:
            es("GET", f"/{idx(kind)}")
        except RuntimeError:
            es("PUT", f"/{idx(kind)}", MAPPINGS)
            continue
        # Additive mapping migration: indices created before a field existed must
        # still accept it (dynamic:strict would otherwise reject the write). PUT
        # _mapping is idempotent for unchanged field types.
        try:
            es("PUT", f"/{idx(kind)}/_mapping",
               {"properties": {"text_hash": {"type": "keyword"},
                               "doc_group": {"type": "keyword"},
                               "passage_index": {"type": "integer"},
                               "passages_total": {"type": "integer"},
                               "tags": {"type": "keyword"}}})
        except RuntimeError:
            pass  # older cluster / no permission: exact-hash dedup degrades to kNN


def idx(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    return f"{PREFIX}am_{kind}"
