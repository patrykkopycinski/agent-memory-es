"""Elasticsearch client + index management for agent-memory-es."""
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

ES_URL = os.environ.get("AMES_ES_URL", "http://localhost:9268")
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


def _guard_path(path: str, method: str) -> None:
    """Under tests (AMES_TEST_GUARD=1, set by tests/conftest.py): refuse any
    request whose path is not an amtest_-prefixed index operation or an ES
    cluster/API path (/_...). Belt-and-braces with the conftest session guard
    — a mis-set prefix must fail loudly here, before bytes hit the wire."""
    p = path.split("?", 1)[0].lstrip("/")
    if p.startswith("_") or p == "":
        return  # cluster-level APIs (_search across indices, _cat, /_bulk...)
    index = p.split("/", 1)[0]
    if not index.startswith(PREFIX):
        raise RuntimeError(
            f"ES test guard: refusing {method} {path} — index {index!r} does "
            f"not start with required test prefix {PREFIX!r}. This would hit "
            "non-test data; check AMES_INDEX_PREFIX.")


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
