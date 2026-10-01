"""Elasticsearch client + index management for agent-memory-es."""
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

ES_URL = os.environ.get("AMES_ES_URL", "http://localhost:9268")
API_KEYS_FILE = os.environ.get("AMES_API_KEYS_FILE", os.path.join(os.path.dirname(__file__), "..", "data", "api_keys.json"))

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
                "dims": 3072,
                "index": True,
                "similarity": "cosine",
            },
            "occurred_at": {"type": "date"},
            "superseded_by": {"type": "keyword"},
            "supersedes": {"type": "keyword"},
            "active": {"type": "boolean"},
            "promoted_from": {"type": "keyword"},
        },
    },
}


def es(method: str, path: str, body: Optional[dict] = None) -> dict:
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
            es("HEAD_OK", f"/am_{kind}") if False else None
            es("GET", f"/am_{kind}")
        except RuntimeError:
            es("PUT", f"/am_{kind}", MAPPINGS)


def idx(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    return f"am_{kind}"
