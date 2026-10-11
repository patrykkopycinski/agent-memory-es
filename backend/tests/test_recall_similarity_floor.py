"""Issue #1: the kNN arm has no similarity floor, so every kind's rank-1 kNN hit lands in
the fused top-3 regardless of relevance.

Contract pinned here:
  * `min_similarity` (request) / AMES_KNN_MIN_SIMILARITY (env) is forwarded to the ES knn
    retriever as `similarity`; unset (the default) leaves the request body unchanged;
  * an irrelevant kNN #1 is dropped below the floor, a relevant one is kept (positive control).

Integration tests run against live ES like the rest of the suite:
  docker compose ... run --rm --no-deps -v <repo>:/srv -w /srv/backend \
         ames-backend python -m pytest tests/test_recall_similarity_floor.py -v
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import embeddings, memory  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qsimfloor"
RELEVANT_DOC = "The quarterly infrastructure budget covers three cloud regions and one dedicated host"
IRRELEVANT_DOC = "Recipe for sourdough bread: mix flour, water and salt, then proof overnight"
QUERY = "how much do we spend on servers each quarter"


def _wipe():
    for kind in ("episodic", "semantic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": OWNER}}})


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# ------------------------------------------------------------------ pure units
def test_parse_similarity_accepts_unset_and_range():
    assert memory._parse_similarity(None) is None
    assert memory._parse_similarity("") is None
    assert memory._parse_similarity("0.8") == 0.8
    assert memory._parse_similarity(-1) == -1.0


@pytest.mark.parametrize("bad", ["1.5", "-1.01", "abc"])
def test_parse_similarity_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        memory._parse_similarity(bad)


class _Capture:
    """Stand-in for memory.es that records search bodies (no ES, no embeddings)."""

    def __init__(self):
        self.bodies = []

    def __call__(self, method, path, body=None):
        if path.endswith("/_search"):
            self.bodies.append(body)
        return {"hits": {"hits": []}}


def _knn_bodies(monkeypatch, **kw):
    cap = _Capture()
    monkeypatch.setattr(memory.embeddings, "embed", lambda texts: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(memory, "es", cap)
    memory.recall(OWNER, QUERY, kinds=["semantic"], size=3, rerank=False, **kw)
    knns = [r["knn"] for b in cap.bodies for r in b["retriever"]["rrf"]["retrievers"] if "knn" in r]
    assert knns, cap.bodies
    return knns


def test_default_sends_no_similarity_to_knn(monkeypatch):
    monkeypatch.setattr(memory, "KNN_MIN_SIMILARITY", None)
    assert all("similarity" not in k for k in _knn_bodies(monkeypatch))


def test_request_floor_is_forwarded_to_knn(monkeypatch):
    monkeypatch.setattr(memory, "KNN_MIN_SIMILARITY", None)
    assert [k["similarity"] for k in _knn_bodies(monkeypatch, min_similarity=0.83)] == [0.83]


def test_env_floor_applies_and_request_overrides_it(monkeypatch):
    monkeypatch.setattr(memory, "KNN_MIN_SIMILARITY", 0.7)
    assert [k["similarity"] for k in _knn_bodies(monkeypatch)] == [0.7]
    assert [k["similarity"] for k in _knn_bodies(monkeypatch, min_similarity=0.9)] == [0.9]


def test_zero_floor_is_forwarded_not_treated_as_unset(monkeypatch):
    monkeypatch.setattr(memory, "KNN_MIN_SIMILARITY", None)
    assert [k["similarity"] for k in _knn_bodies(monkeypatch, min_similarity=0.0)] == [0.0]


# ------------------------------------------------------------------ integration
def test_floor_drops_irrelevant_knn_top_hit_and_keeps_relevant_one(monkeypatch):
    monkeypatch.setattr(memory, "KNN_MIN_SIMILARITY", None)
    _wipe()
    memory.retain(OWNER, "semantic", RELEVANT_DOC)
    qv, rel_v = embeddings.embed([QUERY, RELEVANT_DOC])
    irr_v = embeddings.embed([IRRELEVANT_DOC])[0]
    rel, irr = _cos(qv, rel_v), _cos(qv, irr_v)
    assert rel > irr, (rel, irr)          # the embedding model separates the two at all
    floor = (rel + irr) / 2               # calibrated between them, model-independent

    # positive control: the relevant hit survives the floor
    kept = memory.recall(OWNER, QUERY, kinds=["semantic"], size=8, rerank=False,
                         min_similarity=floor)["results"]
    assert any(RELEVANT_DOC in h["text"] for h in kept), kept

    # an irrelevant doc alone is the kNN #1 with no floor ...
    _wipe()
    memory.retain(OWNER, "semantic", IRRELEVANT_DOC)
    unfloored = memory.recall(OWNER, QUERY, kinds=["semantic"], size=8, rerank=False)["results"]
    assert any(IRRELEVANT_DOC in h["text"] for h in unfloored), unfloored
    # ... and is dropped once the floor is set
    floored = memory.recall(OWNER, QUERY, kinds=["semantic"], size=8, rerank=False,
                            min_similarity=floor)["results"]
    assert not any(IRRELEVANT_DOC in h["text"] for h in floored), floored
    _wipe()
