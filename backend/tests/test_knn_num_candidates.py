"""Filtered kNN on the bbq_hnsw index must examine enough HNSW candidates. A prod probe (active +
visibility filter, am_episodic) lost the two exact top-ranked docs at num_candidates=100 and
found them at 200+; recall@10 vs exact was 0.84 at nc=100 and 0.99 at nc>=200.

Limitation: the miss depends on graph structure and filter correlation of a large real index. It
did not reproduce on synthetic clustered corpora of 20k (10% filter) and 40k (2% filter) docs on
the test ES, so these tests pin the REQUEST BODIES (mocked ES) rather than the miss itself."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest

from app import embeddings, facts, memory  # noqa: E402
from app.store import ES_MAX_NUM_CANDIDATES, knn_num_candidates  # noqa: E402


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("AMES_KNN_NUM_CANDIDATES", raising=False)


def test_default_is_at_least_500_and_scales_with_k():
    assert knn_num_candidates(32) == 500
    assert knn_num_candidates(1) == 500
    assert knn_num_candidates(80) == 800


def test_env_override_is_capped_and_never_below_k(monkeypatch):
    monkeypatch.setenv("AMES_KNN_NUM_CANDIDATES", "1234")
    assert knn_num_candidates(32) == 1234
    monkeypatch.setenv("AMES_KNN_NUM_CANDIDATES", "999999")
    assert knn_num_candidates(32) == ES_MAX_NUM_CANDIDATES == 10000
    monkeypatch.setenv("AMES_KNN_NUM_CANDIDATES", "5")
    assert knn_num_candidates(32) == 32          # ES rejects num_candidates < k
    monkeypatch.setenv("AMES_KNN_NUM_CANDIDATES", "not-a-number")
    assert knn_num_candidates(32) == 500


def _recall_bodies(monkeypatch, fusion, **kw):
    bodies = []

    def spy(method, path, body=None):
        if path.endswith("/_search"):
            bodies.append(body)
        return {"hits": {"hits": []}}
    monkeypatch.setenv("AMES_FUSION", fusion)
    monkeypatch.setattr(memory, "es", spy)
    monkeypatch.setattr(embeddings, "embed", lambda t: [[0.1] * 384 for _ in t])
    memory.recall("o", "q", kinds=["episodic"], rerank=False, **kw)
    return bodies


@pytest.mark.parametrize("fusion", ["app", "es"])
def test_recall_knn_arm_carries_the_wide_candidate_pool(monkeypatch, fusion):
    bodies = _recall_bodies(monkeypatch, fusion, size=8)
    knn = (next(b for b in bodies if "retriever" in b)["retriever"]["rrf"]["retrievers"][1]["knn"]
           if fusion == "es" else next(b for b in bodies if "knn" in b)["knn"])
    assert knn["num_candidates"] >= 500
    assert knn["k"] == 32                         # k stays = fetch (size 8 * factor 4)
    assert knn["num_candidates"] >= knn["k"]


def test_recall_knn_honours_the_env_knob(monkeypatch):
    monkeypatch.setenv("AMES_KNN_NUM_CANDIDATES", "2000")
    bodies = _recall_bodies(monkeypatch, "app", size=8)
    assert next(b for b in bodies if "knn" in b)["knn"]["num_candidates"] == 2000


def test_filtered_recall_uses_the_same_pool(monkeypatch):
    bodies = _recall_bodies(monkeypatch, "app", size=8, filter={"all": ["name:x"], "none": ["state:y"]})
    knn = next(b for b in bodies if "knn" in b)["knn"]
    assert knn["num_candidates"] >= 500 and "must_not" in knn["filter"]["bool"]


def test_write_path_dup_and_fact_replacement_lookups_use_the_same_helper(monkeypatch):
    seen = []

    def spy(method, path, body=None):
        if path.endswith("/_search"):
            seen.append(body["knn"]["num_candidates"])
        return {"hits": {"hits": []}}
    monkeypatch.setattr(memory, "es", spy)
    monkeypatch.setattr(facts, "es", spy)
    memory._nearest_dup("o", "semantic", "private", [0.1] * 384)
    facts._replacement_ids("o", "private", [0.1] * 384, "text", "2026-01-01")
    assert seen == [500, 500]
