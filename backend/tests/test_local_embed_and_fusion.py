"""P0-2: off-license embedding (AMES_EMBED_BACKEND=local) and app-side RRF (AMES_FUSION=app).

Parity: the in-process multilingual-e5-small must reproduce the vectors ES `_inference` returns
(fixtures captured from `.multilingual-e5-small-elasticsearch`), with a wrong-model negative
control proving the threshold actually discriminates. Model files are fetched once into
AMES_TEST_MODEL_CACHE (default ~/.cache/ames-test-models)."""
import json
import math
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest

from app import embeddings, memory  # noqa: E402

CACHE = os.environ.get("AMES_TEST_MODEL_CACHE", os.path.expanduser("~/.cache/ames-test-models"))
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "e5_parity.json")
PARITY_THRESHOLD = 0.9999
BGE_REPO, BGE_REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@pytest.fixture(scope="module")
def fixture():
    return json.load(open(FIXTURE))["items"]


@pytest.fixture(scope="module")
def e5():
    return embeddings.load_local(os.path.join(CACHE, "multilingual-e5-small"))


# ------------------------------------------------------------------ parity
def test_local_e5_matches_es_inference_vectors(e5, fixture):
    session, tok = e5
    got = embeddings.local_embed_with(session, tok, [i["text"] for i in fixture])
    cosines = [_cos(g, i["vector"]) for g, i in zip(got, fixture)]
    print("e5 parity cosines:", [round(c, 6) for c in cosines])
    assert all(len(g) == 384 for g in got)
    assert min(cosines) >= PARITY_THRESHOLD, cosines


def test_local_embedding_is_unit_norm_and_batch_independent(e5, fixture):
    session, tok = e5
    texts = [i["text"] for i in fixture]
    batch = embeddings.local_embed_with(session, tok, texts)
    single = [embeddings.local_embed_with(session, tok, [t])[0] for t in texts]
    for b, s in zip(batch, single):
        assert abs(math.sqrt(sum(x * x for x in b)) - 1.0) < 1e-4
        assert _cos(b, s) >= PARITY_THRESHOLD     # padding must not leak into the pooled vector


def test_negative_control_wrong_model_fails_the_parity_threshold(fixture):
    """bge-small-en-v1.5 is also 384-dim: only the cosine check can tell it apart from E5."""
    session, tok = embeddings.load_local(os.path.join(CACHE, "bge-small-en-v1.5"),
                                         repo=BGE_REPO, revision=BGE_REV)
    got = embeddings.local_embed_with(session, tok, [i["text"] for i in fixture])
    cosines = [_cos(g, i["vector"]) for g, i in zip(got, fixture)]
    print("bge negative-control cosines:", [round(c, 6) for c in cosines])
    assert len(got[0]) == 384
    assert max(cosines) < PARITY_THRESHOLD
    assert max(cosines) <= 0.2, cosines


# ------------------------------------------------------------------ RRF pure function
def test_rrf_fuse_matches_es_rrf_scoring():
    out = memory.rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], rank_constant=60)
    scores = dict(out)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [d for d, _ in out] == ["a", "c", "b", "d"]    # a (1/61+1/62) > c (1/61+1/63) > b > d


def test_rrf_fuse_agreement_beats_a_single_arm_top_hit():
    out = [d for d, _ in memory.rrf_fuse([["solo", "both"], ["x", "both"]])]
    assert out[0] == "both"


def test_rrf_fuse_window_truncates_each_list():
    lex = [f"l{i}" for i in range(60)]
    out = dict(memory.rrf_fuse([lex, ["v0"]], window=50))
    assert "l49" in out and "l50" not in out and "v0" in out


def test_rrf_fuse_ties_are_deterministic():
    # equal scores: better best-rank first, then earlier list, then id
    assert [d for d, _ in memory.rrf_fuse([["b"], ["a"]])] == ["b", "a"]
    assert [d for d, _ in memory.rrf_fuse([["x"], ["x"]])] == ["x"]
    assert memory.rrf_fuse([]) == [] and memory.rrf_fuse([[], []]) == []


def test_fusion_mode_env(monkeypatch):
    monkeypatch.delenv("AMES_FUSION", raising=False)
    assert memory.fusion_mode() == "app_rrf"
    monkeypatch.setenv("AMES_FUSION", "es")
    assert memory.fusion_mode() == "es_rrf"


# ------------------------------------------------------------------ end to end, off-license
@pytest.fixture
def off_license(e5, monkeypatch):
    """Local embedder + app fusion, with every `_inference` call and every retriever search
    made fatal, as under an expired license (ES answers 403)."""
    monkeypatch.setenv("AMES_FUSION", "app")
    monkeypatch.setattr(embeddings, "_backend", "local")
    monkeypatch.setattr(embeddings, "_local", e5)
    real_es = memory.es
    seen = []

    def guarded(method, path, body=None):
        seen.append(path)
        assert "_inference" not in path
        if body and "retriever" in body:
            raise RuntimeError(f"ES {method} {path} -> 403: current license is non-compliant for [retrievers]")
        return real_es(method, path, body)
    monkeypatch.setattr(memory, "es", guarded)
    monkeypatch.setattr(embeddings, "_post", lambda *a, **k: pytest.fail("embed hit ES _inference"))
    return seen


def test_recall_is_hybrid_without_inference_or_es_rrf(off_license):
    owner = "local-fusion-" + uuid.uuid4().hex[:8]
    token = uuid.uuid4().hex[:10]
    memory.retain(owner, "semantic", f"Quarterly harbour crane maintenance schedule {token}", "private")
    memory.retain(owner, "semantic", "Zebra migration routes cross the savanna in spring", "private")
    r = memory.recall(owner, "when is the crane servicing planned", kinds=["semantic"], rerank=False)
    assert r["vector_arm"] is True and r["degraded"] == []
    assert "crane" in r["results"][0]["text"]
    # semantic-only match (no shared term with the query): proves the kNN arm contributed
    r2 = memory.recall(owner, "wildebeest seasonal journeys", kinds=["semantic"], rerank=False)
    assert r2["vector_arm"] is True
    assert any("Zebra" in h["text"] for h in r2["results"]), r2["results"]
    assert all("embedding" not in h for h in r["results"])


def test_retain_with_local_backend_stores_a_384_dim_vector(off_license):
    owner = "local-retain-" + uuid.uuid4().hex[:8]
    out = memory.retain(owner, "semantic", f"local retain probe {uuid.uuid4().hex}", "private")
    assert out["embedded"] is True
    # ES 9.x keeps dense_vector out of _source; ask for it explicitly via `fields`.
    hit = memory.es("POST", f"/{memory.idx('semantic')}/_search", {
        "size": 1, "query": {"ids": {"values": [out["_id"]]}}, "fields": ["embedding"]})["hits"]["hits"][0]
    assert len(hit["fields"]["embedding"]) == 384


def test_embed_failure_falls_back_to_bm25_without_a_500(monkeypatch):
    owner = "local-degrade-" + uuid.uuid4().hex[:8]
    memory.retain(owner, "semantic", "pelican feeding schedule at the estuary", "private")
    monkeypatch.setenv("AMES_FUSION", "app")
    monkeypatch.setattr(embeddings, "embed", lambda t: (_ for _ in ()).throw(RuntimeError("model gone")))
    r = memory.recall(owner, "pelican feeding", kinds=["semantic"], rerank=False)
    assert r["vector_arm"] is False and r["degraded"] == ["vector_arm"]
    assert r["results"] and "pelican" in r["results"][0]["text"]


def test_es_fusion_flag_still_sends_the_retriever_body(monkeypatch):
    monkeypatch.setenv("AMES_FUSION", "es")
    bodies = []

    def spy(method, path, body=None):
        if path.endswith("/_search"):
            bodies.append(body)
        return {"hits": {"hits": []}}
    monkeypatch.setattr(memory, "es", spy)
    monkeypatch.setattr(embeddings, "embed", lambda t: [[0.1] * 384 for _ in t])
    memory.recall("o", "q", kinds=["semantic"], rerank=False)
    assert len(bodies) == 1 and "retriever" in bodies[0]
    rrf = bodies[0]["retriever"]["rrf"]
    assert (rrf["rank_constant"], rrf["rank_window_size"]) == (60, 50)


def test_app_fusion_sends_two_searches_with_identical_filters(monkeypatch):
    monkeypatch.setenv("AMES_FUSION", "app")
    bodies = []

    def spy(method, path, body=None):
        if path.endswith("/_search"):
            bodies.append(body)
        return {"hits": {"hits": []}}
    monkeypatch.setattr(memory, "es", spy)
    monkeypatch.setattr(embeddings, "embed", lambda t: [[0.1] * 384 for _ in t])
    memory.recall("o", "q", kinds=["semantic"], rerank=False, filter={"all": ["name:x"], "none": ["state:y"]})
    lex = next(b for b in bodies if "query" in b)
    knn = next(b for b in bodies if "knn" in b)
    assert not any("retriever" in b for b in bodies)
    assert knn["knn"]["filter"]["bool"]["filter"] == lex["query"]["bool"]["filter"]
    assert knn["knn"]["filter"]["bool"]["must_not"] == lex["query"]["bool"]["must_not"]
