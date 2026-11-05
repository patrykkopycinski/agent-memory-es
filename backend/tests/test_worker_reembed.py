"""The worker re-embeds active docs that were stored without a vector (embedder down/loading at
write time). Bounded per tick, skipped while the embedder is degraded, and safe against
concurrent edits (409 is skipped)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest

from app import embeddings, worker  # noqa: E402


class FakeEs:
    def __init__(self, docs, conflict_ids=()):
        self.docs, self.conflict_ids, self.writes, self.searches = docs, set(conflict_ids), [], []

    def __call__(self, method, path, body=None, timeout=30):
        if path.endswith("/_search"):
            self.searches.append((path, body))
            kind = path.split("am_")[-1].split("/")[0]
            hits = [d for d in self.docs if d["kind"] == kind and "embedding" not in d["_source"]
                    and d["_source"]["active"]]
            hits.sort(key=lambda d: d["_source"]["occurred_at"])
            return {"hits": {"hits": [{**d, "_seq_no": 3, "_primary_term": 1}
                                      for d in hits[:body["size"]]]}}
        assert method == "POST" and "/_update/" in path and "if_seq_no=3&if_primary_term=1" in path
        doc_id = path.split("/_update/")[1].split("?")[0]
        self.writes.append((doc_id, body))
        if doc_id in self.conflict_ids:
            raise RuntimeError(f"ES POST {path} -> 409: version_conflict_engine_exception")
        return {"result": "updated"}


def _doc(i, kind="semantic", occurred="2026-11-05T03:37:0%dZ", **extra):
    return {"_id": f"d{i}", "kind": kind,
            "_source": {"text": f"fact {i}", "active": True, "occurred_at": occurred % (i % 10), **extra}}


@pytest.fixture
def healthy(monkeypatch):
    monkeypatch.setattr(embeddings, "probe", lambda: None)
    monkeypatch.setattr(embeddings, "embed", lambda texts: [[float(len(t))] for t in texts])


def _use(monkeypatch, fake):
    monkeypatch.setattr(worker, "es", fake)
    return fake


def test_a_vectorless_doc_gets_embedded_in_place(monkeypatch, healthy):
    fake = _use(monkeypatch, FakeEs([_doc(1), _doc(2, embedding=[1.0])]))
    out = worker.reembed_sweep()
    assert out["embedded"] == 1
    assert fake.writes == [("d1", {"doc": {"embedding": [6.0]}})]


def test_no_es_writes_when_the_embedder_is_degraded_or_loading(monkeypatch):
    fake = _use(monkeypatch, FakeEs([_doc(1)]))
    for exc in (embeddings.ModelLoading("loading"), RuntimeError("HTTP 403 license expired")):
        def probe(exc=exc):
            raise exc
        monkeypatch.setattr(embeddings, "probe", probe)
        monkeypatch.setattr(embeddings, "embed", lambda t: pytest.fail("embedded while degraded"))
        assert worker.reembed_sweep()["embedded"] == 0
    assert fake.writes == [] and fake.searches == []


def test_batch_cap_is_honoured_oldest_first_across_kinds(monkeypatch, healthy):
    docs = [_doc(i, kind=("semantic", "episodic", "procedural")[i % 3]) for i in range(1, 10)]
    fake = _use(monkeypatch, FakeEs(docs))
    out = worker.reembed_sweep(batch=4)
    assert out["embedded"] == 4 and len(fake.writes) == 4
    chosen = sorted(w[0] for w in fake.writes)
    assert chosen == ["d1", "d2", "d3", "d4"]          # the four oldest occurred_at values


def test_a_409_is_skipped_not_raised(monkeypatch, healthy):
    fake = _use(monkeypatch, FakeEs([_doc(1), _doc(2)], conflict_ids={"d1"}))
    out = worker.reembed_sweep()
    assert out["embedded"] == 1 and out["conflicts"] == 1
    assert [w[0] for w in fake.writes] == ["d1", "d2"]


def test_other_update_errors_still_raise(monkeypatch, healthy):
    def boom(method, path, body=None, timeout=30):
        if path.endswith("/_search"):
            return FakeEs([_doc(1)])(method, path, body)
        raise RuntimeError("ES POST x -> 500: boom")
    monkeypatch.setattr(worker, "es", boom)
    with pytest.raises(RuntimeError):
        worker.reembed_sweep()


def test_run_forever_calls_the_sweep_even_when_consolidation_fails(monkeypatch):
    calls = []

    class Stop(Exception):
        pass

    def stop(_):
        raise Stop
    monkeypatch.setattr(worker, "run_once", lambda: (_ for _ in ()).throw(RuntimeError("consolidate down")))
    monkeypatch.setattr(worker, "reembed_sweep", lambda: calls.append(1) or {})
    monkeypatch.setattr(worker.time, "sleep", stop)
    with pytest.raises(Stop):
        worker.run_forever(interval_s=1)
    assert calls == [1]
