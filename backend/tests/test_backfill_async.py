"""Backfill scheduling and durable runner contract (fake ES transport)."""
import copy
import re

from app import facts, main


class FakeES:
    def __init__(self):
        self.docs = {}
        self.sources = []
        self.processed = []

    def __call__(self, method, path, body=None):
        if f"/{facts.BACKFILLS}/_create/" in path:
            key = path.split("/_create/")[1].split("?")[0]
            if key in self.docs:
                raise RuntimeError("-> 409: conflict")
            self.docs[key] = copy.deepcopy(body)
            return {}
        if f"/{facts.BACKFILLS}/_doc/" in path:
            key = path.rsplit("/", 1)[-1]
            if key not in self.docs:
                raise RuntimeError("-> 404: missing")
            return {"_source": copy.deepcopy(self.docs[key]), "_seq_no": self.docs[key].get("_seq", 0), "_primary_term": 1}
        if f"/{facts.BACKFILLS}/_update/" in path:
            key = path.split("/_update/")[1].split("?")[0]
            seq = int(re.search(r"if_seq_no=(\d+)", path).group(1))
            if seq != self.docs[key].get("_seq", 0):
                raise RuntimeError("-> 409: conflict")
            self.docs[key].update(body["doc"])
            self.docs[key]["_seq"] = seq + 1
            return {}
        if path == f"/{facts.BACKFILLS}/_search":
            hits = [{"_id": k, "_source": copy.deepcopy(v), "_seq_no": v.get("_seq", 0), "_primary_term": 1}
                    for k, v in self.docs.items() if v["state"] == "pending"]
            return {"hits": {"hits": hits}}
        if path == f"/{facts.idx('episodic')}/_search":
            filters = body["query"]["bool"]["filter"]
            owner = next(x["term"]["owner_id"] for x in filters if "owner_id" in x.get("term", {}))
            visibility = next(x["term"]["visibility"] for x in filters if "visibility" in x.get("term", {}))
            rows = [x for x in self.sources if x["owner"] == owner and x["visibility"] == visibility
                    and (not body.get("search_after") or x["sort"] > body["search_after"])]
            rows.sort(key=lambda x: x["sort"])
            return {"hits": {"hits": [{"_id": x["id"], "sort": x["sort"]} for x in rows[:body["size"]]]}}
        if path.startswith(f"/{facts.idx('episodic')}/_update/"):
            return {}
        if f"/{facts.QUEUE}/_doc/" in path:
            return {"_source": {"state": "pending", "next_at": "2000-01-01T00:00:00Z"}}
        raise AssertionError((method, path, body))


def test_scheduling_is_prompt_idempotent_and_owner_scoped(monkeypatch):
    fake = FakeES()
    monkeypatch.setattr(facts, "es", fake)
    one = main.fact_backfill({"owner_id": "alice"})
    assert one["state"] == "pending"
    assert main.fact_backfill({"owner_id": "alice"}) == one
    assert main.fact_backfill({"owner_id": "bob"})["owner_id"] == "bob"
    assert len(fake.docs) == 2
    assert facts.backfill_status("nobody", "private") is None


def test_runner_orders_sources_and_completes_only_after_processing(monkeypatch):
    fake = FakeES()
    fake.sources = [{"id": x, "sort": [t, x], "owner": "alice", "visibility": "private"}
                    for t, x in [(3, "c"), (1, "a"), (2, "b")]]
    monkeypatch.setattr(facts, "es", fake)
    monkeypatch.setattr(facts, "enqueue", lambda owner, source: source)
    monkeypatch.setattr(facts, "run_once", lambda owner, job_id: fake.processed.append(job_id))
    # Simulate the fact job state moving from pending to completed after run_once.
    original = fake.__call__
    def transport(method, path, body=None):
        if f"/{facts.QUEUE}/_doc/" in path and path.rsplit('/', 1)[-1] in fake.processed:
            return {"_source": {"state": "completed"}}
        return original(method, path, body)
    monkeypatch.setattr(facts, "es", transport)
    facts.backfill("alice", "private")
    assert facts.run_backfill_once()
    assert fake.processed == ["a", "b", "c"]
    assert facts.backfill_status("alice", "private")["state"] == "completed"
    assert facts.backfill_status("alice", "private")["queued"] == 3


def test_failure_not_marked_done(monkeypatch):
    fake = FakeES()
    fake.sources = [{"id": "a", "sort": [1, "a"], "owner": "alice", "visibility": "private"}]
    monkeypatch.setattr(facts, "es", fake)
    monkeypatch.setattr(facts, "enqueue", lambda *_: "a")
    def fail(*_):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(facts, "run_once", fail)
    facts.backfill("alice", "private")
    assert facts.run_backfill_once()
    result = facts.backfill_status("alice", "private")
    assert result["state"] == "failed"
    assert "LLM unavailable" in result["error"]
    assert result["finished_at"]
