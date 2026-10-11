"""r7-v4: supersession tolerates hallucinated ids; backfill recovers expired leases."""
import pytest

from app import facts


# ---------------------------------------------------------------- A: supersession
def _fake_es_candidates(candidates):
    def fake_es(method, path, body=None):
        assert method == "POST" and "_search" in path
        return {"hits": {"hits": [
            {"_id": cid, "_score": 0.95, "_source": {"text": t, "valid_from": vf}}
            for cid, t, vf in candidates]}}
    return fake_es


def _run_replacement(monkeypatch, candidates, replace_ids):
    monkeypatch.setattr(facts, "es", _fake_es_candidates(candidates))
    monkeypatch.setattr(facts.llm, "chat_json",
                        lambda s, u: {"replace_ids": replace_ids})
    return facts._replacement_ids("alice", "private", [0.1], "new fact", "2024-05-01")


def test_A1_hallucinated_id_dropped_valid_kept(monkeypatch):
    out = _run_replacement(
        monkeypatch,
        [("old1", "Older fact.", "2020-01-01"), ("old2", "Later fact.", "2020-06-01")],
        ["old1", "ghost", "old2"])
    assert out == ["old1", "old2"]


def test_A2_all_ids_invalid_returns_empty(monkeypatch):
    out = _run_replacement(monkeypatch,
                           [("old1", "Older fact.", "2020-01-01")],
                           ["ghost", "future", 7])
    assert out == []


def test_A3_non_list_still_raises(monkeypatch):
    monkeypatch.setattr(facts, "es", _fake_es_candidates(
        [("old1", "Older fact.", "2020-01-01")]))
    monkeypatch.setattr(facts.llm, "chat_json", lambda s, u: {"replace_ids": "nope"})
    with pytest.raises(ValueError, match="invalid replacement ids"):
        facts._replacement_ids("alice", "private", [0.1], "new fact", "2024-05-01")


# ---------------------------------------------------------------- B: backfill lease recovery
class FakeJobs:
    """One expired-running job then completion; records run_once calls by id."""
    def __init__(self, state, lease_until, next_at, attempts=0):
        self.state = state
        self.lease_until = lease_until
        self.next_at = next_at
        self.calls = []

    def job(self, job_id):
        return {"state": self.state, "lease_until": self.lease_until,
                "next_at": self.next_at, "attempts": self.attempts}


def _backfill_with_job(monkeypatch, job_state, lease_until):
    ran = []
    seq = {"step": 0}
    sleeps = []

    def boom_sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 3:
            raise AssertionError("backfill loop spun without progress (%d sleeps)"
                                 % len(sleeps))
    monkeypatch.setattr(facts.time, "sleep", boom_sleep)

    def fake_es(method, path, body=None):
        # episodic scan: one document the first time, none after (scope finished)
        if "_search" in path and facts.idx("episodic") in path:
            seq["step"] += 1
            if seq["step"] == 1:
                return {"hits": {"hits": [
                    {"_id": "src1", "sort": ["2024-01-01", "g1"], "_source": {}}]}}
            return {"hits": {"hits": []}}
        if path.startswith("/" + facts.QUEUE) and method == "GET":
            # after run_once has been called once, the job is completed
            state = "completed" if ran else job_state
            return {"_source": {"state": state, "lease_until": lease_until,
                                "next_at": "2000-01-01T00:00:00Z", "attempts": 0}}
        if "_update" in path:
            return {}
        return {}

    monkeypatch.setattr(facts, "es", fake_es)
    monkeypatch.setattr(facts, "enqueue", lambda owner, sid: "job1")

    def fake_run_once(owner_id, job_id):
        ran.append(job_id)

    monkeypatch.setattr(facts, "run_once", fake_run_once)
    monkeypatch.setattr(facts, "_update_backfill", lambda *a, **k: None)
    facts._run_backfill("bfkey", {"owner_id": "alice", "visibility": "private"}, "tok")
    return ran


def test_B1_expired_running_job_is_rerun_and_scope_completes(monkeypatch):
    ran = _backfill_with_job(monkeypatch, "running", "2000-01-01T00:00:00Z")
    assert ran == ["job1"]          # expired lease -> run_once -> completed -> break


def test_B2_live_lease_running_job_not_stolen(monkeypatch):
    # live lease: run_once must NOT be called; the bounded sleep guard turns the
    # spin into a test failure instead of a hang
    with pytest.raises(AssertionError, match="spun"):
        _backfill_with_job(monkeypatch, "running", "2999-01-01T00:00:00Z")
