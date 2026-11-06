"""Decision governance (docs/GOVERNANCE.md): team `kind:decision` memories are proposals until
people other than the author approve them. One test (or a few) per numbered case of the contract;
the case number is in each test name's docstring.

Own owners, wiped at import, so reruns on a persistent ES stay green. Every text carries a unique
token: team scope collapses identical text onto one doc, and the recall tests search by token."""
import importlib.util
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (guard: refuses anything but a throwaway ES)

import pytest
from fastapi.testclient import TestClient

from app import auth, embeddings, governance, memory, tagfilter
from app.main import app
from app.memory import extract_entities
from app.store import es, idx, PREFIX

client = TestClient(app)
PEOPLE = ("gov_alice", "gov_bob", "gov_carol", "gov_dave")
ALICE, BOB, CAROL, DAVE = PEOPLE
AGENT = {p: auth.create_key(p, "member") for p in PEOPLE}          # what an agent holds
APPROVER = {p: auth.create_key(p, "approver") for p in PEOPLE}     # minted for the same person
OWNER_KEY = auth.create_key("gov_owner", "owner")
H = lambda k: {"X-API-Key": k}  # noqa: E731
DECISION = "kind:decision"
SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")

for _name in (idx("semantic"), f"{PREFIX}am_mutations"):
    try:
        es("POST", f"/{_name}/_delete_by_query?refresh=true&ignore_unavailable=true",
           {"query": {"terms": {"owner_id": list(PEOPLE) + ["gov_owner", "system", "admin"]}}})
    except RuntimeError:
        pass


def _wipe_owners():
    es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true&ignore_unavailable=true&conflicts=proceed",
       {"query": {"terms": {"owner_id": list(PEOPLE) + ["gov_owner", "system", "admin"]}}})


@pytest.fixture(scope="module", autouse=True)
def _cleanup_team_docs():
    """Team docs are visible to every owner, so leave none behind for other modules' recall tests."""
    yield
    _wipe_owners()


def _tok(label="t"):
    return f"gv{label}{uuid.uuid4().hex[:8]}"


def _propose(who=ALICE, token=None, extra="", **body):
    token = token or _tok()
    payload = {"kind": "semantic", "text": f"decision {token} use the shared queue {extra}".strip(),
               "visibility": "team", "tags": [DECISION], "extract_facts": False, **body}
    r = client.post("/memory/retain", headers=H(AGENT[who]), json=payload)
    assert r.status_code == 200, r.text
    out = r.json()
    out["token"] = token
    return out


def _approve(who, doc_id, note=None):
    return client.post("/governance/approve", headers=H(APPROVER[who]),
                       json={"doc_id": doc_id, **({"note": note} if note else {})})


def _reject(who, doc_id, reason="no"):
    return client.post("/governance/reject", headers=H(APPROVER[who]),
                       json={"doc_id": doc_id, "reason": reason})


def _active(who=ALICE, approver=BOB, **kw):
    p = _propose(who, **kw)
    r = _approve(approver, p["_id"])
    assert r.status_code == 200 and r.json()["decision_status"] == "active", r.text
    return p


def _src(doc_id):
    return es("GET", f"/{idx('semantic')}/_doc/{doc_id}")["_source"]


def _rows(doc_id):
    r = es("POST", f"/{PREFIX}am_mutations/_search",
           {"size": 100, "query": {"term": {"doc_id": doc_id}}, "sort": [{"at": "asc"}]})
    return [h["_source"] for h in r["hits"]["hits"]]


def _recall(who, token, **extra):
    r = client.post("/memory/recall", headers=H(AGENT[who]),
                    json={"query": token, "kinds": ["semantic"], "size": 20, **extra})
    assert r.status_code == 200, r.text
    return {h["id"]: h for h in r.json()["results"]}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- case 1 ----

def test_case1_team_decision_is_proposed_private_and_howto_are_not():
    p = _propose()
    assert p["decision_status"] == "proposed" and p["approvals_needed"] == 1
    src = _src(p["_id"])
    assert src["decision_status"] == "proposed" and src["proposed_by"] == ALICE
    assert src["approvals"] == [] and src["proposal_expires_at"]

    priv = client.post("/memory/retain", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "text": f"private decision {_tok()}", "tags": [DECISION],
        "extract_facts": False}).json()
    assert "decision_status" not in priv and "decision_status" not in _src(priv["_id"])

    howto = client.post("/memory/retain", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "text": f"how to restart the worker {_tok()}", "visibility": "team",
        "tags": ["kind:howto"], "extract_facts": False}).json()
    assert "decision_status" not in howto and "decision_status" not in _src(howto["_id"])


def test_case1_proposals_listing_is_open_to_every_role():
    p = _propose()
    for key in (AGENT[BOB], APPROVER[BOB], OWNER_KEY):
        r = client.get("/governance/proposals", headers=H(key))
        assert r.status_code == 200, r.text
        assert p["_id"] in {x["id"] for x in r.json()["proposals"]}
    assert client.get("/governance/proposals").status_code == 401


# ---- case 2 ----

def test_case2_author_cannot_approve_own_proposal():
    p = _propose()
    r = _approve(ALICE, p["_id"])           # the author's OWN approver key
    assert r.status_code == 403, r.text
    assert _src(p["_id"])["decision_status"] == "proposed" and _src(p["_id"])["approvals"] == []


# ---- case 3 ----

@pytest.mark.parametrize("key", [AGENT[BOB], OWNER_KEY])
def test_case3_agent_keys_get_403_on_approve_and_reject(key):
    p = _propose()
    a = client.post("/governance/approve", headers=H(key), json={"doc_id": p["_id"]})
    b = client.post("/governance/reject", headers=H(key), json={"doc_id": p["_id"], "reason": "x"})
    assert a.status_code == 403 and b.status_code == 403
    assert _src(p["_id"])["decision_status"] == "proposed"


@pytest.mark.parametrize("tool", ["approve", "reject"])
def test_case3_no_mcp_http_tool_for_approval(tool):
    r = client.post(f"/mcp/tools/{tool}", headers=H(APPROVER[BOB]), json={"doc_id": "x"})
    assert r.status_code == 404


def test_case3_public_mcp_server_has_no_approve_or_reject_tool():
    mod = _load_script("ames_mcp_server")
    names = [t["name"] for t in mod.TOOLS]
    assert "memory_proposals" in names
    assert not [n for n in names if "approve" in n or "reject" in n]
    with pytest.raises(ValueError):
        mod.dispatch("http://x", "k", "memory_approve", {"doc_id": "x"})


def test_case3_cli_refuses_without_a_tty_and_takes_no_key_argument(monkeypatch):
    cli = _load_script("ames_approve")

    class NoTty:
        def isatty(self):
            return False

    monkeypatch.setenv("AMES_API_KEY", APPROVER[BOB])         # an env var is never read
    with pytest.raises(SystemExit):
        cli.read_key(stdin=NoTty(), prompt=lambda _p: pytest.fail("prompted without a TTY"))

    class Tty:
        def isatty(self):
            return True

    assert cli.read_key(stdin=Tty(), prompt=lambda _p: " typed-key ") == "typed-key"
    for flag in ("--key", "--api-key", "--key-file"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([flag, "x", "list"])


# ---- case 4 ----

def test_case4_one_non_author_approval_activates_a_new_decision():
    p = _propose()
    r = _approve(BOB, p["_id"], note="looks right")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision_status"] == "active" and body["activated"] is True
    src = _src(p["_id"])
    assert src["decision_status"] == "active" and src["approved_via"] == "approvals"
    assert src["decided_at"] and src["approvals"][0]["by"] == BOB
    assert src["approvals"][0]["text_hash"] == src["text_hash"]
    assert src["active"] is True
    assert _approve(BOB, p["_id"]).status_code == 409           # no longer open


# ---- case 5 ----

def test_case5_override_needs_two_distinct_approvers_and_supersedes_the_old_one():
    old = _active()
    new = _propose(ALICE, supersedes_decision=old["_id"])
    assert new["decision_status"] == "proposed" and new["approvals_needed"] == 2
    assert _src(new["_id"])["proposal_for"] == old["_id"]

    assert _approve(BOB, new["_id"]).status_code == 200
    assert _src(new["_id"])["decision_status"] == "proposed"
    again = _approve(BOB, new["_id"])                           # same approver twice counts once
    assert again.status_code == 409
    assert _src(new["_id"])["decision_status"] == "proposed"
    assert len(_src(new["_id"])["approvals"]) == 1
    assert _src(old["_id"])["decision_status"] == "active"

    r = _approve(CAROL, new["_id"])
    assert r.status_code == 200 and r.json()["decision_status"] == "active"
    o, n = _src(old["_id"]), _src(new["_id"])
    assert o["decision_status"] == "superseded" and o["superseded_by"] == new["_id"]
    assert o["active"] is False
    assert n["decision_status"] == "active" and n["supersedes"] == [old["_id"]]


def test_case5_the_authors_approval_does_not_count_towards_an_override():
    old = _active()
    new = _propose(ALICE, supersedes_decision=old["_id"])
    assert _approve(ALICE, new["_id"]).status_code == 403
    assert _approve(BOB, new["_id"]).status_code == 200
    assert _src(new["_id"])["decision_status"] == "proposed"


def test_case5_an_approval_made_for_other_text_does_not_count():
    src = {"text_hash": "h2", "proposed_by": ALICE, "approvals": [
        {"by": BOB, "text_hash": "h1"}, {"by": CAROL, "text_hash": "h2"},
        {"by": CAROL, "text_hash": "h2"}, {"by": ALICE, "text_hash": "h2"}]}
    assert [a["by"] for a in governance._valid_approvals(src)] == [CAROL]


# ---- case 6 ----

def test_case6_supersedes_must_point_at_an_active_governed_decision():
    proposed = _propose()
    plain = client.post("/memory/retain", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "text": f"plain team note {_tok()}", "visibility": "team",
        "extract_facts": False}).json()
    private = client.post("/memory/retain", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "text": f"private note {_tok()}", "extract_facts": False}).json()
    for target in (proposed["_id"], plain["_id"], private["_id"], "no-such-id"):
        tok = _tok()
        r = client.post("/memory/retain", headers=H(AGENT[ALICE]), json={
            "kind": "semantic", "text": f"override {tok}", "visibility": "team", "tags": [DECISION],
            "supersedes_decision": target, "extract_facts": False})
        assert r.status_code == 422, (target, r.text)
        hits = es("POST", f"/{idx('semantic')}/_count", {"query": {"match": {"text": tok}}})
        assert hits["count"] == 0, "a refused override must leave nothing behind"


def test_case6_supersedes_on_a_non_governed_write_is_refused():
    old = _active()
    r = client.post("/memory/retain", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "text": f"private {_tok()}", "supersedes_decision": old["_id"],
        "extract_facts": False})
    assert r.status_code == 422


# ---- case 7 ----

def test_case7_active_decision_is_read_only_even_for_its_creator():
    d = _active()
    h = H(AGENT[ALICE])
    u = client.post("/memory/update", headers=h, json={
        "kind": "semantic", "doc_id": d["_id"], "text": f"rewritten {_tok()}"})
    x = client.post("/memory/delete", headers=h, json={"kind": "semantic", "doc_id": d["_id"]})
    p = client.post("/memory/promote", headers=h, json={
        "kind": "semantic", "doc_id": d["_id"], "to_visibility": "common"})
    for r in (u, x, p):
        assert r.status_code == 409, r.text
        assert "propose an override" in r.text
    t = client.post("/mcp/tools/update", headers=h, json={
        "kind": "semantic", "doc_id": d["_id"], "text": f"rewritten {_tok()}"})
    assert t.status_code == 409
    assert _src(d["_id"])["decision_status"] == "active"


def test_case7_promoting_a_private_decision_creates_a_proposal_not_an_active_copy():
    private = client.post("/memory/retain", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "text": f"private decision {_tok()}", "tags": [DECISION],
        "extract_facts": False}).json()
    r = client.post("/memory/promote", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": private["_id"], "to_visibility": "team"})
    assert r.status_code == 200, r.text
    assert r.json()["decision_status"] == "proposed"
    assert _src(r.json()["promoted_id"])["decision_status"] == "proposed"


# ---- case 8 ----

def test_case8_editing_a_proposal_resets_its_approvals(monkeypatch):
    monkeypatch.setenv("AMES_GOV_NEW_APPROVALS", "2")
    p = _propose()
    assert _approve(BOB, p["_id"]).status_code == 200
    assert len(_src(p["_id"])["approvals"]) == 1
    old_hash = _src(p["_id"])["text_hash"]

    # only the author edits
    other = client.post("/memory/update", headers=H(AGENT[BOB]), json={
        "kind": "semantic", "doc_id": p["_id"], "text": f"hijack {_tok()}"})
    assert other.status_code == 403

    new_text = f"decision {p['token']} use the dedicated queue instead"
    r = client.post("/memory/update", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": p["_id"], "text": new_text})
    assert r.status_code == 200 and r.json()["updated"] is True, r.text
    src = _src(r.json()["_id"])
    assert src["decision_status"] == "proposed" and src["approvals"] == []
    assert src["text_hash"] != old_hash and src["proposed_by"] == ALICE
    assert src["doc_group"] == p["doc_group"]
    # the earlier approval does not carry over: bob must approve again, and 1 of 2 is not enough
    assert _approve(BOB, r.json()["_id"]).status_code == 200
    assert _src(r.json()["_id"])["decision_status"] == "proposed"
    assert _approve(CAROL, r.json()["_id"]).json()["decision_status"] == "active"


def test_case8_update_cannot_turn_a_proposal_private_or_untag_it():
    p = _propose()
    for body in ({"visibility": "private"}, {"tags": ["kind:howto"]}):
        r = client.post("/memory/update", headers=H(AGENT[ALICE]),
                        json={"kind": "semantic", "doc_id": p["_id"], **body})
        assert r.status_code == 422, r.text
    assert _src(p["_id"])["decision_status"] == "proposed"


def test_case8_update_cannot_create_a_decision_out_of_a_plain_team_memory():
    howto = client.post("/memory/retain", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "text": f"team howto {_tok()}", "visibility": "team",
        "tags": ["kind:howto"], "extract_facts": False}).json()
    r = client.post("/memory/update", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": howto["_id"], "tags": [DECISION]})
    assert r.status_code == 422, r.text


# ---- case 9 ----

def test_case9_consolidation_never_touches_governed_entries():
    t = _tok()
    a = _active(token=t, extra="for billing ledger storage engine")
    b = _active(token=t, extra="for billing ledger storage engine now")
    # control: the pass does run, and does supersede an ordinary near-duplicate pair
    ctl = f"control {_tok()} cache warmup runs nightly on the staging cluster"
    for text in (ctl, ctl + " ok"):
        doc = {"kind": "semantic", "owner_id": ALICE, "visibility": "private", "text": text,
               "entities": extract_entities(text), "occurred_at": "2026-01-01T00:00:00Z",
               "active": True, "embedding": embeddings.embed([text])[0]}
        es("POST", f"/{idx('semantic')}/_doc?refresh=true", doc)
    done = memory.consolidate(ALICE, dry_run=False)
    assert done["superseded"] >= 1, done
    paired = {p["older"] for p in done["proposals"]} | {p["newer"] for p in done["proposals"]}
    assert not {a["_id"], b["_id"]} & paired
    for d in (a, b):
        s = _src(d["_id"])
        assert s["active"] is True and "superseded_by" not in s and s["decision_status"] == "active"
    with pytest.raises(governance.GovernedError):
        memory._supersede(ALICE, a["_id"], b["_id"])


def test_case9_near_duplicate_retain_never_links_a_governed_entry():
    t = _tok()
    d = _active(token=t, extra="rotate the signing material every quarter without exception")
    howto = client.post("/memory/retain", headers=H(AGENT[BOB]), json={
        "kind": "semantic", "visibility": "team", "tags": ["kind:howto"], "extract_facts": False,
        "text": f"decision {t} use the shared queue rotate the signing material every quarter "
                "without exception ok"}).json()
    assert howto.get("dedup_reason") != "near_duplicate_linked"
    s = _src(d["_id"])
    assert s["active"] is True and "superseded_by" not in s
    assert "supersedes" not in _src(howto["_id"]) or d["_id"] not in _src(howto["_id"])["supersedes"]
    # and a governed write never links a plain neighbour either
    p = _propose(token=t, extra="rotate the signing material every quarter without exception now")
    assert "supersedes" not in _src(p["_id"])


# ---- case 10 ----

def test_case10_recall_labels_proposals_and_hides_closed_decisions_unless_asked(monkeypatch):
    live = _active(token=_tok("live"))
    rejected = _propose(token=_tok("rej"))
    expired = _propose(token=_tok("exp"))
    old = _active(token=_tok("old"))
    new = _propose(ALICE, token=_tok("new"), supersedes_decision=old["_id"])
    assert _approve(BOB, new["_id"]).status_code == 200
    assert _approve(CAROL, new["_id"]).status_code == 200            # old -> superseded
    monkeypatch.setenv("AMES_GOV_VETO_OWNERS", CAROL)
    assert _reject(CAROL, rejected["_id"]).json()["decision_status"] == "rejected"
    governance.expire_stale_proposals(now=time.time() + 400 * 86400)
    assert _src(expired["_id"])["decision_status"] == "expired"

    def seen(token, **extra):
        return _recall(DAVE, token, **extra)

    got = seen(live["token"])
    assert got[live["_id"]]["decision_status"] == "active"
    # the sweep above also expired `proposed`: label-check a fresh proposal instead
    fresh = _propose(token=_tok("fresh"))
    got = seen(fresh["token"])
    assert got[fresh["_id"]]["decision_status"] == "proposed"

    for closed, status in ((rejected, "rejected"), (expired, "expired"), (old, "superseded")):
        assert closed["_id"] not in seen(closed["token"])
        shown = seen(closed["token"], include_inactive_decisions=True)
        assert shown[closed["_id"]]["decision_status"] == status
    assert new["_id"] in seen(new["token"])


def test_case10_include_inactive_decisions_must_be_a_boolean():
    r = client.post("/memory/recall", headers=H(AGENT[BOB]), json={
        "query": "x", "include_inactive_decisions": "yes please"})
    assert r.status_code == 422


def test_case10_public_mcp_server_labels_proposed_hits():
    mod = _load_script("ames_mcp_server")
    res = mod.label_proposals({"results": [
        {"id": "1", "text": "use queue", "decision_status": "proposed"},
        {"id": "2", "text": "use cache", "decision_status": "active"}]})
    assert "PROPOSED, not in force" in res["results"][0]["text"]
    assert res["results"][1]["text"] == "use cache"


# ---- case 11 ----

def test_case11_veto_owner_rejection_rejects_others_only_object(monkeypatch):
    monkeypatch.setenv("AMES_GOV_VETO_OWNERS", CAROL)
    p = _propose()
    r = _reject(BOB, p["_id"], "too risky")
    assert r.status_code == 200 and r.json()["decision_status"] == "proposed"
    assert r.json()["veto"] is False
    s = _src(p["_id"])
    assert s["decision_status"] == "proposed" and s["objections"][0]["by"] == BOB
    assert s["objections"][0]["reason"] == "too risky"
    assert _reject(BOB, p["_id"]).status_code == 409                 # one objection per person

    r = _reject(CAROL, p["_id"], "vetoed")
    assert r.status_code == 200 and r.json()["decision_status"] == "rejected"
    s = _src(p["_id"])
    assert s["decision_status"] == "rejected" and s["active"] is False and s["decided_at"]
    assert _approve(DAVE, p["_id"]).status_code == 409               # closed
    assert p["_id"] not in {x["id"] for x in client.get(
        "/governance/proposals", headers=H(AGENT[BOB])).json()["proposals"]}


def test_case11_author_withdraws_by_deleting_which_rejects_the_proposal():
    p = _propose()
    r = client.post("/memory/delete", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": p["_id"]})
    assert r.status_code == 200 and r.json()["withdrawn"] is True, r.text
    s = _src(p["_id"])                                               # never hard-deleted
    assert s["decision_status"] == "rejected" and s["objections"][-1]["reason"] == "withdrawn"
    assert client.post("/memory/delete", headers=H(AGENT[BOB]), json={
        "kind": "semantic", "doc_id": _propose()["_id"]}).status_code == 403


# ---- case 12 ----

def test_case12_stale_proposal_expires_and_can_no_longer_be_approved():
    p = _propose()
    fresh = _propose()
    out = governance.expire_stale_proposals(now=time.time() + 400 * 86400)
    assert out["expired"] >= 2
    s = _src(p["_id"])
    assert s["decision_status"] == "expired" and s["active"] is False and s["decided_at"]
    assert _approve(BOB, p["_id"]).status_code == 409
    assert _src(fresh["_id"])["decision_status"] == "expired"
    # nothing newer than the cut-off is touched
    live = _propose()
    governance.expire_stale_proposals()
    assert _src(live["_id"])["decision_status"] == "proposed"


def test_case12_an_expired_deadline_blocks_approval_before_the_worker_runs():
    p = _propose()
    es("POST", f"/{idx('semantic')}/_update/{p['_id']}?refresh=true",
       {"doc": {"proposal_expires_at": "2000-01-01T00:00:00Z"}})
    assert _approve(BOB, p["_id"]).status_code == 409


# ---- case 13 ----

def test_case13_concurrent_override_approvals_both_count_and_activation_happens_once(monkeypatch):
    old = _active()
    new = _propose(ALICE, supersedes_decision=old["_id"])
    barrier = threading.Barrier(2, timeout=30)
    real_audit = governance._audit

    def gated(intent, actor, doc_id, detail):
        # both threads have read the same version of the proposal before either writes: the
        # second compare-and-set must conflict and recount
        if intent == "decision_approve" and detail.get("attempt") == 1:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return real_audit(intent, actor, doc_id, detail)

    monkeypatch.setattr(governance, "_audit", gated)
    out = {}

    def go(who):
        out[who] = _approve(who, new["_id"])

    threads = [threading.Thread(target=go, args=(w,)) for w in (BOB, CAROL)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert {r.status_code for r in out.values()} == {200}, {k: v.text for k, v in out.items()}
    s = _src(new["_id"])
    assert sorted(a["by"] for a in s["approvals"]) == [BOB, CAROL]
    assert s["decision_status"] == "active"
    assert _src(old["_id"])["decision_status"] == "superseded"
    rows = _rows(new["_id"])
    assert max(r["detail"]["attempt"] for r in rows if r["intent"] == "decision_approve") == 2, \
        "no seq_no conflict happened: the race was not exercised"
    assert len([r for r in rows if r["intent"] == "decision_activate"]) == 1
    assert len([r for r in _rows(old["_id"]) if r["intent"] == "decision_supersede"]) == 1
    assert sum(1 for r in out.values() if r.json()["activated"]) == 1


# ---- case 14 ----

def test_case14_every_transition_is_audited_before_it_takes_effect(monkeypatch):
    real_cas = governance._cas
    seen = []

    def checking(doc, patch):
        seen.append((patch.get("decision_status"), {r["intent"] for r in _rows(doc["id"])}))
        return real_cas(doc, patch)

    monkeypatch.setattr(governance, "_cas", checking)

    # propose: the row exists, keyed by the doc group, and no passage had been written yet
    group = f"gov-grp-{uuid.uuid4().hex[:8]}"
    real_prop = governance.audit_propose

    def propose_first(owner, grp, text_hash, override_of):
        n = es("POST", f"/{idx('semantic')}/_count", {"query": {"term": {"doc_group": grp}}})
        assert n["count"] == 0, "state was written before the propose audit row"
        return real_prop(owner, grp, text_hash, override_of)

    monkeypatch.setattr(governance, "audit_propose", propose_first)
    p = _propose(doc_group=group)
    rows = _rows(group)
    assert [r["intent"] for r in rows] == ["decision_propose"]
    assert rows[0]["owner_id"] == ALICE and rows[0]["detail"]["text_hash"] == _src(p["_id"])["text_hash"]

    # approve + activate
    assert _approve(BOB, p["_id"]).status_code == 200
    status, intents = seen[-1]
    assert status == "active" and {"decision_approve", "decision_activate"} <= intents
    rows = _rows(p["_id"])
    approve = next(r for r in rows if r["intent"] == "decision_approve")
    assert approve["owner_id"] == BOB and approve["detail"]["text_hash"] == _src(p["_id"])["text_hash"]

    # veto reject, withdraw, expire
    monkeypatch.setenv("AMES_GOV_VETO_OWNERS", CAROL)
    r = _propose()
    assert _reject(CAROL, r["_id"]).status_code == 200
    assert seen[-1][0] == "rejected" and "decision_reject" in seen[-1][1]
    w = _propose()
    assert client.post("/memory/delete", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": w["_id"]}).status_code == 200
    assert seen[-1][0] == "rejected" and "decision_withdraw" in seen[-1][1]
    e = _propose()
    governance.expire_stale_proposals(now=time.time() + 400 * 86400)
    assert "decision_expire" in {r["intent"] for r in _rows(e["_id"])}
    assert [s for s, i in seen if s == "expired"] and "decision_expire" in seen[-1][1]


def test_case14_a_transition_that_cannot_be_audited_does_not_happen(monkeypatch):
    p = _propose()

    def broken(intent, actor, doc_id, detail):
        raise RuntimeError("audit store down")

    monkeypatch.setattr(governance, "_audit", broken)
    with pytest.raises(RuntimeError):
        governance.approve(BOB, p["_id"])
    monkeypatch.undo()
    s = _src(p["_id"])
    assert s["decision_status"] == "proposed" and s["approvals"] == []


# ---- case 15 ----

def test_case15_backfill_marks_existing_team_decisions_active_and_is_idempotent(monkeypatch):
    def seed(visibility, tags, text):
        doc = {"kind": "semantic", "owner_id": ALICE, "visibility": visibility, "text": text,
               "text_hash": memory.text_hash(text), "entities": extract_entities(text),
               "occurred_at": "2026-01-01T00:00:00Z", "active": True, "tags": tags,
               "embedding": embeddings.embed([text])[0]}
        return es("POST", f"/{idx('semantic')}/_doc?refresh=true", doc)["_id"]

    t = _tok()
    legacy = [seed("team", [DECISION], f"legacy decision {t} one"),
              seed("team", [DECISION], f"legacy decision {t} two")]
    private = seed("private", [DECISION], f"legacy private decision {t}")
    howto = seed("team", ["kind:howto"], f"legacy howto {t}")

    assert client.post("/admin/governance/backfill").status_code == 403
    monkeypatch.setenv("AMES_ADMIN_TOKEN", "backfill-token")
    assert client.post("/admin/governance/backfill",
                       headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.post("/admin/governance/backfill", headers=H(APPROVER[BOB])).status_code == 403
    r = client.post("/admin/governance/backfill", headers={"X-Admin-Token": "backfill-token"})
    assert r.status_code == 200 and r.json()["backfilled"] >= 2, r.text
    for doc_id in legacy:
        s = _src(doc_id)
        assert s["decision_status"] == "active" and s["approved_via"] == "backfill"
        assert s["decided_at"] and s["active"] is True
    assert "decision_status" not in _src(private) and "decision_status" not in _src(howto)

    again = client.post("/admin/governance/backfill", headers={"X-Admin-Token": "backfill-token"})
    assert again.status_code == 200 and again.json()["backfilled"] == 0
    # a backfilled decision is in force: read-only, and overridable
    assert client.post("/memory/delete", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": legacy[0]}).status_code == 409
    o = _propose(BOB, supersedes_decision=legacy[0])
    assert o["approvals_needed"] == 2


def test_case15_pre_governance_team_decisions_are_never_consolidated_before_backfill():
    text = f"unbackfilled decision {_tok()} keep the nightly export job on the old host"
    ids = []
    for t in (text, text + " ok"):
        doc = {"kind": "semantic", "owner_id": ALICE, "visibility": "team", "text": t,
               "text_hash": memory.text_hash(t), "entities": extract_entities(t),
               "occurred_at": "2026-01-01T00:00:00Z", "active": True, "tags": [DECISION],
               "embedding": embeddings.embed([t])[0]}
        ids.append(es("POST", f"/{idx('semantic')}/_doc?refresh=true", doc)["_id"])
    memory.consolidate(ALICE, dry_run=False)
    for i in ids:
        assert _src(i)["active"] is True and "superseded_by" not in _src(i)


def test_governance_mapping_fields_exist():
    props = es("GET", f"/{idx('semantic')}/_mapping")[idx("semantic")]["mappings"]["properties"]
    for f in ("decision_status", "proposed_by", "proposal_for", "approvals", "objections",
              "decided_at", "proposal_expires_at", "approved_via"):
        assert f in props, f
    assert tagfilter.TAG_FIELD in props
