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
AGENT = {p: auth.create_key(p, "member", person=p) for p in PEOPLE}          # what an agent holds
APPROVER = {p: auth.create_key(p, "approver", person=p) for p in PEOPLE}     # minted for the same person
WIPE = list(PEOPLE) + ["gov_owner", "gov_alice2", "gov_alice3", "gov_legacy", "gov_other", "system", "admin"]
OWNER_KEY = auth.create_key("gov_owner", "owner", person="gov_owner")
H = lambda k: {"X-API-Key": k}  # noqa: E731
DECISION = "kind:decision"
SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")

for _name in (idx("semantic"), f"{PREFIX}am_mutations"):
    try:
        es("POST", f"/{_name}/_delete_by_query?refresh=true&ignore_unavailable=true",
           {"query": {"terms": {"owner_id": WIPE}}})
    except RuntimeError:
        pass


def _wipe_owners():
    es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true&ignore_unavailable=true&conflicts=proceed",
       {"query": {"terms": {"owner_id": WIPE}}})


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
    r = client.post(f"/mcp/tools/{tool}", headers=H(AGENT[BOB]), json={"doc_id": "x"})
    assert r.status_code == 404
    # an approver key never reaches the tools surface at all (case 22)
    assert client.post(f"/mcp/tools/{tool}", headers=H(APPROVER[BOB]),
                       json={"doc_id": "x"}).status_code == 403


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

    def untouched(doc_id, status):
        s = _src(doc_id)
        assert s["active"] is True and s.get("decision_status") == status, (doc_id, s)
        assert "superseded_by" not in s and "supersedes" not in s, (doc_id, s)

    def plain_side_clean(doc_id, governed_ids):
        s = _src(doc_id)
        assert "decision_status" not in s and s["active"] is True
        assert not set(s.get("supersedes") or []) & set(governed_ids), s
        assert s.get("superseded_by") not in governed_ids, s

    untouched(d["_id"], "active")                     # the governed side of the pair
    plain_side_clean(howto["_id"], [d["_id"]])        # the plain side of the pair
    # and a governed write never links a plain neighbour either: both sides stay clean
    p = _propose(token=t, extra="rotate the signing material every quarter without exception now")
    untouched(p["_id"], "proposed")
    untouched(d["_id"], "active")
    plain_side_clean(howto["_id"], [d["_id"], p["_id"]])


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
    groups = []
    real_prop = governance.audit_propose

    def propose_first(owner, grp, *a, **k):
        n = es("POST", f"/{idx('semantic')}/_count", {"query": {"term": {"doc_group": grp}}})
        assert n["count"] == 0, "state was written before the propose audit row"
        groups.append(grp)
        return real_prop(owner, grp, *a, **k)

    monkeypatch.setattr(governance, "audit_propose", propose_first)
    p = _propose()
    group = p["doc_group"]
    assert groups == [group]
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


# ---- review round 2: cases 16-22 + findings 9-13 ----

def _stray(owner, group, text, **extra):
    """A doc of `owner` in `group`, written straight to the index (what a caller-chosen doc_group
    or pre-governance data can produce)."""
    doc = {"kind": "semantic", "owner_id": owner, "visibility": "private", "text": text,
           "text_hash": memory.text_hash(text), "doc_group": group, "passage_index": 0,
           "passages_total": 1, "entities": extract_entities(text),
           "occurred_at": "2026-01-01T00:00:00Z", "active": True, **extra}
    return es("POST", f"/{idx('semantic')}/_doc?refresh=true", doc)["_id"]


def _retain_private(who, text, **body):
    r = client.post("/memory/retain", headers=H(AGENT[who]), json={
        "kind": "semantic", "text": text, "extract_facts": False, **body})
    assert r.status_code == 200, r.text
    return r.json()


# ---- case 16 ----

def test_case16_approving_a_proposal_never_changes_a_sibling_in_a_caller_chosen_group():
    group = f"shared-{uuid.uuid4().hex[:8]}"
    sibling = _retain_private(ALICE, f"plain note {_tok()}", doc_group=group)
    assert sibling["doc_group"] == group
    p = _propose(ALICE, doc_group=group)            # same caller-chosen group
    assert p["doc_group"] != group, "a governed write gets its own server-generated group"
    assert _approve(BOB, p["_id"]).status_code == 200
    s = _src(sibling["_id"])
    assert "decision_status" not in s and "proposed_by" not in s and s["active"] is True


def test_case16_mirror_and_load_are_scoped_to_the_approved_text():
    p = _propose()
    stray = _stray(ALICE, p["doc_group"], f"stray passage {_tok()}", passage_index=1)
    assert _approve(BOB, p["_id"]).status_code == 200
    assert "decision_status" not in _src(stray)
    reject_target = _propose()
    stray2 = _stray(ALICE, reject_target["doc_group"], f"stray two {_tok()}", passage_index=1)
    assert _reject(BOB, reject_target["_id"]).status_code == 200
    assert _src(stray2)["active"] is True and "decision_status" not in _src(stray2)
    # a stray doc that sits at passage 0 must never be mistaken for the head
    p2 = _propose()
    _stray(ALICE, p2["doc_group"], f"stray head {_tok()}", passage_index=0)
    mid = es("POST", f"/{idx('semantic')}/_search", {"size": 5, "query": {"bool": {"filter": [
        {"term": {"doc_group": p2["doc_group"]}}, {"term": {"decision_status": "proposed"}}]}}})
    assert [h["_id"] for h in mid["hits"]["hits"]] == [p2["_id"]]


def test_case16_promoting_a_private_decision_gets_its_own_group():
    group = f"shared-{uuid.uuid4().hex[:8]}"
    private = _retain_private(ALICE, f"private decision {_tok()}", doc_group=group,
                              tags=[DECISION])
    other = _retain_private(ALICE, f"private neighbour {_tok()}", doc_group=group)
    r = client.post("/memory/promote", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": private["_id"], "to_visibility": "team"})
    assert r.status_code == 200, r.text
    copy = r.json()["promoted_id"]
    assert _src(copy)["doc_group"] != group
    assert _rows(_src(copy)["doc_group"])[0]["intent"] == "decision_propose"
    assert _approve(BOB, copy).status_code == 200
    for doc_id in (private["_id"], other["_id"]):
        assert "decision_status" not in _src(doc_id)


# ---- case 17 ----

def test_case17_an_active_decision_cannot_be_changed_through_a_sibling_doc():
    d = _active()
    sibling = _retain_private(ALICE, f"sibling {_tok()}", doc_group=d["doc_group"])
    h = H(AGENT[ALICE])
    u = client.post("/memory/update", headers=h, json={
        "kind": "semantic", "doc_id": sibling["_id"], "text": f"overwrite {_tok()}"})
    x = client.post("/memory/delete", headers=h, json={"kind": "semantic", "doc_id": sibling["_id"]})
    x2 = client.delete(f"/memory/semantic/{sibling['_id']}", headers=h)
    t = client.post("/mcp/tools/delete", headers=h, json={"kind": "semantic", "doc_id": sibling["_id"]})
    p = client.post("/memory/promote", headers=h, json={
        "kind": "semantic", "doc_id": sibling["_id"], "to_visibility": "common"})
    for r in (u, x, x2, t, p):
        assert r.status_code == 409, r.text
    s = _src(d["_id"])
    assert s["decision_status"] == "active" and s["active"] is True
    assert _src(sibling["_id"])["active"] is True


def test_case17_a_proposal_cannot_be_hard_deleted_through_a_sibling_doc():
    p = _propose()
    sibling = _retain_private(ALICE, f"sibling {_tok()}", doc_group=p["doc_group"])
    for path, body in (("/memory/delete", {}), ("/memory/update", {"text": f"x {_tok()}"})):
        r = client.post(path, headers=H(AGENT[ALICE]), json={
            "kind": "semantic", "doc_id": sibling["_id"], **body})
        assert r.status_code == 409, r.text
    assert _src(p["_id"])["decision_status"] == "proposed"


# ---- case 18 ----

def test_case18_edit_that_collides_with_another_team_doc_leaves_the_proposal_intact():
    mine = _propose(ALICE)
    theirs = _propose(BOB)
    r = client.post("/memory/update", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": mine["_id"], "text": _src(theirs["_id"])["text"]})
    assert r.status_code in (409, 422), r.text
    s = _src(mine["_id"])
    assert s["decision_status"] == "proposed" and s["text_hash"] == mine["text_hash"]
    assert s["doc_group"] == mine["doc_group"]


def test_case18_edit_that_fails_midway_leaves_the_original_proposal_intact(monkeypatch):
    mine = _propose(ALICE)
    real = embeddings.embed_safe
    calls = []

    def flaky(texts):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("embedder down")
        return real(texts)

    monkeypatch.setattr(embeddings, "embed_safe", flaky)
    long_text = f"decision {mine['token']} rewritten. " + ("Sentence about the queue. " * 120)
    with pytest.raises(RuntimeError):
        memory.update(ALICE, "semantic", mine["_id"], text=long_text, person=ALICE)
    monkeypatch.undo()
    s = _src(mine["_id"])
    assert s["decision_status"] == "proposed" and s["text_hash"] == mine["text_hash"]
    heads = es("POST", f"/{idx('semantic')}/_count", {"query": {"bool": {"filter": [
        {"term": {"doc_group": mine["doc_group"]}}]}}})
    assert heads["count"] == 1, "a failed edit left partial passages behind"
    assert _approve(BOB, mine["_id"]).status_code == 200


# ---- case 19 ----

def test_case19_editing_a_proposal_keeps_its_original_deadline():
    p = _propose()
    fixed = "2031-01-01T00:00:00Z"
    es("POST", f"/{idx('semantic')}/_update/{p['_id']}?refresh=true",
       {"doc": {"proposal_expires_at": fixed}})
    r = client.post("/memory/update", headers=H(AGENT[ALICE]), json={
        "kind": "semantic", "doc_id": p["_id"], "text": f"decision {p['token']} reworded"})
    assert r.status_code == 200, r.text
    assert _src(r.json()["_id"])["proposal_expires_at"] == fixed


# ---- case 20 ----

def test_case20_two_overrides_of_one_decision_cannot_both_activate(monkeypatch):
    old = _active()
    n1 = _propose(ALICE, supersedes_decision=old["_id"])
    n2 = _propose(ALICE, supersedes_decision=old["_id"])
    assert _approve(BOB, n1["_id"]).status_code == 200
    assert _approve(BOB, n2["_id"]).status_code == 200
    barrier = threading.Barrier(2, timeout=30)
    gated_threads, lock = set(), threading.Lock()
    real_audit = governance._audit

    def gated(intent, actor, doc_id, detail):
        # both overrides have read the target as active before either one moves it
        if intent == "decision_supersede" and doc_id == old["_id"]:
            with lock:
                first = threading.get_ident() not in gated_threads
                gated_threads.add(threading.get_ident())
            if first:
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    pass
        return real_audit(intent, actor, doc_id, detail)

    monkeypatch.setattr(governance, "_audit", gated)
    out = {}

    def go(who, doc):
        out[doc] = _approve(who, doc)

    threads = [threading.Thread(target=go, args=(CAROL, n1["_id"])),
               threading.Thread(target=go, args=(DAVE, n2["_id"]))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    monkeypatch.undo()
    codes = sorted(r.status_code for r in out.values())
    assert codes == [200, 409], {k: v.text for k, v in out.items()}
    states = {n["_id"]: _src(n["_id"]) for n in (n1, n2)}
    winners = [i for i, s in states.items() if s["decision_status"] == "active"]
    assert len(winners) == 1, {i: s["decision_status"] for i, s in states.items()}
    loser = next(i for i in states if i != winners[0])
    assert states[loser]["decision_status"] == "proposed"
    assert "override target is no longer an active decision" in out[loser].text
    o = _src(old["_id"])
    assert o["decision_status"] == "superseded" and o["superseded_by"] == winners[0]
    assert states[winners[0]]["supersedes"] == [old["_id"]]
    assert "supersedes" not in states[loser]
    assert _approve(DAVE if loser == n2["_id"] else CAROL, loser).status_code == 409


def _legacy_key(owner, role="member"):
    """A key minted before `person` was required: no person on the entry."""
    import hashlib
    token = "ame_" + uuid.uuid4().hex * 2
    with auth._guard():
        keys = auth._load()
        keys[hashlib.sha256(token.encode()).hexdigest()] = {"owner_id": owner, "role": role}
        auth._save(keys)
    return token


# ---- case 21 ----

def test_case21_two_keys_of_one_person_count_as_one_and_cannot_self_approve():
    second_alice = auth.create_key("gov_alice2", "approver", person=ALICE)
    second_bob = auth.create_key("gov_alice3", "approver", person=BOB)
    old = _active()
    mine = _propose(ALICE)
    r = client.post("/governance/approve", headers=H(second_alice), json={"doc_id": mine["_id"]})
    assert r.status_code == 403, r.text
    assert _src(mine["_id"])["decision_status"] == "proposed"

    new = _propose(ALICE, supersedes_decision=old["_id"])
    assert _approve(BOB, new["_id"]).status_code == 200
    again = client.post("/governance/approve", headers=H(second_bob), json={"doc_id": new["_id"]})
    assert again.status_code == 409, again.text
    s = _src(new["_id"])
    assert s["decision_status"] == "proposed" and len(s["approvals"]) == 1
    assert _approve(CAROL, new["_id"]).json()["decision_status"] == "active"


def test_case21_an_approver_key_without_a_person_cannot_approve():
    anonymous = _legacy_key("gov_alice3", "approver")      # minted before person was required
    p = _propose()
    r = client.post("/governance/approve", headers=H(anonymous), json={"doc_id": p["_id"]})
    assert r.status_code == 403, r.text
    r = client.post("/governance/reject", headers=H(anonymous),
                    json={"doc_id": p["_id"], "reason": "x"})
    assert r.status_code == 403, r.text
    assert _src(p["_id"])["approvals"] == []


def test_case21_the_author_is_the_person_not_the_key_owner():
    # alice's agent key belongs to another owner id but carries her person
    other = auth.create_key("gov_alice2", "member", person=ALICE)
    r = client.post("/memory/retain", headers=H(other), json={
        "kind": "semantic", "text": f"decision {_tok()} by the other key", "visibility": "team",
        "tags": [DECISION], "extract_facts": False})
    assert r.status_code == 200, r.text
    assert _src(r.json()["_id"])["proposed_by"] == ALICE
    assert _approve(ALICE, r.json()["_id"]).status_code == 403


def test_case21_key_roles_are_an_allowlist_and_person_is_set_at_mint_time(monkeypatch):
    with pytest.raises(ValueError):
        auth.create_key("gov_alice2", "root")
    monkeypatch.setenv("AMES_ADMIN_TOKEN", "mint-token")
    bad = client.post("/admin/keys?owner_id=gov_alice2&role=root", headers={"X-Admin-Token": "mint-token"})
    assert bad.status_code == 422, bad.text
    ok = client.post(f"/admin/keys?owner_id=gov_alice2&role=approver&person={BOB}",
                     headers={"X-Admin-Token": "mint-token"})
    assert ok.status_code == 200, ok.text
    assert auth.owner_of(ok.json()["api_key"]) == {
        "owner_id": "gov_alice2", "role": "approver", "person": BOB}
    plain = client.post("/admin/keys?owner_id=gov_alice2", headers={"X-Admin-Token": "mint-token"})
    assert plain.status_code == 422, plain.text                  # person is required (round 2 finding 3)


# ---- case 22 ----

def test_case22_an_approver_key_is_refused_everywhere_but_governance():
    p = _propose()
    h = H(APPROVER[BOB])
    mine = _retain_private(BOB, f"bob note {_tok()}")
    calls = [
        client.post("/memory/retain", headers=h, json={"kind": "semantic", "text": f"x {_tok()}"}),
        client.post("/memory/update", headers=h, json={
            "kind": "semantic", "doc_id": mine["_id"], "text": f"y {_tok()}"}),
        client.post("/memory/delete", headers=h, json={"kind": "semantic", "doc_id": mine["_id"]}),
        client.post("/memory/recall", headers=h, json={"query": p["token"]}),
        client.post("/memory/promote", headers=h, json={
            "kind": "semantic", "doc_id": mine["_id"], "to_visibility": "team"}),
        client.post("/mcp/tools/retain", headers=h, json={"kind": "semantic", "text": f"z {_tok()}"}),
        client.post("/mcp/tools/recall", headers=h, json={"query": p["token"]}),
        client.post("/mcp/tools/delete", headers=h, json={"kind": "semantic", "doc_id": mine["_id"]}),
    ]
    assert [c.status_code for c in calls] == [403] * len(calls), [c.text for c in calls]
    assert _src(mine["_id"])["active"] is True
    # the governance surface stays open to it
    assert client.get("/governance/proposals", headers=h).status_code == 200
    assert _approve(BOB, p["_id"]).status_code == 200


# ---- finding 8: the approval CLI ----

def test_cli_refuses_to_send_the_key_over_plain_http_to_a_remote_host():
    cli = _load_script("ames_approve")
    for url in ("http://example.com:8123", "http://localhost.example.com", "http://10.0.0.5:8123",
                "ftp://localhost", "http://", "localhost:8123"):
        with pytest.raises(SystemExit):
            cli.check_url(url)
    for url in ("http://localhost:8123", "http://127.0.0.1:8123", "http://[::1]:8123",
                "https://example.com", "https://example.com:8443/base"):
        assert cli.check_url(url)


def test_cli_confirms_the_target_before_prompting_and_refuses_without_a_tty(monkeypatch, capsys):
    cli = _load_script("ames_approve")

    class NoTty:
        def isatty(self):
            return False

    class Tty:
        def isatty(self):
            return True

    with pytest.raises(SystemExit):
        cli.confirm_target("https://example.com", stdin=NoTty(),
                           ask=lambda _p: pytest.fail("asked without a TTY"))
    with pytest.raises(SystemExit):
        cli.confirm_target("https://example.com", stdin=Tty(), ask=lambda _p: "n")
    assert cli.confirm_target("https://example.com", stdin=Tty(), ask=lambda _p: "y") is None
    assert "example.com" in capsys.readouterr().out
    # refusal happens before the key is read, with no TTY: main() never reaches the prompt
    monkeypatch.setattr(cli.sys, "stdin", NoTty())
    monkeypatch.setattr(cli.getpass, "getpass", lambda _p: pytest.fail("prompted for the key"))
    for url in ("http://example.com:8123", "http://localhost:8123"):
        with pytest.raises(SystemExit):
            cli.main(["--url", url, "list"])


# ---- finding 9: governance mapping migration fails closed ----

def _mapping_probe(monkeypatch, mutate=None):
    """store.es with the governance PUT failing and the mapping GET optionally altered by
    `mutate(properties)`. Returns the list of governance PUT attempts."""
    import copy
    from app import store
    real = store.es
    puts = []

    def probe(method, path, body=None, *a, **k):
        if method == "PUT" and path.endswith("/_mapping") and body and "decision_status" in body.get("properties", {}):
            puts.append(path)
            raise RuntimeError("ES PUT mapping -> 403: no put_mapping privilege")
        out = real(method, path, body, *a, **k)
        if mutate and method == "GET" and path.endswith("/_mapping"):
            out = copy.deepcopy(out)
            for v in out.values():
                mutate(v["mappings"]["properties"])
        return out

    monkeypatch.setattr(store, "es", probe)
    monkeypatch.setattr(store, "_indices_ready", False)
    return puts


@pytest.mark.parametrize("mutate", [
    lambda props: props.pop("decision_status"),
    lambda props: props.__setitem__("decision_status", {"type": "text"}),
    lambda props: props["approvals"]["properties"].pop("by"),
], ids=["missing", "wrong-type", "nested-missing"])
def test_governance_mapping_failure_fails_closed(monkeypatch, mutate):
    from app import store
    _mapping_probe(monkeypatch, mutate)
    with pytest.raises(RuntimeError):
        store.ensure_indices()
    assert store._indices_ready is False
    with pytest.raises(RuntimeError):                  # so retain / recall fail too, not just startup
        memory.retain(ALICE, "semantic", f"fail closed {_tok()}")


def test_finding6_already_mapped_fields_skip_the_put(monkeypatch):
    from app import store
    puts = _mapping_probe(monkeypatch)
    store.ensure_indices()                             # the PUT would fail: it must not be needed
    assert puts == [] and store._indices_ready is True


# ---- finding 10: audit rows of lost compare-and-set attempts ----

def test_finding10_audit_rows_of_a_lost_cas_are_marked_conflict(monkeypatch):
    old = _active()
    new = _propose(ALICE, supersedes_decision=old["_id"])
    barrier = threading.Barrier(2, timeout=30)
    real_audit = governance._audit

    def gated(intent, actor, doc_id, detail):
        if intent == "decision_approve" and detail.get("attempt") == 1:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return real_audit(intent, actor, doc_id, detail)

    monkeypatch.setattr(governance, "_audit", gated)
    out = {}
    threads = [threading.Thread(target=lambda w=w: out.setdefault(w, _approve(w, new["_id"])))
               for w in (BOB, CAROL)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    monkeypatch.undo()
    assert {r.status_code for r in out.values()} == {200}
    rows = [r for r in _rows(new["_id"]) if r["intent"] == "decision_approve"]
    lost = [r for r in rows if r["detail"].get("outcome") == "conflict"]
    won = [r for r in rows if r["detail"].get("outcome") != "conflict"]
    assert len(lost) == 1 and lost[0]["detail"]["attempt"] == 1, rows
    assert len(won) == 2, rows                        # the winner's attempt 1 and the retry
    assert len(_src(new["_id"])["approvals"]) == 2


# ---- finding 11: backfill is compare-and-set and records an author ----

def test_finding11_backfill_does_not_overwrite_a_concurrent_transition(monkeypatch):
    t = _tok()
    text = f"legacy decision {t} racing"
    doc_id = _stray(ALICE, f"dg_{uuid.uuid4().hex}", text, visibility="team", tags=[DECISION])
    real_audit = governance._audit

    def interfering(intent, actor, did, detail):
        if intent == "decision_backfill" and did == doc_id:
            es("POST", f"/{idx('semantic')}/_update/{doc_id}?refresh=true",
               {"doc": {"decision_status": "rejected", "active": False}})
        return real_audit(intent, actor, did, detail)

    monkeypatch.setattr(governance, "_audit", interfering)
    governance.backfill()
    monkeypatch.undo()
    s = _src(doc_id)
    assert s["decision_status"] == "rejected" and s["active"] is False


def test_finding11_backfilled_decisions_record_proposed_by():
    doc_id = _stray(ALICE, f"dg_{uuid.uuid4().hex}", f"legacy decision {_tok()}",
                    visibility="team", tags=[DECISION])
    governance.backfill()
    s = _src(doc_id)
    assert s["decision_status"] == "active" and s["proposed_by"] == ALICE
    assert governance.backfill()["backfilled"] == 0 or _src(doc_id)["decision_status"] == "active"


# ---- finding 12: the expiry pass reads every page ----

def test_finding12_expiry_pass_paginates_past_one_page():
    n = governance.PAGE + 5
    ids = []
    for i in range(n):
        text = f"bulk proposal {_tok()} {i}"
        ids.append(_stray(ALICE, f"dg_{uuid.uuid4().hex}", text, visibility="team",
                          tags=[DECISION], decision_status="proposed", proposed_by=ALICE,
                          approvals=[], proposal_expires_at="2000-01-01T00:00:00Z"))
    out = governance.expire_stale_proposals()
    assert out["expired"] >= n
    left = es("POST", f"/{idx('semantic')}/_count", {"query": {"bool": {"filter": [
        {"terms": {"_id": ids}}, {"term": {"decision_status": "proposed"}}]}}})["count"]
    assert left == 0


# ---- review round 3 (second review): claim release, shared claims, author identity ----

def _override(approvals=2, token=None):
    """(old, new): an active decision and an open override proposal for it."""
    old = _active()
    new = _propose(ALICE, supersedes_decision=old["_id"], token=token)
    return old, new


def _orphan_claim(old, new):
    """The state a crash leaves: the target claimed for the override, the override never activated."""
    governance._claim_target(BOB, new["_id"], old["_id"])
    assert _src(old["_id"])["decision_status"] == "superseded"


def _far(days):
    return governance._iso(time.time() + days * 86400)


def test_round3_sweep_never_restores_a_decision_a_later_override_replaced():
    a = _active()
    b = _propose(ALICE, supersedes_decision=a["_id"])
    assert _approve(BOB, b["_id"]).status_code == 200 and _approve(CAROL, b["_id"]).status_code == 200
    c = _propose(ALICE, supersedes_decision=b["_id"])
    assert _approve(BOB, c["_id"]).status_code == 200 and _approve(CAROL, c["_id"]).status_code == 200
    assert _src(b["_id"])["decision_status"] == "superseded"
    governance._release_orphaned_claims(_far(0.01))                 # past the grace, inside the old 24h window
    assert _src(a["_id"])["decision_status"] == "superseded"        # B is superseded, not an orphan
    assert _src(b["_id"])["decision_status"] == "superseded"
    assert _src(c["_id"])["decision_status"] == "active"


def test_round3_sweep_releases_a_real_orphan_after_the_grace_period_only():
    old, new = _override()
    _orphan_claim(old, new)
    governance._release_orphaned_claims(governance._iso(time.time()))           # inside the grace
    assert _src(old["_id"])["decision_status"] == "superseded"
    assert governance._release_orphaned_claims(_far(0.01)) >= 1
    s = _src(old["_id"])
    assert s["decision_status"] == "active" and not s.get("superseded_by")
    assert any(r["intent"] == "decision_restore" for r in _rows(old["_id"]))


def test_round3_sweep_has_no_24h_upper_bound():
    old, new = _override()
    _orphan_claim(old, new)
    governance._release_orphaned_claims(_far(3))                 # the claim is 3 days old
    assert _src(old["_id"])["decision_status"] == "active"


def test_round3_sweep_reads_every_page():
    succ = _stray(ALICE, f"dg_{uuid.uuid4().hex}", f"orphan successor {_tok()}", visibility="team",
                  tags=[DECISION], decision_status="proposed", proposed_by=ALICE, approvals=[])
    n = governance.PAGE + 5
    ids = [_stray(ALICE, f"dg_{uuid.uuid4().hex}", f"orphan target {_tok()} {i}", visibility="team",
                  tags=[DECISION], decision_status="superseded", superseded_by=succ, active=False,
                  decided_at="2000-01-01T00:00:00Z") for i in range(n)]
    assert governance._release_orphaned_claims(_far(1)) >= n
    left = es("POST", f"/{idx('semantic')}/_count", {"query": {"bool": {"filter": [
        {"terms": {"_id": ids}}, {"term": {"decision_status": "superseded"}}]}}})["count"]
    assert left == 0


def test_round3_a_vetoed_override_gives_its_claim_back(monkeypatch):
    old, new = _override()
    _orphan_claim(old, new)
    monkeypatch.setenv("AMES_GOV_VETO_OWNERS", CAROL)
    assert _reject(CAROL, new["_id"]).json()["decision_status"] == "rejected"
    assert _src(old["_id"])["decision_status"] == "active"


def test_round3_a_withdrawn_override_gives_its_claim_back():
    old, new = _override()
    _orphan_claim(old, new)
    r = client.post("/memory/delete", headers=H(AGENT[ALICE]), json={"kind": "semantic", "doc_id": new["_id"]})
    assert r.status_code == 200 and r.json()["withdrawn"] is True, r.text
    assert _src(old["_id"])["decision_status"] == "active"


def test_round3_an_expired_override_gives_its_claim_back():
    old, new = _override()
    _orphan_claim(old, new)
    es("POST", f"/{idx('semantic')}/_update/{new['_id']}?refresh=true",
       {"doc": {"proposal_expires_at": "2000-01-01T00:00:00Z"}})
    governance.expire_stale_proposals()
    assert _src(new["_id"])["decision_status"] == "expired"
    assert _src(old["_id"])["decision_status"] == "active"


def test_round3_a_released_shared_claim_is_taken_again_by_the_approver_who_succeeds():
    old, new = _override()
    assert _approve(BOB, new["_id"]).status_code == 200            # one more approval activates it
    d_claimed, c_claimed, d_released = (threading.Event() for _ in range(3))
    real_audit, real_release = governance._audit, governance._release_target

    def gated(intent, actor, doc_id, detail):
        if intent == "decision_activate" and actor == DAVE:         # D: claimed, proposal write fails
            d_claimed.set()
            c_claimed.wait(30)
            raise RuntimeError("proposal write failed")
        if intent == "decision_activate" and actor == CAROL:        # C: shares D's claim
            c_claimed.set()
            d_released.wait(30)
        return real_audit(intent, actor, doc_id, detail)

    def release(*a, **k):
        try:
            return real_release(*a, **k)
        finally:
            d_released.set()

    governance._audit, governance._release_target = gated, release
    out = {}

    def go(who):
        try:
            out[who] = governance.approve(who, new["_id"], None, who)
        except Exception as exc:  # noqa: BLE001
            out[who] = exc

    try:
        td = threading.Thread(target=go, args=(DAVE,))
        td.start()
        assert d_claimed.wait(30)
        tc = threading.Thread(target=go, args=(CAROL,))
        tc.start()
        td.join(60)
        tc.join(60)
    finally:
        governance._audit, governance._release_target = real_audit, real_release
    assert isinstance(out[DAVE], RuntimeError) and out[CAROL]["activated"] is True, out
    assert _src(new["_id"])["decision_status"] == "active"
    o = _src(old["_id"])
    assert o["decision_status"] == "superseded" and o["superseded_by"] == new["_id"], o
    assert _src(new["_id"])["supersedes"] == [old["_id"]]


def _force_override_state(new_id, old_id, **fields):
    es("POST", f"/{idx('semantic')}/_update/{old_id}?refresh=true", {"doc": fields})


def test_round3_worker_reconciles_an_active_override_whose_target_is_not_superseded():
    old, new = _override()
    assert _approve(BOB, new["_id"]).status_code == 200 and _approve(CAROL, new["_id"]).status_code == 200
    assert _src(new["_id"])["supersedes"] == [old["_id"]]            # linked, target superseded
    _force_override_state(new["_id"], old["_id"], decision_status="active", active=True, superseded_by=None)
    out = governance.expire_stale_proposals()
    assert out["completed_overrides"] >= 1
    o = _src(old["_id"])
    assert o["decision_status"] == "superseded" and o["superseded_by"] == new["_id"]


def test_round3_a_stuck_override_is_flagged_once_and_not_retried(caplog):
    old, new = _override()
    assert _approve(BOB, new["_id"]).status_code == 200 and _approve(CAROL, new["_id"]).status_code == 200
    _force_override_state(new["_id"], old["_id"], superseded_by=f"someone-else-{uuid.uuid4().hex}")
    es("POST", f"/{idx('semantic')}/_update/{new['_id']}?refresh=true",
       {"doc": {"supersedes": None}})
    with caplog.at_level("WARNING", logger="ames.governance"):
        out = governance.expire_stale_proposals()
    assert out["override_conflicts"] >= 1
    assert any(new["_id"] in r.getMessage() for r in caplog.records if r.name == "ames.governance")
    s = _src(new["_id"])
    assert s["approved_via"] == governance.CONFLICT_VIA and s["decision_status"] == "active"
    rows = [r for r in _rows(new["_id"]) if r["intent"] == "decision_override_conflict"]
    assert len(rows) == 1 and rows[0]["detail"]["outcome"] == "conflict"
    governance.expire_stale_proposals()                               # a second pass leaves it alone
    assert len([r for r in _rows(new["_id"]) if r["intent"] == "decision_override_conflict"]) == 1


def test_round3_worker_reconcile_reads_every_page():
    n = governance.PAGE + 5
    ids = [_stray(ALICE, f"dg_{uuid.uuid4().hex}", f"stuck override {_tok()} {i}", visibility="team",
                  tags=[DECISION], decision_status="active", proposed_by=ALICE, approvals=[],
                  approved_via="approvals", proposal_for=f"missing-{uuid.uuid4().hex}")
           for i in range(n)]
    governance.expire_stale_proposals()
    flagged = es("POST", f"/{idx('semantic')}/_count", {"query": {"bool": {"filter": [
        {"terms": {"_id": ids}}, {"term": {"approved_via": governance.CONFLICT_VIA}}]}}})["count"]
    assert flagged == n


def test_round3_approver_key_of_the_proposals_owner_is_refused_whatever_its_person():
    legacy = _legacy_key("gov_legacy", "member")
    r = client.post("/memory/retain", headers=H(legacy), json={
        "kind": "semantic", "text": f"decision {_tok()} from a key without a person",
        "visibility": "team", "tags": [DECISION], "extract_facts": False})
    assert r.status_code == 200, r.text
    doc = r.json()["_id"]
    assert _src(doc)["proposed_by"] == "gov_legacy"                   # no person: the owner id
    mine = auth.create_key("gov_legacy", "approver", person=CAROL)    # same owner, another person
    for path, body in (("/governance/approve", {"doc_id": doc}),
                       ("/governance/reject", {"doc_id": doc, "reason": "x"})):
        r = client.post(path, headers=H(mine), json=body)
        assert r.status_code == 403, (path, r.text)
    assert _src(doc)["approvals"] == [] and _src(doc)["decision_status"] == "proposed"
    assert _approve(BOB, doc).status_code == 200                      # a real second person still can


def test_round3_the_person_behind_a_person_less_agent_key_cannot_approve_its_proposals():
    legacy = _legacy_key("gov_other", "member")
    auth.create_key("gov_other", "member", person=ALICE)              # the owner's other key names her
    r = client.post("/memory/retain", headers=H(legacy), json={
        "kind": "semantic", "text": f"decision {_tok()} from the person-less key",
        "visibility": "team", "tags": [DECISION], "extract_facts": False})
    assert r.status_code == 200, r.text
    doc = r.json()["_id"]
    assert _approve(ALICE, doc).status_code == 403
    assert _reject(ALICE, doc).status_code == 403
    assert _approve(BOB, doc).status_code == 200


def test_round3_person_is_required_to_mint_and_can_be_set_on_a_legacy_key(monkeypatch):
    with pytest.raises(ValueError):
        auth.create_key("gov_alice2", "member")
    with pytest.raises(ValueError):
        auth.create_key("gov_alice2", "approver", person="  ")
    monkeypatch.setenv("AMES_ADMIN_TOKEN", "mint-token")
    adm = {"X-Admin-Token": "mint-token"}
    assert client.post("/admin/keys?owner_id=gov_alice2&role=approver", headers=adm).status_code == 422
    legacy = _legacy_key("gov_alice3", "member")
    assert auth.owner_of(legacy) == {"owner_id": "gov_alice3", "role": "member"}     # still works
    assert client.post("/admin/keys/person?owner_id=gov_alice3&person=" + CAROL).status_code == 403
    assert client.post("/admin/keys/person?owner_id=gov_alice3&person=%20", headers=adm).status_code == 422
    r = client.post("/admin/keys/person?owner_id=gov_alice3&person=" + CAROL, headers=adm)
    assert r.status_code == 200 and r.json()["updated"] >= 1, r.text
    assert auth.owner_of(legacy)["person"] == CAROL
    p = client.post("/memory/retain", headers=H(legacy), json={
        "kind": "semantic", "text": f"decision {_tok()} after set-person", "visibility": "team",
        "tags": [DECISION], "extract_facts": False}).json()
    assert _src(p["_id"])["proposed_by"] == CAROL
    assert _approve(CAROL, p["_id"]).status_code == 403
