"""Integration tests against live ES (AMES_ES_URL). Run: pytest backend/tests -v"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AMES_API_KEYS_FILE", "/tmp/ames_test_keys.json")
if os.path.exists(os.environ["AMES_API_KEYS_FILE"]):
    os.remove(os.environ["AMES_API_KEYS_FILE"])

from app.main import app  # noqa: E402
from app import auth  # noqa: E402

client = TestClient(app)
ALICE = auth.create_key("alice")
BOB = auth.create_key("bob")
H = lambda k: {"X-API-Key": k}  # noqa: E731


def _retain(key, kind="semantic", text="x", visibility="private"):
    r = client.post("/memory/retain", headers=H(key),
                    json={"kind": kind, "text": text, "visibility": visibility})
    assert r.status_code == 200, r.text
    return r.json()


def test_health():
    assert client.get("/health").json()["ok"]


def test_auth_reject():
    assert client.post("/memory/recall", headers={"X-API-Key": "ame_bogus"},
                       json={"query": "x"}).status_code == 401


def test_private_isolation():
    _retain(ALICE, text="Alice secret gateway token value xyz")
    r = client.post("/memory/recall", headers=H(BOB), json={"query": "secret gateway token"})
    texts = [h["text"] for k in ("fused",) for h in r.json()[k]]
    assert not any("secret" in t.lower() for t in texts), f"leak: {texts}"


def test_own_private_recall():
    _retain(ALICE, text="Alice keeps her deploy password in 1Password vault")
    r = client.post("/memory/recall", headers=H(ALICE), json={"query": "deploy password vault"})
    assert any("1Password" in h["text"] for h in r.json()["fused"])


def test_common_shared():
    _retain(ALICE, text="Team eval convention: suite_sweep.py on Azure VMs always", visibility="common")
    r = client.post("/memory/recall", headers=H(BOB), json={"query": "eval convention suite_sweep Azure"})
    assert any("suite_sweep" in h["text"] for h in r.json()["fused"])


def test_team_shared():
    _retain(ALICE, text="m1max RAM ledger threshold warn at 9450MB", visibility="team")
    r = client.post("/memory/recall", headers=H(BOB), json={"query": "m1max RAM ledger threshold"})
    assert any("9450" in h["text"] for h in r.json()["fused"])


def test_promote_copy_on_write():
    doc = _retain(ALICE, text="Debug trick: PIPESTATUS is bash-only, zsh ships scripts as files")
    r = client.post("/memory/promote", headers=H(ALICE),
                    json={"kind": "semantic", "doc_id": doc["_id"], "to_visibility": "common"})
    assert r.status_code == 200, r.text
    # bob can now see the promoted copy
    r2 = client.post("/memory/recall", headers=H(BOB), json={"query": "PIPESTATUS zsh scripts"})
    assert any("PIPESTATUS" in h["text"] for h in r2.json()["fused"])


def test_promote_only_creator():
    doc = _retain(ALICE, text="alice-only fact for promotion guard test")
    r = client.post("/memory/promote", headers=H(BOB),
                    json={"kind": "semantic", "doc_id": doc["_id"], "to_visibility": "common"})
    assert r.status_code == 403


def test_consolidate_supersession():
    _retain(ALICE, text="Alice uses React for frontend development work")
    _retain(ALICE, text="Alice uses Vue for frontend development work")
    r = client.post("/memory/consolidate", headers=H(ALICE))
    assert r.status_code == 200
    # both variants may remain if below threshold; at least the call works and returns counts
    assert "superseded" in r.json()


def test_reflect_no_llm():
    _retain(ALICE, text="Hermes runs on the local Mac plus remote m1max host")
    r = client.post("/memory/reflect", headers=H(ALICE), json={"question": "Where does Hermes run?"})
    j = r.json()
    assert j["sources"] and j["answer"]


def test_mcp_tool_surface():
    r = client.post("/mcp/tools/retain", headers=H(BOB),
                    json={"kind": "episodic", "text": "coding farm worker checked out PR branch"})
    assert r.status_code == 200 and r.json()["result"]["owner_id"] == "bob"
    r2 = client.post("/mcp/tools/recall", headers=H(BOB), json={"query": "coding farm PR branch"})
    assert any("coding farm" in h["text"] for h in r2.json()["result"]["fused"])


def test_visibility_validation():
    r = client.post("/memory/retain", headers=H(ALICE),
                    json={"kind": "semantic", "text": "x", "visibility": "public"})
    assert r.status_code == 422
