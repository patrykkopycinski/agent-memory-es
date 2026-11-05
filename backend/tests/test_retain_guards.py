"""P3-3 retain guards: secret-shaped text is refused (422), long text only gets a `too_long` hint,
and /mcp/tools maps ValueError->422 / PermissionError->403 instead of a 200 {error}.

Runs against the throwaway test ES only (tests/_safety.py). Fake credentials are built at runtime
so no secret-looking literal lives in the repo.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest
from fastapi.testclient import TestClient

from app import auth, memory, secrets_guard
from app.main import app
from app.store import es, idx

OWNER = "p3retainguard"
client = TestClient(app)
H = {"X-API-Key": auth.create_key(OWNER)}

FAKE_SK = "sk-" + "x1" * 12
FAKE_GH = "ghp_" + "A1" * 15
FAKE_GH_FINE = "github_" + "pat_" + "B2" * 15
FAKE_KEYWORD = "api_key = " + "k9" * 8


def _post(path, **body):
    body.setdefault("kind", "semantic")
    return client.post(path, headers=H, json=body)


def _clean():
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true", {"query": {"term": {"owner_id": OWNER}}})


_clean()


@pytest.mark.parametrize("secret", [FAKE_SK, FAKE_GH, FAKE_GH_FINE, FAKE_KEYWORD,
                                    "-----BEGIN RSA PRIVATE KEY-----\nabc",
                                    "Authorization: Bearer " + "q7" * 10])
def test_secret_shaped_text_is_422_on_both_retain_surfaces(secret):
    text = f"deploy note: use {secret} for the staging box"
    direct = _post("/memory/retain", text=text)
    assert direct.status_code == 422 and "secret" in direct.json()["detail"], direct.text
    assert secret not in direct.text                  # the error must not echo the credential back
    via_mcp = _post("/mcp/tools/retain", text=text)
    assert via_mcp.status_code == 422 and "secret" in via_mcp.json()["detail"], via_mcp.text


def test_a_normal_fact_is_200_without_a_too_long_hint():
    r = _post("/memory/retain", text="The kibana auth migration finished; token estimator docs live in the wiki.")
    assert r.status_code == 200, r.text
    assert "too_long" not in r.json()
    r2 = _post("/mcp/tools/retain", text="The ferry timetable changed to a 6 o'clock departure.")
    assert r2.status_code == 200 and "too_long" not in r2.json()["result"]


def test_a_long_text_is_accepted_with_a_soft_too_long_hint():
    text = "Fact: " + ("the harbour crane log is kept by the port engineer. " * 40)
    assert len(text) > 1500
    for path in ("/memory/retain", "/mcp/tools/retain"):
        r = _post(path, text=text + path, kind="episodic")
        assert r.status_code == 200, r.text
        body = r.json() if path == "/memory/retain" else r.json()["result"]
        assert body["too_long"] is True and "1500" in body["too_long_hint"]


def test_exactly_1500_chars_has_no_hint_and_1501_has_one():
    base = "limit probe zebra "
    t1500 = (base * 100)[:1500]
    t1501 = (base * 100)[:1500] + "x"
    assert len(t1500) == 1500 and len(t1501) == 1501
    assert "too_long" not in _post("/memory/retain", text=t1500).json()
    assert _post("/memory/retain", text=t1501).json()["too_long"] is True


def test_mcp_tools_maps_value_error_to_422_and_permission_error_to_403():
    bad = _post("/mcp/tools/retain", text="x", kind="nonsense")
    assert bad.status_code == 422 and "kind must be" in bad.json()["detail"]
    mine = _post("/memory/retain", text="permission probe: the shared fact owned by the guard owner")
    other = {"X-API-Key": auth.create_key(OWNER + "other")}
    r = client.post("/mcp/tools/update", headers=other,
                    json={"kind": "semantic", "doc_id": mine.json()["_id"], "text": "hijack"})
    assert r.status_code == 403, r.text


def test_guard_ignores_prose_that_merely_mentions_credential_words():
    for t in ("the auth migration is done", "token usage estimator post", "rotate the password quarterly",
              "SERVICE_LABEL=change-me", "see the Credential Leak Handling doc"):
        assert secrets_guard.find_secret(t) is None, t
    assert memory.RETAIN_SOFT_MAX_CHARS == 1500
