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
from app.store import PREFIX, es, idx

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
    try:                                   # the mutation log outlives runs; a stale row would fake a leak
        es("POST", f"/{PREFIX}am_mutations/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": OWNER}}})
    except RuntimeError:
        pass                               # index not created yet


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


MUST_ACCEPT = [
    "token: gpt-4o-2024-08-06",
    "Tokenizer note — token: cl100k_base is used by gpt-4",
    "judge combo uses model token: deepseek-v3.2-flash",
    "auth: v2-oauth2-flow",
    "oauth: client2-credentials flow",
]


@pytest.mark.parametrize("text", MUST_ACCEPT)
def test_ordinary_prose_with_a_credential_word_and_a_model_or_flow_name_is_accepted(text):
    assert secrets_guard.find_secret(text) is None, text
    r = _post("/memory/retain", text=text + " (accept probe)")
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("text", [
    "api_key = " + "Xk9mQ2" * 4,                     # mixed case + digits, 24 chars
    "password: " + "hunter2" + "Zq8" * 3,
    "secret=" + "a1b2c3d4" * 3,                     # 24 chars, hex-ish
    "token: " + "Zx81" + "Qw" * 6,
    "credentials: " + "Ab3" * 7,
])
def test_real_looking_keyword_secrets_are_still_rejected(text):
    assert secrets_guard.find_secret(text) == "keyword_value", text


def test_keyword_rule_needs_a_letter_and_a_digit_M4():
    # 24 chars of letters only: not a keyword secret (the letter+digit requirement)
    assert secrets_guard.find_secret("secret: " + "abcdefgh" * 3) is None
    # 24 digits only
    assert secrets_guard.find_secret("secret: " + "12345678" * 3) is None


def test_keyword_rule_requires_a_colon_or_equals_separator_M5():
    # redact.ts allows a bare space as the separator; the guard must not, or prose is rejected
    for t in ("the token Ab12Cd34Ef56Gh78Ij90 is mentioned", "auth Zx81QwQwQwQwQwQwQw is a name"):
        assert secrets_guard.find_secret(t) is None, t
    assert secrets_guard.find_secret("token: Zx81QwQwQwQwQwQwQw") == "keyword_value"


SHAPE_CASES = {
    "shape_xox": "xox" + "b-" + "1234567890" + "ab",
    "shape_akia": "AK" + "IA" + "ABCDEFGH12345678",
    "shape_aiza": "AI" + "za" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5",
    "shape_jwt": "ey" + "J" + "hbGciOiJIUzI1.eyJzdWIiOiIx.abcDEF123456",
    "shape_bearer": "Bearer " + "abcdefgh" + "ijkl",       # letters only: keyword rule cannot catch it
}


@pytest.mark.parametrize("name", sorted(SHAPE_CASES))
def test_each_token_shape_is_rejected_on_its_own(name):
    text = "fyi " + SHAPE_CASES[name]                       # no keyword word in front
    assert secrets_guard.find_secret(text) == name, (name, text)
    r = _post("/memory/retain", text=text)
    assert r.status_code == 422 and name in r.json()["detail"], r.text


def test_update_with_a_secret_is_422_and_changes_nothing():
    old = "update probe: the dock number is forty two and the crane is red"
    doc = _post("/memory/retain", text=old).json()["_id"]
    secret = FAKE_GH
    ctl = _post("/memory/retain", text="update control: the lighthouse keeper left at noon").json()["_id"]
    ok = _post("/memory/update", doc_id=ctl, text="update control: the lighthouse keeper left at one")
    assert ok.status_code == 200, ok.text         # positive control: a clean update IS logged
    r = _post("/memory/update", doc_id=doc, text="new note with " + secret)
    assert r.status_code == 422 and "secret" in r.json()["detail"], r.text
    assert secret not in r.text
    got = es("GET", f"/{idx('semantic')}/_doc/{doc}")
    assert got["found"] and got["_source"]["text"] == old        # old doc survives
    es("POST", f"/{PREFIX}am_mutations/_refresh")
    muts = es("POST", f"/{PREFIX}am_mutations/_search",
              {"size": 100, "query": {"term": {"owner_id": OWNER}}})
    assert secret not in str(muts), "secret leaked into the mutation log"
    assert [h for h in muts["hits"]["hits"] if h["_source"]["doc_id"] == ctl], "control update not logged"
    assert not [h for h in muts["hits"]["hits"] if h["_source"]["doc_id"] == doc], \
        "a rejected update must not even log an intent"


def test_tags_are_guarded_on_retain_and_on_tags_only_update():
    secret = FAKE_GH
    r = _post("/memory/retain", text="tag guard probe one", tags=["note:" + secret])
    assert r.status_code == 422 and "tag" in r.json()["detail"], r.text
    assert secret not in r.text
    # underscore/hyphen shapes survive only on the raw string: norm_tags would turn them into spaces
    r2 = _post("/memory/retain", text="tag guard probe two", tags=["note:" + FAKE_GH_FINE])
    assert r2.status_code == 422, r2.text
    doc = _post("/memory/retain", text="tag guard probe three", tags=["area:harbour"]).json()["_id"]
    r2b = _post("/mcp/tools/update", doc_id=doc, tags=["note:" + FAKE_GH_FINE])
    assert r2b.status_code == 422, r2b.text
    r3 = _post("/memory/update", doc_id=doc, tags=["note:" + secret])
    assert r3.status_code == 422 and "tag" in r3.json()["detail"], r3.text
    got = es("GET", f"/{idx('semantic')}/_doc/{doc}")["_source"]
    assert "area:harbour" in got["tags"] and secret.lower() not in str(got)
    # a normal tag is still fine
    assert _post("/memory/update", doc_id=doc, tags=["area:crane"]).status_code == 200


def test_mcp_tools_maps_a_missing_doc_to_404_and_a_missing_field_to_422():
    r = _post("/mcp/tools/update", doc_id="does-not-exist-zzz", text="x")
    assert r.status_code == 404, r.text
    r2 = client.post("/mcp/tools/update", headers=H, json={"doc_id": "x"})   # no `kind`
    assert r2.status_code == 422, r2.text


def test_keyword_value_shorter_than_16_chars_is_accepted():
    for t in ("token: ab12cd34ef", "password: Passw0rd123", "api_key = abc123def4"):   # 10-12 chars
        assert secrets_guard.find_secret(t) is None, t
    assert secrets_guard.find_secret("token: " + "aB3" * 5 + "x") == "keyword_value"     # 16 chars
