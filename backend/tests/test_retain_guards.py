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
    "model: qwen3.8-35b-a3b-instruct",
    "version=9.6.0-SNAPSHOT",
    "commit: 1470bfd8b2fc1",
    "token path: owner/repo/token-file",
    "token: owner/repo/token-file2",
    "secret: /Users/someone/ames-data/ROLLBACK-b400b54-20261105T091828.txt",
    "auth: commit 1470bfd8b2fc1a2b3c4d5e6f7a8b9c0d1e2f3a4b",
    "credentials: elastic/kibana-evals/weekly2",
    "SERVICE_KEY=deepseek-v3.2-flash",
    "api key: gpt-4o-2024-08-06",
    "model: meta-llama/Llama-3.3-70B-Instruct",
    "token: Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "token file: src/plugins/auth/README.md",
    "token: src/plugins2/auth/README.md",             # same path where the keyword rule really fires
    "secret: projects/123456789/secrets/my-api/versions/3",   # the path is the secret's NAME, not its value
    "token: ~/src/plugins2/auth/README.md",
    "token: ./src/plugins2/auth/README.md",
    "token: ../src/plugins2/auth/README.md",
    "secret: /srv/app2/config/keys.json",
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


REAL_LOOKING = [
    "token: 1470bfd8b2fc1a2b3c4d5e6f7a8b9c0d1e2f3a4b",    # 40 hex with no commit context
    "api_key = " + "Xk9mQ2" * 4,
    "secret=" + "a1b2c3d4" * 3,
    "credentials: " + "Ab3" * 7,
    "password: hunter2" + "Zq8" * 3,
    "secret: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",   # AWS-style: '/' inside, mixed-case segments
    "token: Qw8rT2yU5i/O9pA3sD6fG7hJ",
    "auth: 3f9a8c7b6d5e4f3a2b1c0d9e8f7a6b5c",
    "token: ABCD1234-EFGH5678-IJKL9012",
    "OPENAI_" + "API_KEY=Qw8rT2yU5iO9pA3sD6fG",
    "SERVICE" + "_KEY=Qw8rT2yU5iO9pA3sD6fG",
    "my-key: Qw8rT2yU5iO9pA3sD6fG",
    "api key: Qw8rT2yU5iO9pA3sD6fG",
    "API KEY = Qw8rT2yU5iO9pA3sD6fG",
]


@pytest.mark.parametrize("text", REAL_LOOKING)
def test_real_looking_keyword_secrets_are_rejected_incl_suffix_and_spaced_forms(text):
    assert secrets_guard.find_secret(text) == "keyword_value", text


def test_key_words_in_prose_are_not_a_keyword_match():
    for t in ("the primary key and the foreign-key index", "an api key rotation policy is due",
              "monkey: Qw8rT2yU5iO9pA3sD6fG is not a thing"):   # `monkey:` ends in 'key' but not '_key'
        assert secrets_guard.find_secret(t) is None, t


def test_keyword_regex_is_linear_on_adversarial_input():
    import time
    t0 = time.time()
    for t in ("a_key " * 20000, "x" + "_key" * 20000, "api" + " " * 50000 + "key",
              "token:" + "-" * 50000, "token: " + "a/" * 30000, "SERVICE_KEY=" + "Ab1" * 30000):
        secrets_guard.find_secret(t)
    assert time.time() - t0 < 2.0


# --- slug exemption (kills "slug removed" and "slug widened") ---
def test_slug_exemption_exists_lowercase_multi_part_ids_pass():
    for v in ("gpt-4o-2024-08-06", "deepseek-v3.2-flash", "claude-opus-4-5-20251101", "qwen3.8-35b-a3b-instruct"):
        assert secrets_guard.find_secret("model: " + v) is None, v


def test_slug_exemption_is_narrow_secret_like_values_are_not_slugs():
    # each is 16+ chars, letters+digits, joined by separators -- and must NOT be treated as a slug
    for v in ("Zx81-QwQw-Ab12-Cd34-Ef56",          # digits inside UPPER/mixed-case parts
              "qwertyuiopasdfgh1234zz",            # single part: a slug needs 2+ parts
              "abc12def-ghi34jkl-mno56pqr",        # digit runs in the middle of parts
              "k9k9k9k9-x1x1x1x1-z7z7z7z7",        # repeating letter+digit blocks
              "AbCdEf12-GhIjKl34"):
        assert secrets_guard.find_secret("token: " + v) == "keyword_value", v


def test_uppercase_secret_shaped_tags_are_rejected_on_retain_and_update():
    doc = _post("/memory/retain", text="upper tag probe: the quay is closed on sundays").json()["_id"]
    for tag in ("NOTE:" + FAKE_GH.upper(), "NOTE:" + FAKE_SK.upper(), "API_KEY:" + ("XK9MQ2" * 4),
                "NOTE:AK" + "IAABCDEFGH12345678"):
        r = _post("/memory/retain", text="upper tag retain " + tag[:8], tags=[tag])
        assert r.status_code == 422 and "tag" in r.json()["detail"], (tag, r.text)
        u = _post("/memory/update", doc_id=doc, tags=[tag])
        assert u.status_code == 422 and "tag" in u.json()["detail"], (tag, u.text)
    got = es("GET", f"/{idx('semantic')}/_doc/{doc}")["_source"]
    assert "XK9MQ2" not in str(got).upper()


def test_promote_of_a_missing_doc_is_404_not_500_on_both_surfaces():
    r = _post("/memory/promote", doc_id="no-such-doc-zzz", to_visibility="team")
    assert r.status_code == 404, r.text
    r2 = _post("/mcp/tools/promote", doc_id="no-such-doc-zzz", to_visibility="team")
    assert r2.status_code == 404, r2.text
    mine = _post("/memory/retain", text="promote control: a real private fact").json()["_id"]
    assert _post("/memory/promote", doc_id=mine, to_visibility="team").status_code == 200


# --- review round 2 ---
HEX32 = "3f9a8c7b6d5e4f3a2b1c0d9e8f7a6b5c"                       # 32 hex: Datadog / Sentry / npm key shape
HEX40 = "1470bfd8b2fc1a2b3c4d5e6f7a8b9c0d1e2f3a4b"               # 40 hex: also a git SHA-1
HEX64 = HEX32 + HEX32[::-1]

MUST_REJECT = [
    # rooted values get the same per-segment check as relative ones
    "aws_secret: /K7MDENG/bPxRfiCY9kLmQ2zX8vB7nR4tY0wJalrXUtn",
    "secret: ./Xk9mQ2pL7vR4tN8wB3zH/a1",
    "token: ~/Xk9mQ2pL7vR4tN8w/B3zH",
    # prose around a hex value is not commit context
    "pushed the merge; deploy token: " + HEX32,
    "After the merge, set the datadog api key: " + HEX32,
    "Since commit 1a2b3c, DD_" + "API_KEY=" + HEX32,
    "pinned at HEAD; token: " + HEX40,
    "The commit hook uses token: " + HEX40,
    "api_key: at " + HEX32,
    # scheme words that were removed from the allow-list
    "token: ref " + HEX40, "token: head " + HEX40, "token: pr " + HEX40, "token: at " + HEX40,
    # hash exemption is length- and case-bound
    "auth: commit " + HEX32,
    "auth: sha " + HEX64,
    "auth: commit " + HEX40.upper(),
    "auth: commit " + HEX40[:20],
    "auth: commit " + HEX40 + "ab",
    # a path needs EVERY segment and every piece to be an ordinary word or id (all, not any)
    "token: owner/repo/k9x2m7q1p4z8",
    "token: owner/repo/k9x2m7q1-p4z8w3n6-owner",
    "secret: ./k9x2m7q1/p4z8w3n6/r5t1y8u2",
    # a Capitalized word is a path piece, but a 12+ char mixed-case-and-digit run never is
    "token: owner/Abcdefghijkl9/repo",
    # hex chunks joined by '/' are not a path
    "secret: 3f9a8c7b/6d5e4f3a/2b1c0d9e/8f7a6b5c",
    "token: 3f9a8c7b6d5e/4f3a2b1c0d9e",
]


@pytest.mark.parametrize("text", MUST_REJECT)
def test_review_round_2_bypasses_stay_rejected(text):
    assert secrets_guard.find_secret(text) == "keyword_value", text


@pytest.mark.parametrize("text", [
    "commit: 1470bfd8b2fc1", "auth: commit " + HEX40, "token: sha " + HEX40[:12], "auth: rev 1470bfd8b2fc1",
    "auth: hash " + HEX40,
])
def test_a_commit_hash_with_a_hash_scheme_word_is_still_accepted(text):
    assert secrets_guard.find_secret(text) is None, text


def test_hash_exemption_has_no_before_context_window():                                  # N5c is moot
    assert not hasattr(secrets_guard, "_COMMIT_CONTEXT")
    assert secrets_guard.find_secret("commit " + HEX40[:7] + " then token: " + HEX40) == "keyword_value"


def test_hash_exemption_is_bounded_hex_only():                                             # N5e
    for n in (16, 20, 32, 39, 41, 64):
        assert secrets_guard.find_secret("auth: commit " + (HEX64 * 2)[:n]) == "keyword_value", n
    assert secrets_guard.find_secret("auth: commit " + HEX40[:7] + "0000") is None          # 11 hex ok
    assert secrets_guard.find_secret("auth: commit " + "AB" + HEX40[2:]) == "keyword_value"  # A-F not hex here


def test_mixed_case_alpha_pieces_are_not_slug_pieces():                                    # N6
    for v in ("aBcDeFgHiJkLmNoP-v3", "token-xK" + "pQrStUvWxYzAbC-v3", "camelCaseTokenValueHere-2"):
        assert secrets_guard.find_secret("token: " + v) == "keyword_value", v


def test_digit_bearing_slug_pieces_must_be_lowercase():                                    # N6b
    for v in ("Abc12-def34-ghi56-jkl78", "ABC12-DEF34-GHI56-JKL78", "abc12-Def34-ghi56-jkl78"):
        assert secrets_guard.find_secret("token: " + v) == "keyword_value", v


def test_every_slug_part_and_dot_piece_must_pass_N10():
    for v in ("deepseek-Xk9mQ2pL7vR4tN8w", "gpt-4o.Xk9mQ2pL7vR4tN8w-flash", "Xk9mQ2pL7vR4tN8w-flash-v3"):
        assert secrets_guard.find_secret("token: " + v) == "keyword_value", v
    assert secrets_guard.find_secret("token: owner/Xk9mQ2pL7vR4tN8w3/repo") == "keyword_value"


def test_slash_is_not_a_slug_separator_and_a_lone_segment_is_not_a_path_N11():
    assert not secrets_guard._is_slug("abcdefghijk1/m")
    assert not secrets_guard._is_path("abcdefghijklmnop1/")                                   # one segment only
    assert not secrets_guard._is_path("Xk9mQ2pL7vR4tN8w3zH")                                  # no slash at all


def test_scheme_word_group_is_linear_time_N12b():
    import subprocess
    code = ("import sys,time; sys.path.insert(0,'.'); from app import secrets_guard as g\n"
            "t=time.time()\n"
            "for k in (3, 5, 8, 40):\n"
            "    g.find_secret('token: ' + 'abcdefgh'*k + '!')\n"
            "    g.find_secret('token: ' + 'abcdefg '*k + '!')\n"
            "print(time.time()-t)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20,
                       cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert r.returncode == 0, r.stderr
    assert float(r.stdout.strip()) < 1.0


def test_promote_checks_the_kind_before_touching_the_store_N15():
    with pytest.raises(ValueError, match="kind must be one of"):
        memory.promote(OWNER, "bogus", "any-doc", "team")
    r = _post("/memory/promote", kind="bogus", doc_id="any-doc", to_visibility="team")
    assert r.status_code == 422 and "kind must be one of" in r.text, r.text
