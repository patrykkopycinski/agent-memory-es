"""Legacy/hand-written key files must self-heal: plaintext token under an
owner-name key becomes a sha256-digest entry on first _load, so owner_of()
resolves instead of 401ing every token (m1max deploy incident 2026-10-01)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import auth  # noqa: E402


def test_legacy_plaintext_entry_migrates(tmp_path):
    keys_file = tmp_path / "keys.json"
    token = "ame_" + "a" * 64
    keys_file.write_text(json.dumps({"hermes-default": token}))
    auth._keys_file = lambda: str(keys_file)  # type: ignore[assignment]

    who = auth.owner_of(token)
    assert who == {"owner_id": "hermes-default", "role": "owner"}

    # persisted: digest entry written back alongside the name-keyed plaintext
    on_disk = json.loads(keys_file.read_text())
    import hashlib
    digest = hashlib.sha256(token.encode()).hexdigest()
    assert on_disk[digest] == {"owner_id": "hermes-default", "role": "owner"}
    assert on_disk["hermes-default"] == token  # original entry untouched


def test_digest_entries_untouched(tmp_path):
    keys_file = tmp_path / "keys.json"
    auth._keys_file = lambda: str(keys_file)  # type: ignore[assignment]
    token = auth.create_key("alice")
    before = keys_file.read_text()
    auth.owner_of(token)  # triggers _load -> migrate (no-op)
    assert keys_file.read_text() == before
    assert auth.owner_of(token)["owner_id"] == "alice"  # type: ignore[index]


def test_non_ame_strings_ignored(tmp_path):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"note": "not-a-token"}))
    auth._keys_file = lambda: str(keys_file)  # type: ignore[assignment]
    assert json.loads(keys_file.read_text()) == {"note": "not-a-token"}
    assert auth.owner_of("ame_" + "b" * 64) is None
