"""Legacy/hand-written key files must self-heal: plaintext token under an
owner-name key becomes a sha256-digest entry on first _load, so owner_of()
resolves instead of 401ing every token (self-hosted deploy incident 2026-10-01)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import auth  # noqa: E402


def test_legacy_plaintext_entry_migrates(tmp_path, monkeypatch):
    keys_file = tmp_path / "keys.json"
    token = "ame_" + "a" * 64
    keys_file.write_text(json.dumps({"hermes-default": token}))
    monkeypatch.setattr(auth, "_keys_file", lambda: str(keys_file))

    who = auth.owner_of(token)
    assert who == {"owner_id": "hermes-default", "role": "owner"}

    # persisted: digest entry written back alongside the name-keyed plaintext
    on_disk = json.loads(keys_file.read_text())
    import hashlib
    digest = hashlib.sha256(token.encode()).hexdigest()
    assert on_disk[digest] == {"owner_id": "hermes-default", "role": "owner"}
    assert on_disk["hermes-default"] == token  # original entry untouched


def test_digest_entries_untouched(tmp_path, monkeypatch):
    keys_file = tmp_path / "keys.json"
    monkeypatch.setattr(auth, "_keys_file", lambda: str(keys_file))
    token = auth.create_key("alice")
    before = keys_file.read_text()
    auth.owner_of(token)  # triggers _load -> migrate (no-op)
    assert keys_file.read_text() == before
    assert auth.owner_of(token)["owner_id"] == "alice"  # type: ignore[index]


def test_non_ame_strings_ignored(tmp_path, monkeypatch):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"note": "not-a-token"}))
    monkeypatch.setattr(auth, "_keys_file", lambda: str(keys_file))
    assert json.loads(keys_file.read_text()) == {"note": "not-a-token"}
    assert auth.owner_of("ame_" + "b" * 64) is None


def test_worker_known_owners_never_writes_legacy_key_file(tmp_path, monkeypatch):
    """The worker mounts the key volume read-only: loading a legacy-plaintext file must
    normalize in memory and leave the file byte-identical."""
    from app import worker
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"hermes-default": "ame_" + "c" * 64}))
    monkeypatch.setattr(auth, "_keys_file", lambda: str(keys_file))
    before = keys_file.read_bytes()
    assert worker.known_owners() == ["hermes-default"]
    assert keys_file.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keys.json"]  # no .tmp/.lock side files


def test_worker_known_owners_works_on_readonly_directory(tmp_path, monkeypatch):
    from app import worker
    d = tmp_path / "ro"
    d.mkdir()
    (d / "keys.json").write_text(json.dumps({"hermes-default": "ame_" + "d" * 64}))
    monkeypatch.setattr(auth, "_keys_file", lambda: str(d / "keys.json"))
    d.chmod(0o500)
    try:
        assert worker.known_owners() == ["hermes-default"]
    finally:
        d.chmod(0o700)


def _mint(path, owner, n, flock):
    os.environ["AMES_API_KEYS_FILE"] = path
    if not flock:
        import contextlib
        auth._guard = contextlib.nullcontext  # emulate the old per-process threading.Lock only
    return [auth.create_key(f"{owner}-{i}") for i in range(n)]


def _mint_to_queue(path, owner, n, flock, q):
    try:
        q.put(_mint(path, owner, n, flock))
    except Exception as e:  # a crashed writer must fail the test fast, not hang the parent
        q.put(f"{type(e).__name__}: {e}")


def _concurrent_mint(path, flock, procs=4, n=25):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    ps = [ctx.Process(target=_mint_to_queue, args=(path, f"p{i}", n, flock, q)) for i in range(procs)]
    for p in ps:
        p.start()
    results = [q.get(timeout=60) for _ in ps]
    for p in ps:
        p.join(timeout=30)
    errors = [r for r in results if isinstance(r, str)]
    assert not errors, f"writer crashed: {errors}"
    return [t for r in results for t in r]


def test_concurrent_create_key_across_processes_loses_nothing(tmp_path):
    path = str(tmp_path / "keys.json")
    tokens = _concurrent_mint(path, flock=True)
    on_disk = json.loads(open(path).read())
    import hashlib
    missing = [t for t in tokens if hashlib.sha256(t.encode()).hexdigest() not in on_disk]
    assert len(tokens) == 100 and not missing, f"{len(missing)} of {len(tokens)} keys lost"
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
