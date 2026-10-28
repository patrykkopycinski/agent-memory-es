"""Per-owner API keys and tenancy. Keys are random 32-byte tokens; owner identity
is derived server-side from the key, never trusted from the caller."""
import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import tempfile
import threading
from typing import Optional

_lock = threading.Lock()


def _keys_file() -> str:
    # resolved per call so tests (and multi-tenant deploys) can scope key stores
    return os.environ.get("AMES_API_KEYS_FILE",
                          os.path.join(os.path.dirname(__file__), "..", "data", "api_keys.json"))


def _normalize_legacy(keys: dict) -> bool:
    """Legacy/hand-written key files stored plaintext tokens under owner-name keys
    ({"hermes-default": "ame_..."}). owner_of() looks up sha256 digests, so those
    entries 401 for every token. Normalize in place: add a digest entry for each
    plaintext token (the plaintext token is never persisted by create_key; the
    name-keyed copy stays so hand-managed file expectations don't break).
    Returns whether anything was added. Pure: never touches the file."""
    changed = False
    for name, val in list(keys.items()):
        if isinstance(val, str) and val.startswith("ame_"):
            digest = hashlib.sha256(val.encode()).hexdigest()
            if digest not in keys:
                keys[digest] = {"owner_id": name, "role": "owner"}
                changed = True
    return changed


def _read() -> dict:
    path = _keys_file()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_readonly() -> dict:
    """Keys with legacy entries normalized in memory only. For processes that share the
    key volume read-only (the consolidation worker): never writes, never takes the lock."""
    keys = _read()
    _normalize_legacy(keys)
    return keys


def _load() -> dict:
    """Read + persist legacy normalization. Callers that mutate or may migrate must hold
    _guard() so the load-modify-save cycle is atomic across threads AND processes."""
    keys = _read()
    if _normalize_legacy(keys):
        _save(keys)
    return keys


class KeyStoreNotWritable(OSError):
    """The key store directory or its lock file cannot be created/opened for writing."""


@contextlib.contextmanager
def _guard():
    """Serialize load-modify-save: threading.Lock within a process, flock on a sidecar file
    across processes (backend and any other writer sharing the key volume). Writer paths
    only: readers must not take it (see owner_of)."""
    with _lock:
        path = _keys_file()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            lf = open(path + ".lock", "a")
        except OSError as e:
            raise KeyStoreNotWritable(f"key store not writable: {path}: {e}") from e
        with lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


def _save(keys: dict) -> None:
    path = _keys_file()
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    # unique temp in the same dir: a fixed ".tmp" name lets two writers clobber each other's
    # half-written file before os.replace
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ames_keys.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(keys, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # data on disk before the rename can make it visible
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o777)
        except FileNotFoundError:
            pass  # new file keeps mkstemp's 0600
        os.replace(tmp, path)
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)  # persist the rename itself
        finally:
            os.close(dfd)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def create_key(owner_id: str, role: str = "member") -> str:
    """Mint an API key for owner_id. Returns the plaintext key once."""
    token = "ame_" + secrets.token_hex(32)
    digest = hashlib.sha256(token.encode()).hexdigest()
    with _guard():
        keys = _load()
        keys[digest] = {"owner_id": owner_id, "role": role}
        _save(keys)
    return token


def owner_of(token: str) -> Optional[dict]:
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    # lock-free: _save is an atomic os.replace, so a reader sees a whole old or new file
    keys = _read()
    if _normalize_legacy(keys):
        try:  # persist the migration best-effort; the in-memory result already answers this lookup
            with _guard():
                _load()
        except OSError:
            pass
    return keys.get(digest)


def delete_key(token: str) -> bool:
    digest = hashlib.sha256(token.encode()).hexdigest()
    with _guard():
        keys = _load()
        if digest in keys:
            del keys[digest]
            _save(keys)
            return True
    return False
