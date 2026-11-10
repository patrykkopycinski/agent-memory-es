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
import time
from typing import Optional

_lock = threading.Lock()


class KeyStoreUnavailable(RuntimeError):
    """The Elasticsearch key store cannot be read or written. Lookups fail closed on it."""


def es_mode() -> bool:
    """AMES_KEY_STORE=es keeps keys in Elasticsearch (stateless deploys: Cloud Run); the default
    `file` is the key volume. Read per call so tests can scope it."""
    mode = os.environ.get("AMES_KEY_STORE", "file").strip().lower() or "file"
    if mode not in ("file", "es"):
        raise ValueError("AMES_KEY_STORE must be 'file' or 'es'")
    return mode == "es"


def validate_config() -> None:
    """Startup check: key store mode and, for es mode, the key cache TTL."""
    if es_mode():
        from . import keystore_es
        keystore_es.validate_config()


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
    Also lowercases every stored `person` (one human, one person string, however the file was
    written). Returns whether anything changed. Pure: never touches the file."""
    changed = False
    for entry in keys.values():
        p = entry.get("person") if isinstance(entry, dict) else None
        if isinstance(p, str) and p != p.strip().lower():
            entry["person"] = p.strip().lower()
            changed = True
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
    if es_mode():
        from . import keystore_es
        return keystore_es.list_all()
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
    only: readers must not take it (see owner_of).

    makedirs lives here, and in _save, on purpose: create_key is what creates the key store
    directory on first use. Never move it onto a read path (owner_of/_read/load_readonly):
    lookups on a missing or read-only store must neither create nor fail."""
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


@contextlib.contextmanager
def _try_guard():
    """Non-blocking _guard for best-effort writers on a hot path. Yields True when both the
    thread lock and the file lock were taken without waiting; yields False (and does
    nothing) when either is busy or the lock file cannot be opened. No makedirs: the
    caller has just read the key file, so the directory exists."""
    if not _lock.acquire(blocking=False):
        yield False
        return
    try:
        try:
            lf = open(_keys_file() + ".lock", "a")
        except OSError:
            lf = None
        locked = False
        if lf is not None:
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                pass
        try:
            yield locked
        finally:
            if lf is not None:
                if locked:
                    fcntl.flock(lf, fcntl.LOCK_UN)
                lf.close()
    finally:
        _lock.release()


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


ROLES = ("member", "owner", "approver")
DEFAULT_ROLE = "member"    # create_key without a role; file entries that carry none behave the same


def clean_person(person: Optional[str]) -> str:
    """Trimmed and lowercased: one human, one person string (`Alice` and `alice` are one person)."""
    if not isinstance(person, str) or not person.strip() or len(person) > 200:
        raise ValueError("person is required: the name of the human behind the key (1-200 chars)")
    return person.strip().lower()


def _mint() -> tuple:
    token = "ame_" + secrets.token_hex(32)
    return token, hashlib.sha256(token.encode()).hexdigest()


def create_key(owner_id: str, role: str = DEFAULT_ROLE, person: Optional[str] = None) -> str:
    """Mint an API key for owner_id. Returns the plaintext key once. `person` is REQUIRED: it names
    the human behind the key, and governance counts people, not keys (a key with no person would
    store the owner id as the author of its proposals, which an approver key could not tell from
    its own person). Keys that predate this carry no person and keep working as agent keys; give
    them one with set_person."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    person = clean_person(person)
    entry = {"owner_id": owner_id, "role": role, "person": person}
    if es_mode():
        from . import keystore_es
        return keystore_es.create(entry, _mint)
    token, digest = _mint()
    with _guard():
        keys = _load()
        keys[digest] = entry
        _save(keys)
    return token


def set_person(owner_id: str, person: str) -> int:
    """Admin: name the person behind every key of `owner_id` that has none (legacy keys). Keys that
    already name a person are left alone. Returns how many keys were updated."""
    person = clean_person(person)
    if es_mode():
        from . import keystore_es
        return keystore_es.set_person(owner_id, person)
    with _guard():
        keys = _load()
        n = 0
        for entry in keys.values():
            if isinstance(entry, dict) and entry.get("owner_id") == owner_id and not entry.get("person"):
                entry["person"] = person
                n += 1
        if n:
            _save(keys)
    return n


def persons_of(owner_id: Optional[str]) -> set:
    """Every person named on a key of `owner_id` (read-only, like the worker's key view)."""
    if es_mode():
        from . import keystore_es
        return keystore_es.persons_of(owner_id)
    return {e["person"].lower() for e in load_readonly().values()
            if isinstance(e, dict) and e.get("owner_id") == owner_id
            and isinstance(e.get("person"), str) and e["person"]}


# Best-effort legacy migration is attempted at most once per (file version, backoff window)
# per path, so a read-only or contended store costs one failed attempt, not one per request.
_MIGRATE_BACKOFF_S = 60.0
_migrate_attempts: dict = {}  # path -> ((st_mtime_ns, st_size), monotonic time of attempt)


def _migration_due(path: str) -> bool:
    try:
        st = os.stat(path)
    except OSError:
        return False
    sig = (st.st_mtime_ns, st.st_size)
    now = time.monotonic()
    prev = _migrate_attempts.get(path)
    if prev and prev[0] == sig and now - prev[1] < _MIGRATE_BACKOFF_S:
        return False
    _migrate_attempts[path] = (sig, now)
    return True


def owner_of(token: str) -> Optional[dict]:
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    if es_mode():
        from . import keystore_es
        return keystore_es.get(digest)
    # lock-free: _save is an atomic os.replace, so a reader sees a whole old or new file
    keys = _read()
    if _normalize_legacy(keys) and _migration_due(_keys_file()):
        # persist the migration best-effort and never wait for it: the in-memory result
        # already answers this lookup. _load re-reads inside the guard, so a key deleted
        # since our read stays deleted.
        try:
            with _try_guard() as got:
                if got:
                    _load()
        except OSError:
            pass
    return keys.get(digest)


def delete_key(token: str) -> bool:
    digest = hashlib.sha256(token.encode()).hexdigest()
    if es_mode():
        from . import keystore_es
        return keystore_es.delete(digest)
    with _guard():
        keys = _load()
        if digest in keys:
            del keys[digest]
            _save(keys)
            return True
    return False
