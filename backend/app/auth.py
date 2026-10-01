"""Per-owner API keys and tenancy. Keys are random 32-byte tokens; owner identity
is derived server-side from the key, never trusted from the caller."""
import hashlib
import json
import os
import secrets
import threading
from typing import Optional

_lock = threading.Lock()


def _keys_file() -> str:
    # resolved per call so tests (and multi-tenant deploys) can scope key stores
    return os.environ.get("AMES_API_KEYS_FILE",
                          os.path.join(os.path.dirname(__file__), "..", "data", "api_keys.json"))


def _migrate_legacy(keys: dict) -> dict:
    """Legacy/hand-written key files stored plaintext tokens under owner-name keys
    ({"hermes-default": "ame_..."}). owner_of() looks up sha256 digests, so those
    entries 401 for every token. Normalize: rewrite each plaintext entry as a
    digest entry (plaintext token is never persisted by create_key; keep the
    name-keyed copy too so hand-managed file expectations don't break)."""
    changed = False
    for name, val in list(keys.items()):
        if isinstance(val, str) and val.startswith("ame_"):
            digest = hashlib.sha256(val.encode()).hexdigest()
            role = "owner"
            if digest not in keys:
                keys[digest] = {"owner_id": name, "role": role}
                changed = True
    if changed:
        _save(keys)
    return keys


def _load() -> dict:
    path = _keys_file()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        keys = json.load(f)
    return _migrate_legacy(keys)


def _save(keys: dict) -> None:
    path = _keys_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(keys, f, indent=2)
    os.replace(tmp, path)


def create_key(owner_id: str, role: str = "member") -> str:
    """Mint an API key for owner_id. Returns the plaintext key once."""
    token = "ame_" + secrets.token_hex(32)
    digest = hashlib.sha256(token.encode()).hexdigest()
    with _lock:
        keys = _load()
        keys[digest] = {"owner_id": owner_id, "role": role}
        _save(keys)
    return token


def owner_of(token: str) -> Optional[dict]:
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    with _lock:
        return _load().get(digest)


def delete_key(token: str) -> bool:
    digest = hashlib.sha256(token.encode()).hexdigest()
    with _lock:
        keys = _load()
        if digest in keys:
            del keys[digest]
            _save(keys)
            return True
    return False
