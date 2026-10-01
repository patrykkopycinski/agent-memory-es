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


def _load() -> dict:
    path = _keys_file()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


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
