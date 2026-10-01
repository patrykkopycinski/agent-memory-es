"""Per-owner API keys and tenancy. Keys are random 32-byte tokens; owner identity
is derived server-side from the key, never trusted from the caller."""
import hashlib
import json
import os
import secrets
import threading
from typing import Optional

from .store import API_KEYS_FILE

_lock = threading.Lock()


def _load() -> dict:
    if not os.path.exists(API_KEYS_FILE):
        return {}
    with open(API_KEYS_FILE) as f:
        return json.load(f)


def _save(keys: dict) -> None:
    os.makedirs(os.path.dirname(API_KEYS_FILE), exist_ok=True)
    tmp = API_KEYS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(keys, f, indent=2)
    os.replace(tmp, API_KEYS_FILE)


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
