"""Elasticsearch key store (AMES_KEY_STORE=es): minting, lookup, cache, migration, worker view."""
import hashlib
import json
import logging
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import auth, keystore_es, keystore_migrate, worker  # noqa: E402
from app.store import es  # noqa: E402


@pytest.fixture
def es_keys(monkeypatch):
    monkeypatch.setenv("AMES_KEY_STORE", "es")
    keystore_es._cache.clear()
    yield
    keystore_es._cache.clear()


def _digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("AMES_KEY_STORE", "redis")
    with pytest.raises(ValueError):
        auth.es_mode()


def test_create_and_lookup(es_keys):
    token = auth.create_key("ks-alice", "approver", person="Alice")
    assert auth.owner_of(token) == {"owner_id": "ks-alice", "role": "approver", "person": "alice"}
    doc = es("GET", f"/{keystore_es.INDEX}/_doc/{_digest(token)}")["_source"]
    assert set(doc) == {"owner_id", "role", "person", "created_at"}
    assert token not in json.dumps(doc)


def test_plaintext_token_is_never_stored(es_keys):
    token = auth.create_key("ks-plain", person="plain")
    hits = es("POST", f"/{keystore_es.INDEX}/_search",
              {"query": {"term": {"owner_id": "ks-plain"}}})["hits"]["hits"]
    assert hits and all(token not in json.dumps(h) for h in hits)
    assert hits[0]["_id"] == _digest(token)


def test_unknown_key_is_none_and_never_cached(es_keys):
    bogus = "ame_" + "0" * 64
    assert auth.owner_of(bogus) is None
    assert _digest(bogus) not in keystore_es._cache
    assert auth.owner_of("") is None


def test_negative_lookup_does_not_lock_out_a_later_mint(es_keys):
    token, digest = auth._mint()
    assert auth.owner_of(token) is None
    keystore_es.put_digest(digest, {"owner_id": "ks-late", "role": "member"})
    assert auth.owner_of(token)["owner_id"] == "ks-late"


def test_positive_cache_serves_within_ttl_and_expires(es_keys, monkeypatch):
    monkeypatch.setenv("AMES_KEY_CACHE_TTL_S", "0.2")
    token = auth.create_key("ks-cache", person="cache")
    assert auth.owner_of(token)["owner_id"] == "ks-cache"
    es("DELETE", f"/{keystore_es.INDEX}/_doc/{_digest(token)}?refresh=true")
    assert auth.owner_of(token) is not None      # cached
    time.sleep(0.3)
    assert auth.owner_of(token) is None          # expired, re-read from ES


def test_delete_key_drops_cache(es_keys):
    token = auth.create_key("ks-del", person="del")
    assert auth.owner_of(token)
    assert auth.delete_key(token) is True
    assert auth.owner_of(token) is None
    assert auth.delete_key(token) is False


def test_roles_and_validation(es_keys):
    for role in auth.ROLES:
        assert auth.owner_of(auth.create_key(f"ks-{role}", role, person="p"))["role"] == role
    with pytest.raises(ValueError):
        auth.create_key("ks-x", "root", person="p")
    with pytest.raises(ValueError):
        auth.create_key("ks-x", "member")


def test_mint_collision_is_retried_never_overwritten(es_keys):
    first, first_digest = auth._mint()
    keystore_es.put_digest(first_digest, {"owner_id": "ks-victim", "role": "owner"})
    second = auth._mint()
    tokens = iter([(first, first_digest), second])
    minted = keystore_es.create({"owner_id": "ks-new", "role": "member", "person": "n"}, lambda: next(tokens))
    assert minted == second[0]
    assert auth.owner_of(first)["owner_id"] == "ks-victim"


def test_set_person_only_fills_missing(es_keys):
    _t, d = auth._mint()
    keystore_es.put_digest(d, {"owner_id": "ks-legacy", "role": "owner"})
    named = auth.create_key("ks-legacy", person="Already")
    assert auth.set_person("ks-legacy", "Bob") == 1
    assert auth.persons_of("ks-legacy") == {"already", "bob"}
    assert auth.owner_of(named)["person"] == "already"


def test_known_owners_via_es(es_keys):
    auth.create_key("ks-owner-a", person="a")
    auth.create_key("ks-owner-b", person="b")
    assert {"ks-owner-a", "ks-owner-b"} <= set(worker.known_owners())


def test_migration_is_idempotent_and_prints_counts_only(es_keys, tmp_path, capsys):
    plain = "ame_" + "c" * 64
    kept, kept_digest = auth._mint()
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({
        "ks-mig-legacy": plain,
        kept_digest: {"owner_id": "ks-mig-new", "role": "approver", "person": "Mig"}}))
    before = path.read_text()
    assert keystore_migrate.main(["--from-file", str(path)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first == {"created": 2, "existing": 0, "skipped": 1}
    assert keystore_migrate.main(["--from-file", str(path)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == {"created": 0, "existing": 2, "skipped": 1}
    assert path.read_text() == before            # source file untouched
    assert auth.owner_of(plain) == {"owner_id": "ks-mig-legacy", "role": "owner"}
    assert auth.owner_of(kept)["owner_id"] == "ks-mig-new"
    stored = json.dumps(es("POST", f"/{keystore_es.INDEX}/_search",
                           {"query": {"term": {"owner_id": "ks-mig-legacy"}}}))
    assert plain not in stored


def test_migration_missing_file_reports_failure(es_keys, tmp_path, capsys):
    assert keystore_migrate.main(["--from-file", str(tmp_path / "nope.json")]) == 1
    assert "ame_" not in capsys.readouterr().err


def test_unreachable_store_fails_closed(es_keys, monkeypatch):
    def down(*_a, **_k):
        raise ConnectionRefusedError("boom")
    monkeypatch.setattr(keystore_es, "es", down)
    with pytest.raises(auth.KeyStoreUnavailable):
        auth.owner_of("ame_" + "d" * 64)
    assert keystore_es.health() is False
    assert worker.known_owners() == []


def test_zero_owner_guard_logs_error_when_docs_exist(monkeypatch, caplog):
    monkeypatch.setattr(worker, "known_owners", lambda: [])
    monkeypatch.setattr(worker, "es", lambda *a, **k: {"count": 3})
    with caplog.at_level(logging.ERROR, logger=worker.log.name):
        assert worker.run_once() == {}
    assert "zero owners: key store unreachable or empty" in caplog.text


def test_zero_owner_guard_silent_on_fresh_install(monkeypatch, caplog):
    monkeypatch.setattr(worker, "known_owners", lambda: [])
    monkeypatch.setattr(worker, "es", lambda *a, **k: {"count": 0})
    with caplog.at_level(logging.ERROR, logger=worker.log.name):
        assert worker.run_once() == {}
    assert "zero owners" not in caplog.text


def test_zero_owner_guard_never_raises(monkeypatch):
    monkeypatch.setattr(worker, "known_owners", lambda: [])
    def broken(*_a, **_k):
        raise RuntimeError("es down")
    monkeypatch.setattr(worker, "es", broken)
    assert worker.run_once() == {}


def test_health_reports_key_store_without_key_material(es_keys, monkeypatch):
    from fastapi.testclient import TestClient
    from app import embeddings, main
    monkeypatch.setattr(embeddings, "probe", lambda: None)
    client = TestClient(main.app)
    token = auth.create_key("ks-health", person="h")
    main._health_reset()
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["key_store"] == "es" and body["key_store_ok"] is True
    assert token not in r.text and _digest(token) not in r.text
    main._health_reset()
    monkeypatch.setattr(keystore_es, "es", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    r = client.get("/health/ready")
    assert r.status_code == 503 and r.json()["key_store_ok"] is False
    assert client.get("/memory/facts/drain", headers={"X-API-Key": token}).status_code == 503
    main._health_reset()


def test_file_mode_health_has_no_key_store_field(monkeypatch):
    from fastapi.testclient import TestClient
    from app import embeddings, main
    monkeypatch.delenv("AMES_KEY_STORE", raising=False)
    monkeypatch.setattr(embeddings, "probe", lambda: None)
    main._health_reset()
    assert "key_store" not in TestClient(main.app).get("/health").json()
    main._health_reset()
