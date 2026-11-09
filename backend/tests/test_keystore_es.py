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
    keystore_es.ensure_index()
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
    _clear_sentinel()
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
    assert keystore_migrate.main(["--from-file", str(path), "--force"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == {"created": 0, "existing": 2, "skipped": 1}
    assert path.read_text() == before            # source file untouched
    assert auth.owner_of(plain) == {"owner_id": "ks-mig-legacy", "role": "owner"}
    assert auth.owner_of(kept)["owner_id"] == "ks-mig-new"
    stored = json.dumps(es("POST", f"/{keystore_es.INDEX}/_search",
                           {"query": {"term": {"owner_id": "ks-mig-legacy"}}}))
    assert plain not in stored


def _clear_sentinel():
    try:
        es("DELETE", f"/{keystore_es.INDEX}/_doc/{keystore_es.SENTINEL_ID}?refresh=true")
    except RuntimeError:
        pass


def test_migration_second_run_is_refused_without_force(es_keys, tmp_path, capsys):
    _clear_sentinel()
    _t, digest = auth._mint()
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({digest: {"owner_id": "ks-rerun", "role": "member"}}))
    assert keystore_migrate.main(["--from-file", str(path)]) == 0
    capsys.readouterr()
    es("DELETE", f"/{keystore_es.INDEX}/_doc/{digest}?refresh=true")     # revoked after migration
    assert keystore_migrate.main(["--from-file", str(path)]) == 1
    assert "refused" in capsys.readouterr().err
    with pytest.raises(RuntimeError):
        es("GET", f"/{keystore_es.INDEX}/_doc/{digest}")                 # not resurrected
    assert keystore_migrate.main(["--from-file", str(path), "--force"]) == 0
    _clear_sentinel()


def test_sentinel_is_invisible_to_key_listings(es_keys):
    auth.create_key("ks-sent", person="sentp")
    keystore_es.mark_migrated()
    assert keystore_es.migration_done() is True
    assert keystore_es.SENTINEL_ID not in keystore_es.list_all()
    assert all(v.get("owner_id") for v in keystore_es.list_all().values())
    assert "ks-sent" in worker.known_owners()
    assert auth.persons_of("ks-sent") == {"sentp"}
    assert auth.persons_of(None) == set()
    _clear_sentinel()


def test_migration_skips_invalid_digest_and_role(es_keys, tmp_path, capsys):
    _clear_sentinel()
    _t, good = auth._mint()
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({
        good: {"owner_id": "ks-valid", "role": "member"},
        "migration-sentinel": {"owner_id": "ks-evil", "role": "owner"},
        "A" * 64: {"owner_id": "ks-upper", "role": "member"},
        "b" * 63: {"owner_id": "ks-short", "role": "member"},
        "c" * 64: {"owner_id": "ks-badrole", "role": "root"}}))
    assert keystore_migrate.main(["--from-file", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"created": 1, "existing": 0, "skipped": 4}
    assert "ks-evil" not in worker.known_owners()
    _clear_sentinel()


def test_write_paths_validate_id_and_role(es_keys):
    with pytest.raises(ValueError):
        keystore_es.put_digest("not-a-digest", {"owner_id": "ks-v", "role": "member"})
    with pytest.raises(ValueError):
        keystore_es.put_digest("d" * 64, {"owner_id": "ks-v", "role": "root"})
    with pytest.raises(ValueError):
        keystore_es._create_doc("../x", {"owner_id": "ks-v", "role": "member"})


def test_missing_index_fails_closed_not_unknown(es_keys):
    es("DELETE", f"/{keystore_es.INDEX}")
    keystore_es._cache.clear()
    try:
        with pytest.raises(auth.KeyStoreUnavailable):    # index_not_found is a fault, not "unknown key"
            auth.owner_of("ame_" + "e" * 64)
    finally:
        keystore_es._index_ready = False
        keystore_es.ensure_index()


def test_wrong_mapping_is_refused(es_keys):
    es("DELETE", f"/{keystore_es.INDEX}")
    es("PUT", f"/{keystore_es.INDEX}", {"mappings": {"properties": {"owner_id": {"type": "text"}}}})
    keystore_es._index_ready = False
    try:
        with pytest.raises(auth.KeyStoreUnavailable):
            keystore_es.ensure_index()
        with pytest.raises(auth.KeyStoreUnavailable):
            auth.owner_of("ame_" + "f" * 64)
    finally:
        es("DELETE", f"/{keystore_es.INDEX}")
        keystore_es._index_ready = False
        keystore_es.ensure_index()


def test_recreated_index_gets_strict_mapping_from_template(es_keys):
    es("DELETE", f"/{keystore_es.INDEX}")
    keystore_es._index_ready = False
    token = auth.create_key("ks-recreate", person="rc")      # ES auto-creates the index on write
    props = es("GET", f"/{keystore_es.INDEX}")[keystore_es.INDEX]["mappings"]["properties"]
    assert props["owner_id"]["type"] == "keyword"
    assert auth.persons_of("ks-recreate") == {"rc"} and auth.owner_of(token)


def test_ttl_is_capped_and_bad_values_rejected(monkeypatch):
    monkeypatch.setenv("AMES_KEY_CACHE_TTL_S", "100000")
    assert keystore_es._ttl() == keystore_es.MAX_TTL_S
    for bad in ("nan", "inf", "-1", "abc"):
        monkeypatch.setenv("AMES_KEY_CACHE_TTL_S", bad)
        assert keystore_es._ttl() == keystore_es.DEFAULT_TTL_S
        with pytest.raises(ValueError):
            keystore_es.validate_config()
    monkeypatch.setenv("AMES_KEY_CACHE_TTL_S", "0")
    keystore_es.validate_config()
    assert keystore_es._ttl() == 0


def test_startup_refuses_invalid_key_store_config(monkeypatch, caplog):
    from app import main
    monkeypatch.setenv("AMES_KEY_STORE", "redis")
    with pytest.raises(ValueError):
        main._startup()
    monkeypatch.setenv("AMES_KEY_STORE", "es")
    monkeypatch.setenv("AMES_KEY_CACHE_TTL_S", "nan")
    with pytest.raises(ValueError):
        main._startup()
    assert "invalid configuration" in caplog.text


def test_list_all_logs_error_when_truncated(monkeypatch, caplog):
    seen = {}

    def fake(method, path, body=None, timeout=30):
        seen["body"] = body
        return {"hits": {"total": {"value": 5, "relation": "eq"}, "hits": []}}
    monkeypatch.setattr(keystore_es, "es", fake)
    monkeypatch.setattr(keystore_es, "MAX_KEYS", 2)
    with caplog.at_level(logging.ERROR, logger=keystore_es.log.name):
        assert keystore_es.list_all() == {}
    assert seen["body"]["track_total_hits"] is True
    assert "only 2 are read" in caplog.text


def test_health_uses_the_short_timeout(monkeypatch):
    seen = {}

    def fake(method, path, body=None, timeout=30):
        seen["timeout"] = timeout
        return {"count": 0}
    monkeypatch.setattr(keystore_es, "es", fake)
    assert keystore_es.health() is True and seen["timeout"] == 5
    assert keystore_es.health(2.5) is True and seen["timeout"] == 2.5
    from app import main
    assert main.HEALTH_ES_TIMEOUT_S == 5


def test_zero_owner_probe_only_runs_when_no_owners(monkeypatch):
    called = []
    monkeypatch.setattr(worker, "_warn_if_zero_owners_with_data", lambda: called.append(1))
    monkeypatch.setattr(worker, "_consolidate_owners", lambda owners: {})
    monkeypatch.setattr(worker, "known_owners", lambda: [])
    worker.run_once()
    assert called == [1]
    called.clear()
    monkeypatch.setattr(worker, "known_owners", lambda: ["someone"])
    worker.run_once()
    assert called == []


def test_migration_missing_file_reports_failure(es_keys, tmp_path, capsys):
    _clear_sentinel()
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
