"""P2-3/P2-4/P3-6: the local model is never fetched at runtime, a wrong or unmarked model dir is
refused with the offending path, the image pins (revision + sha256) agree with the code, and
auto-detect prefers `local` only when the model dir is valid."""
import hashlib
import http.server
import importlib.util
import os
import re
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest

from app import embeddings  # noqa: E402

BACKEND = os.path.join(os.path.dirname(__file__), "..")
REV = embeddings.LOCAL_MODEL_REVISION


def _no_network(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("runtime network access")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    return calls


def _write_model(dest, marker=REV):
    for rel in embeddings.MODEL_FILES:
        path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"x")
    if marker is not None:
        with open(os.path.join(dest, embeddings.REVISION_FILE), "w") as fh:
            fh.write(marker + "\n")


# ------------------------------------------------------------------ P2-3: no runtime download
def test_missing_files_raise_naming_the_path_and_never_hit_the_network(tmp_path, monkeypatch):
    monkeypatch.delenv("AMES_LOCAL_MODEL_ALLOW_FETCH", raising=False)
    calls = _no_network(monkeypatch)
    dest = str(tmp_path / "absent")
    with pytest.raises(embeddings.ModelUnavailable) as exc:
        embeddings.load_local(dest)
    assert dest in str(exc.value) and "model.onnx" in str(exc.value)
    assert calls == []


def test_unmarked_model_dir_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("AMES_LOCAL_MODEL_ALLOW_FETCH", raising=False)
    _write_model(str(tmp_path), marker=None)
    with pytest.raises(embeddings.ModelUnavailable, match="REVISION marker missing"):
        embeddings.load_local(str(tmp_path))


def test_marker_for_another_revision_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("AMES_LOCAL_MODEL_ALLOW_FETCH", raising=False)
    _write_model(str(tmp_path), marker="0" * 40)
    with pytest.raises(embeddings.ModelUnavailable, match="!= pinned"):
        embeddings.load_local(str(tmp_path))


def test_embed_error_names_the_missing_path(tmp_path, monkeypatch):
    monkeypatch.delenv("AMES_LOCAL_MODEL_ALLOW_FETCH", raising=False)
    gone = str(tmp_path / "gone")
    monkeypatch.setattr(embeddings, "LOCAL_MODEL_DIR", gone)
    monkeypatch.setattr(embeddings, "_local", None)
    monkeypatch.setattr(embeddings, "_backend", "local")
    vecs, err = embeddings.embed_safe(["hello"])
    assert not vecs
    assert gone in err


def test_opt_in_env_downloads_and_writes_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("AMES_LOCAL_MODEL_ALLOW_FETCH", "1")
    urls = []

    class Resp:
        def __init__(self):
            self.data = [b"bytes", b""]

        def read(self, n=-1):
            return self.data.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(url, timeout=None):
        urls.append(url)
        return Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    embeddings.fetch_model(str(tmp_path))
    assert len(urls) == 2 and all(f"/resolve/{REV}/" in u for u in urls)
    assert embeddings.model_problem(str(tmp_path)) is None


def test_load_local_fetches_only_with_the_opt_in(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(embeddings, "fetch_model", lambda *a, **k: fetched.append(a))
    monkeypatch.delenv("AMES_LOCAL_MODEL_ALLOW_FETCH", raising=False)
    with pytest.raises(embeddings.ModelUnavailable):
        embeddings.load_local(str(tmp_path / "x"))
    assert fetched == []
    monkeypatch.setenv("AMES_LOCAL_MODEL_ALLOW_FETCH", "1")
    with pytest.raises(embeddings.ModelUnavailable):      # the fake fetch wrote nothing
        embeddings.load_local(str(tmp_path / "x"))
    assert len(fetched) == 1


# ------------------------------------------------------------------ P2-4: pins agree
def _dockerfile_args():
    with open(os.path.join(BACKEND, "Dockerfile")) as fh:
        return dict(re.findall(r"^ARG (\w+)=(\S+)$", fh.read(), re.M))


def test_dockerfile_revision_equals_code_default():
    assert _dockerfile_args()["E5_REV"] == REV


def test_dockerfile_sha256_pins_present_and_wellformed():
    args = _dockerfile_args()
    for key in ("E5_MODEL_SHA256", "E5_TOKENIZER_SHA256"):
        assert re.fullmatch(r"[0-9a-f]{64}", args[key]), key


def test_requirements_pin_runtime_exactly():
    with open(os.path.join(BACKEND, "requirements.txt")) as fh:
        reqs = fh.read()
    for pkg in ("onnxruntime", "tokenizers", "numpy"):
        assert re.search(rf"^{pkg}==\d+(\.\d+)+$", reqs, re.M), pkg


@pytest.fixture
def file_server(tmp_path):
    root = tmp_path / "srv"
    rel = root / "org" / "model" / "resolve" / "rev1"
    (rel / "onnx").mkdir(parents=True)
    (rel / "onnx" / "model.onnx").write_bytes(b"weights")
    (rel / "tokenizer.json").write_bytes(b"{}")

    def handler(*a, **k):
        return http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **k)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _bake(argv):
    spec = importlib.util.spec_from_file_location("bake", os.path.join(BACKEND, "scripts", "bake_e5_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(argv)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_bake_verifies_hashes_and_writes_the_marker(tmp_path, file_server):
    dest = str(tmp_path / "out")
    _bake(["--dest", dest, "--repo", "org/model", "--rev", "rev1", "--base", file_server,
           "--sha", f"onnx/model.onnx={_sha(b'weights')}", "--sha", f"tokenizer.json={_sha(b'{}')}"])
    with open(os.path.join(dest, "REVISION")) as fh:
        assert fh.read().strip() == "rev1"


def test_bake_hash_mismatch_fails_without_a_marker(tmp_path, file_server):
    dest = str(tmp_path / "out")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        _bake(["--dest", dest, "--repo", "org/model", "--rev", "rev1", "--base", file_server,
               "--sha", f"onnx/model.onnx={'0' * 64}", "--sha", f"tokenizer.json={_sha(b'{}')}"])
    assert not os.path.exists(os.path.join(dest, "REVISION"))
    assert not os.path.exists(os.path.join(dest, "onnx", "model.onnx"))


# ------------------------------------------------------------------ P3-6: auto-detect
@pytest.fixture
def unpinned(monkeypatch):
    monkeypatch.delenv("AMES_EMBED_BACKEND", raising=False)
    monkeypatch.setattr(embeddings, "_backend", None)
    yield
    embeddings._backend = None


def test_auto_detect_prefers_local_when_model_dir_is_valid(tmp_path, monkeypatch, unpinned):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    _write_model(str(tmp_path))
    monkeypatch.setattr(embeddings, "LOCAL_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(embeddings, "_es_embed", lambda t: pytest.fail("probed ES"))
    assert embeddings.backend() == "local"


def test_auto_detect_falls_back_without_a_valid_model(tmp_path, monkeypatch, unpinned):
    monkeypatch.setattr(embeddings, "LOCAL_MODEL_DIR", str(tmp_path / "none"))
    monkeypatch.setattr(embeddings, "_es_embed", lambda t: [[0.0]])
    assert embeddings.backend() == "es"


def test_explicit_pin_beats_auto_detect(tmp_path, monkeypatch, unpinned):
    _write_model(str(tmp_path))
    monkeypatch.setattr(embeddings, "LOCAL_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("AMES_EMBED_BACKEND", "omniroute")
    assert embeddings.backend() == "omniroute"
