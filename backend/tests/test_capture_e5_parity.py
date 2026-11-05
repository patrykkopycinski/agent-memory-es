"""P2-2: the parity fixture is produced by scripts/capture_e5_parity.py (stdlib, read-only against
ES). The script is exercised against a fake ES; the texts file must contain inputs longer than the
512-token window; a fixture without such items is flagged as a skip, never silently green."""
import importlib.util
import json
import os
import sys
import threading
import http.server

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest

from model_cache import E5, model_or_skip  # noqa: E402

BACKEND = os.path.join(os.path.dirname(__file__), "..")
TEXTS = os.path.join(BACKEND, "tests", "fixtures", "e5_parity_texts.json")
FIXTURE = os.path.join(BACKEND, "tests", "fixtures", "e5_parity.json")


def _load_script():
    spec = importlib.util.spec_from_file_location("capture", os.path.join(BACKEND, "scripts", "capture_e5_parity.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_es():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen.append(("GET", self.path))
            self._send({"version": {"number": "9.9.9"}} if self.path == "/"
                       else {"nodes": {"n1": {"os": {"arch": "aarch64"}}}})

        def do_POST(self):
            seen.append(("POST", self.path))
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send({"text_embedding": [{"embedding": [float(len(t) % 7)] * 384} for t in body["input"]]})

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def test_capture_writes_fixture_with_arch_and_model_id_and_is_read_only(tmp_path, fake_es):
    url, seen = fake_es
    out = tmp_path / "fx.json"
    _load_script().main(["--es", url, "--out", str(out)])
    fx = json.loads(out.read_text())
    with open(TEXTS) as fh:
        texts = json.load(fh)["texts"]
    assert [i["text"] for i in fx["items"]] == texts
    assert fx["dims"] == 384 and all(len(i["vector"]) == 384 for i in fx["items"])
    assert fx["source"]["model_id"] == ".multilingual-e5-small-elasticsearch"
    assert fx["source"]["es_node_arch"] == ["aarch64"] and fx["source"]["capture_host_arch"]
    assert {m for m, _ in seen} <= {"GET", "POST"}
    assert all(p.startswith("/_inference/text_embedding/") for m, p in seen if m == "POST")


def test_capture_fails_loudly_on_unreachable_es_and_writes_nothing(tmp_path):
    out = tmp_path / "fx.json"
    with pytest.raises(Exception):
        _load_script().main(["--es", "http://127.0.0.1:1", "--out", str(out)])
    assert not out.exists()


def _token_counts(texts):
    _, tok = model_or_skip(E5)
    tok.no_truncation()
    return [len(tok.encode(t).ids) for t in texts]


def test_texts_file_has_inputs_beyond_the_512_token_window():
    with open(TEXTS) as fh:
        texts = json.load(fh)["texts"]
    assert sum(1 for n in _token_counts(texts) if n > 512) >= 2


def test_committed_fixture_covers_long_inputs():
    with open(FIXTURE) as fh:
        items = json.load(fh)["items"]
    long_items = sum(1 for n in _token_counts([i["text"] for i in items]) if n > 512)
    if long_items < 2:
        from model_cache import _unavailable
        _unavailable("fixture has no >512-token items yet: run scripts/capture_e5_parity.py against "
                     "the reference ES and commit the output")
