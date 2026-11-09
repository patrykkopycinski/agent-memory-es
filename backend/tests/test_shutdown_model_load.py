"""Shutdown waits (bounded) for an in-flight local-model load before flushing spans: tearing the
interpreter down while the warm-up or probe-load thread is inside the native ONNX load can abort
the process and lose the final batch."""
import logging
import os
import subprocess
import sys
import textwrap
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

from app import embeddings, main, telemetry as _otel  # noqa: E402

BACKEND = os.path.join(os.path.dirname(__file__), "..")


def _slow_thread(name, seconds, done):
    t = threading.Thread(target=lambda: (done.wait(seconds), done.set()), daemon=True, name=name)
    t.start()
    return t


@pytest.fixture()
def flush_log(monkeypatch):
    """Records, at flush time, which loader threads were still alive."""
    seen = []

    def fake_shutdown():
        seen.append([t.name for t in (getattr(main, "_warmup_thread", None), embeddings._loader)
                     if t is not None and t.is_alive()])

    monkeypatch.setattr(_otel, "shutdown", fake_shutdown)
    monkeypatch.setattr(main, "_warmup_thread", None, raising=False)
    monkeypatch.setattr(embeddings, "_loader", None)
    return seen


def test_flush_waits_for_warmup_thread(flush_log, monkeypatch):
    monkeypatch.setattr(main, "_warmup_thread", _slow_thread("embed-warmup", 1.0, threading.Event()), raising=False)
    main._shutdown()
    assert flush_log == [[]]


def test_flush_waits_for_probe_load_thread(flush_log, monkeypatch):
    monkeypatch.setattr(embeddings, "_loader", _slow_thread("embed-probe-load", 1.0, threading.Event()))
    main._shutdown()
    assert flush_log == [[]]


def test_wait_is_bounded_warns_and_still_flushes(flush_log, monkeypatch, caplog):
    stuck = threading.Event()
    monkeypatch.setenv("AMES_SHUTDOWN_LOADER_WAIT_S", "0.3")
    monkeypatch.setattr(main, "_warmup_thread", _slow_thread("embed-warmup", 30, stuck), raising=False)
    try:
        with caplog.at_level(logging.WARNING, logger="ames.shutdown"):
            main._shutdown()
        assert flush_log == [["embed-warmup"]]
        assert "embed-warmup still loading" in caplog.text
    finally:
        stuck.set()


def test_wait_budget_is_shared_between_loaders(flush_log, monkeypatch):
    stuck = threading.Event()
    monkeypatch.setenv("AMES_SHUTDOWN_LOADER_WAIT_S", "0.4")
    monkeypatch.setattr(main, "_warmup_thread", _slow_thread("embed-warmup", 30, stuck), raising=False)
    monkeypatch.setattr(embeddings, "_loader", _slow_thread("embed-probe-load", 30, stuck))
    try:
        import time
        t0 = time.monotonic()
        main._shutdown()
        assert time.monotonic() - t0 < 0.9
    finally:
        stuck.set()


@pytest.mark.parametrize("raw,expected", [("", 10.0), ("abc", 10.0), ("-3", 0.0), ("2.5", 2.5)])
def test_wait_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("AMES_SHUTDOWN_LOADER_WAIT_S", raw)
    assert main._loader_wait_s() == expected


def test_shutdown_without_loaders_flushes_immediately(flush_log):
    main._shutdown()
    assert flush_log == [[]]


def test_process_flushes_after_slow_load_and_exits_zero():
    """Separate process: the warm-up is mid-load (slow fake) when the shutdown hook runs. The
    flush must come after the load finished, and the process must exit 0."""
    code = textwrap.dedent("""
        import time
        from app import embeddings, main, telemetry
        state = {"loaded": False}
        def slow_model():
            time.sleep(2)
            state["loaded"] = True
        embeddings.backend = lambda: "local"
        embeddings._local_model = slow_model
        main._health_cached = lambda: {}
        main._warm_embedder_in_background()
        time.sleep(0.2)
        telemetry.shutdown = lambda: print("FLUSH loaded=%s" % state["loaded"], flush=True)
        main._shutdown()
    """)
    env = dict(os.environ, AMES_OTEL_ENABLED="0")
    env.pop("AMES_SHUTDOWN_LOADER_WAIT_S", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "FLUSH loaded=True" in out.stdout, out.stdout + out.stderr[-1000:]
