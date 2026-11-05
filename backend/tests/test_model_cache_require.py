"""AMES_REQUIRE_MODEL=1 turns the model-dependent skips into failures; unset keeps them skips."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest

import model_cache


@pytest.fixture
def empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache, "CACHE", str(tmp_path))
    monkeypatch.delenv("AMES_LOCAL_MODEL_ALLOW_FETCH", raising=False)


def test_missing_model_skips_by_default(empty_cache, monkeypatch):
    monkeypatch.delenv("AMES_REQUIRE_MODEL", raising=False)
    with pytest.raises(pytest.skip.Exception):
        model_cache.model_or_skip(model_cache.E5)


def test_missing_model_fails_when_required(empty_cache, monkeypatch):
    monkeypatch.setenv("AMES_REQUIRE_MODEL", "1")
    with pytest.raises(pytest.fail.Exception, match="AMES_REQUIRE_MODEL=1"):
        model_cache.model_or_skip(model_cache.E5)


def test_missing_runtime_fails_when_required(monkeypatch):
    monkeypatch.setenv("AMES_REQUIRE_MODEL", "1")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)       # makes `import onnxruntime` raise
    with pytest.raises(pytest.fail.Exception, match="onnxruntime is not installed"):
        model_cache.model_or_skip(model_cache.E5)
