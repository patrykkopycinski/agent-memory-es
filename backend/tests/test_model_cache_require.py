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


def test_auto_detect_test_fails_not_skips_without_runtime_when_required(tmp_path, monkeypatch):
    """The auto-detect test in test_local_model_loading used a bare importorskip, which stayed a
    skip under AMES_REQUIRE_MODEL=1. Call its body with the runtime made unimportable."""
    import test_local_model_loading as t
    monkeypatch.setenv("AMES_REQUIRE_MODEL", "1")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    try:
        t.test_auto_detect_prefers_local_when_model_dir_is_valid(tmp_path, monkeypatch, None)
    except BaseException as outcome:        # Failed/Skipped are BaseException subclasses
        assert isinstance(outcome, pytest.fail.Exception), f"expected a failure, got {outcome!r}"
    else:
        pytest.fail("test body did not fail without the runtime")
