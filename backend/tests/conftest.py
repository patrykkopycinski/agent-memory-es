import os
import sys

# SAFETY GUARD — the shared module (tests/_safety.py) enforces the same
# checks for pytest AND script-style runs; import it first here too so the
# guard runs before any test module (and any `app` module) is imported.
sys.path.insert(0, os.path.dirname(__file__))
import _safety  # noqa: F401

# Single session-wide key store: modules must not fight over AMES_API_KEYS_FILE
# (pytest imports all test modules before running any test).
os.environ["AMES_API_KEYS_FILE"] = "/tmp/ames_session_keys.json"
if os.path.exists(os.environ["AMES_API_KEYS_FILE"]):
    os.remove(os.environ["AMES_API_KEYS_FILE"])

# Create prefixed indices at import time: test modules clean them at module
# import (collection), before any fixture can run.
import pathlib  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app import store, tombstone as _tb  # noqa: E402
from app import mental_models as _mm, pages as _pg  # noqa: E402

store.ensure_indices()
_mm.ensure_models_index()
_pg.ensure_pages_index()
_tb._ensure_index()
