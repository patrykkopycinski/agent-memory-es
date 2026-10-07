import os

# Single session-wide key store: modules must not fight over AMES_API_KEYS_FILE
# (pytest imports all test modules before running any test).
os.environ["AMES_API_KEYS_FILE"] = "/tmp/ames_session_keys.json"
if os.path.exists(os.environ["AMES_API_KEYS_FILE"]):
    os.remove(os.environ["AMES_API_KEYS_FILE"])
# Test isolation: never read/write the live cluster's shared data.
os.environ["AMES_INDEX_PREFIX"] = "amtest_"

# Create prefixed indices at import time: test modules clean them at module
# import (collection), before any fixture can run.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app import store, tombstone as _tb
from app import mental_models as _mm, pages as _pg

if os.environ.get("AMES_TEST_OFFLINE") != "1":
    store.ensure_indices()
    _mm.ensure_models_index()
    _pg.ensure_pages_index()
    _tb._ensure_index()
