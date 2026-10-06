import os
import sys

# SAFETY GUARD — must run before ANY test module is imported (pytest loads
# conftest first) and before the store module reads AMES_ES_URL.
# The default ES URL (localhost:9268) is an ssh tunnel to the PRODUCTION
# AMES Elasticsearch cluster. Previous test runs polluted prod. Tests must
# only ever run against a throwaway cluster: AMES_ES_URL must be set
# explicitly, its port must not be 9268, and the index prefix must be a
# throwaway amtest_ prefix.
_es_url = os.environ.get("AMES_ES_URL")
if (not _es_url
        or _es_url.rstrip("/").endswith(":9268")
        or not os.environ.get("AMES_INDEX_PREFIX", "").startswith("amtest_")):
    import pytest
    pytest.exit(
        "REFUSING TO RUN TESTS: unsafe Elasticsearch target.\n"
        "Set AMES_ES_URL explicitly to a throwaway cluster (port must NOT be "
        "9268 — that is the prod tunnel) and AMES_INDEX_PREFIX=amtest_...",
        returncode=99)

# Single session-wide key store: modules must not fight over AMES_API_KEYS_FILE
# (pytest imports all test modules before running any test).
os.environ["AMES_API_KEYS_FILE"] = "/tmp/ames_session_keys.json"
if os.path.exists(os.environ["AMES_API_KEYS_FILE"]):
    os.remove(os.environ["AMES_API_KEYS_FILE"])
# Test isolation: never read/write the live cluster's shared data.
os.environ["AMES_INDEX_PREFIX"] = os.environ.get("AMES_INDEX_PREFIX")

# Belt-and-braces path guard (see store._guard_path) — active only under tests.
os.environ["AMES_TEST_GUARD"] = "1"

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
