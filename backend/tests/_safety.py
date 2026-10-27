"""Shared ES safety guard — MUST be imported before any `app` module.

Imported first by tests/conftest.py (pytest path) AND as the first app-adjacent
import by every test module (script path: `python tests/test_x.py` runs the
module directly and bypasses conftest entirely — this file is the only guard
that fires there).

The default-free app requires AMES_ES_URL, but a stray env (e.g. an old shell
export pointing at the prod tunnel, localhost:9268) would still pass through.
This module refuses to let tests run against anything but a throwaway cluster.
"""
import os

_es_url = os.environ.get("AMES_ES_URL")
if (not _es_url
        or _es_url.rstrip("/").endswith(":9268")
        or not os.environ.get("AMES_INDEX_PREFIX", "").startswith("amtest_")):
    raise SystemExit(
        "REFUSING TO RUN TESTS: unsafe Elasticsearch target.\n"
        "Set AMES_ES_URL explicitly to a throwaway cluster (port must NOT be "
        "9268 — that is the prod tunnel) and AMES_INDEX_PREFIX=amtest_...")

# Belt-and-braces path guard (see store._guard_path) — active under tests.
os.environ["AMES_TEST_GUARD"] = "1"
