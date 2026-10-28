"""Issue #8: on a cluster where the memory indices do not exist yet, the library
entry points must bootstrap them instead of 404ing. The API's startup hook does
this, but memory.recall/retain are also called directly (scripts, hermes_provider).

Each case runs in a subprocess with a fresh index prefix, so conftest's import-time
ensure_indices() (which created the amtest_ indices) cannot mask the bug.
"""
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import uuid

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
SUFFIXES = ("am_episodic", "am_semantic", "am_procedural", "am_rejected",
            "am_models", "am_pages", "am_mutations")


def _delete(prefix):
    for s in SUFFIXES:
        req = urllib.request.Request(f"{os.environ['AMES_ES_URL']}/{prefix}{s}", method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise


@pytest.fixture
def bare_prefix():
    # must stay under the amtest_ prefix: the test guard refuses anything else
    prefix = f"{os.environ.get('AMES_INDEX_PREFIX', 'amtest_')}boot{uuid.uuid4().hex[:8]}_"
    yield prefix
    _delete(prefix)


def _run(prefix, code):
    env = {**os.environ, "AMES_INDEX_PREFIX": prefix, "AMES_TEST_GUARD": "1"}
    p = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_recall_before_any_write_returns_empty_not_404(bare_prefix):
    out = _run(bare_prefix, (
        "import json; from app import memory\n"
        "r = memory.recall('bootowner', 'anything at all', rerank=False)\n"
        "print(json.dumps({'n': len(r['results'])}))\n"))
    assert out == {"n": 0}


def test_retain_then_recall_on_bare_cluster(bare_prefix):
    out = _run(bare_prefix, (
        "import json; from app import memory\n"
        "memory.retain('bootowner', 'semantic', 'The kettle in room 4 is descaled every Friday')\n"
        "r = memory.recall('bootowner', 'when is the kettle descaled', kinds=['semantic'], rerank=False)\n"
        "print(json.dumps({'hit': any('kettle' in h['text'] for h in r['results'])}))\n"))
    assert out == {"hit": True}
