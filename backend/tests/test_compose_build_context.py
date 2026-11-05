"""Every compose `build:` context must resolve (against the compose file's own directory) to a
directory holding a Dockerfile. docker-compose.r7.yml once said `build: ..`, the repo root, which
has none, so `docker compose up` could not build the image the baked-model change depends on."""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
COMPOSE = sorted(glob.glob(os.path.join(BACKEND, "docker-compose*.yml")))


def _contexts(path):
    return re.findall(r"^\s+build:\s*(\S+)\s*(?:#.*)?$", open(path).read(), re.M)


def test_compose_files_are_found():
    assert COMPOSE and any(_contexts(p) for p in COMPOSE)


@pytest.mark.parametrize("path", COMPOSE, ids=os.path.basename)
def test_build_context_has_a_dockerfile(path):
    for ctx in _contexts(path):
        resolved = os.path.normpath(os.path.join(os.path.dirname(path), ctx))
        assert os.path.isfile(os.path.join(resolved, "Dockerfile")), f"{path}: build: {ctx} -> {resolved}"
