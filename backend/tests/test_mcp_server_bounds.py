"""The stdio MCP server is a stdlib script outside the backend package: its recall `size` schema
and dispatch must carry the same bound as the HTTP API (memory.MAX_RECALL_SIZE)."""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest

from app import memory  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "ames_mcp_server.py")


@pytest.fixture
def server():
    spec = importlib.util.spec_from_file_location("ames_mcp_server", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_schema_bounds_match_the_backend(server):
    tool = next(t for t in server.TOOLS if t["name"] == "ames_recall")
    size = tool["inputSchema"]["properties"]["size"]
    assert size["minimum"] == 1 and size["maximum"] == memory.MAX_RECALL_SIZE


@pytest.mark.parametrize("asked,sent", [(5_000_000, memory.MAX_RECALL_SIZE), (0, 1), (-4, 1), (8, 8),
                                        (memory.MAX_RECALL_SIZE, memory.MAX_RECALL_SIZE)])
def test_dispatch_clamps_size(server, monkeypatch, asked, sent):
    seen = {}
    monkeypatch.setattr(server, "http", lambda url, key, path, body: seen.update(body) or {})
    server.dispatch("http://x", "k", "ames_recall", {"query": "q", "size": asked})
    assert seen["size"] == sent


def test_dispatch_default_size_is_unchanged(server, monkeypatch):
    seen = {}
    monkeypatch.setattr(server, "http", lambda url, key, path, body: seen.update(body) or {})
    server.dispatch("http://x", "k", "ames_recall", {"query": "q"})
    assert seen["size"] == 8
