"""Unit tests: AMES backend is mocked with httpx.MockTransport; the MCP layer is
exercised through FastMCP's own tool listing/calling (no network)."""
import asyncio
import json
import logging
import sys
import pathlib

import httpx
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from mcp_server.client import AmesClient, AmesError  # noqa: E402
from mcp_server.server import build_server, make_server_from_env, write_enabled  # noqa: E402

KEY = "ame_SECRETKEY_0123456789"


def _backend(calls, results=None, status=200):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        calls.append((req.method, req.url.path, req.headers.get("x-api-key"), body))
        if status != 200:
            return httpx.Response(status, json={"detail": "nope"})
        if req.url.path == "/memory/recall":
            return httpx.Response(200, json={"query": body["query"], "abstained": False,
                                             "results": results or [], "by_kind": {}})
        if req.url.path == "/memory/reflect":
            return httpx.Response(200, json={"question": body["question"], "answer": "A",
                                             "sources": [], "synthesized": True})
        if req.url.path == "/memory/stats":
            return httpx.Response(200, json={"owner_id": "me", "counts": {}})
        if req.url.path == "/memory/retain":
            return httpx.Response(200, json={"_id": "x1"})
        return httpx.Response(404)
    return handler


def _client(calls, **kw):
    return AmesClient("http://ames.test", KEY, transport=httpx.MockTransport(_backend(calls, **kw)))


def _call(mcp, name, args):
    async def go():
        return await mcp.call_tool(name, args)
    res = asyncio.run(go())
    # FastMCP returns (content, structured) or content depending on version
    if isinstance(res, tuple):
        res = res[1] if isinstance(res[1], dict) else res[0]
    if isinstance(res, list):
        return json.loads(res[0].text)
    return res


def _tools(mcp):
    return {t.name: t for t in asyncio.run(mcp.list_tools())}


ITEMS = [
    {"id": "1", "kind": "semantic", "score": 0.9, "visibility": "private", "text": "p", "embedding": [0.1]},
    {"id": "2", "kind": "semantic", "score": 0.8, "visibility": "team", "text": "t", "rrf": 1},
    {"id": "3", "kind": "episodic", "score": 0.7, "visibility": "common", "text": "c"},
]


def test_readonly_by_default_hides_retain():
    mcp = build_server(_client([]))
    assert set(_tools(mcp)) == {"memory_recall", "memory_reflect", "memory_stats"}


def test_retain_registered_only_with_allow_write():
    mcp = build_server(_client([]), allow_write=True)
    assert "memory_retain" in _tools(mcp)


@pytest.mark.parametrize("val,exp", [(None, False), ("", False), ("0", False), ("false", False),
                                     ("1", True), ("true", True), ("YES", True), ("on", True)])
def test_allow_write_env_parsing(val, exp):
    env = {} if val is None else {"ALLOW_WRITE": val}
    assert write_enabled(env) is exp


def test_make_server_from_env_gating():
    base = {"AMES_API_KEY": KEY, "AMES_URL": "http://ames.test"}
    assert "memory_retain" not in _tools(make_server_from_env(base))
    assert "memory_retain" in _tools(make_server_from_env({**base, "ALLOW_WRITE": "1"}))


def test_missing_key_refuses_to_start():
    with pytest.raises(AmesError, match="AMES_API_KEY"):
        make_server_from_env({"AMES_URL": "http://ames.test"})


def test_tool_schemas():
    t = _tools(build_server(_client([]), allow_write=True))
    rec = t["memory_recall"].inputSchema
    assert rec["required"] == ["query"]
    assert set(rec["properties"]) == {"query", "k", "visibility", "kinds", "as_of"}
    assert "private" in json.dumps(rec["properties"]["visibility"])
    assert t["memory_reflect"].inputSchema["required"] == ["question"]
    assert t["memory_stats"].inputSchema.get("properties", {}) == {}
    ret = t["memory_retain"].inputSchema
    assert ret["required"] == ["text"]
    assert "private" in json.dumps(ret["properties"]["visibility"])


def test_recall_sends_key_header_and_slims_results():
    calls = []
    mcp = build_server(_client(calls, results=ITEMS))
    out = _call(mcp, "memory_recall", {"query": "q", "k": 3})
    assert calls[0][1:] == ("/memory/recall", KEY, {"query": "q", "size": 3})
    assert [r["id"] for r in out["results"]] == ["1", "2", "3"]
    assert all("embedding" not in r and "rrf" not in r for r in out["results"])


def test_recall_visibility_filter_is_client_side_and_overfetches():
    calls = []
    mcp = build_server(_client(calls, results=ITEMS))
    out = _call(mcp, "memory_recall", {"query": "q", "k": 2, "visibility": "team"})
    assert [r["id"] for r in out["results"]] == ["2"]
    assert calls[0][3]["size"] >= 8  # over-fetched
    assert "visibility" not in calls[0][3]  # backend has no such param


def test_recall_k_clamped():
    calls = []
    mcp = build_server(_client(calls))
    _call(mcp, "memory_recall", {"query": "q", "k": 9999})
    assert calls[0][3]["size"] == 50
    _call(mcp, "memory_recall", {"query": "q", "k": 0})
    assert calls[1][3]["size"] == 1


def test_recall_rejects_bad_visibility():
    mcp = build_server(_client([]))
    with pytest.raises(Exception):
        _call(mcp, "memory_recall", {"query": "q", "visibility": "everyone"})


def test_reflect_and_stats():
    calls = []
    mcp = build_server(_client(calls))
    assert _call(mcp, "memory_reflect", {"question": "why?"})["answer"] == "A"
    assert _call(mcp, "memory_stats", {})["owner_id"] == "me"
    assert [c[:2] for c in calls] == [("POST", "/memory/reflect"), ("GET", "/memory/stats")]


def test_retain_passthrough_when_enabled():
    calls = []
    mcp = build_server(_client(calls), allow_write=True)
    _call(mcp, "memory_retain", {"text": "fact", "visibility": "team"})
    assert calls[0][1] == "/memory/retain"
    assert calls[0][3] == {"kind": "semantic", "text": "fact", "visibility": "team"}


def test_disabled_retain_cannot_be_called():
    mcp = build_server(_client([]))
    with pytest.raises(Exception):
        _call(mcp, "memory_retain", {"text": "x"})


def test_401_error_is_clear_and_does_not_leak_key():
    mcp = build_server(_client([], status=401))
    with pytest.raises(Exception) as ei:
        _call(mcp, "memory_recall", {"query": "q"})
    assert "401" in str(ei.value) and KEY not in str(ei.value)


def test_key_not_in_repr_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    c = _client([])
    assert KEY not in repr(c) and KEY not in str(c)
    build_server(c)
    c.recall("q", 1)
    assert KEY not in caplog.text


def test_unreachable_backend_error_is_clean():
    def boom(req):
        raise httpx.ConnectError("conn refused " + KEY)
    c = AmesClient("http://ames.test", KEY, transport=httpx.MockTransport(boom))
    with pytest.raises(AmesError) as ei:
        c.stats()
    assert KEY not in str(ei.value) and "unreachable" in str(ei.value)
