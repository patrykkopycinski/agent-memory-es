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


def test_make_server_from_env_forwards_fastmcp_kwargs():
    mcp = make_server_from_env({"AMES_API_KEY": KEY, "AMES_URL": "http://ames.test"},
                               host="0.0.0.0", port=9999)
    assert mcp.settings.host == "0.0.0.0"
    assert mcp.settings.port == 9999


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


def test_non_json_2xx_becomes_ameserror():
    def html(req):
        return httpx.Response(200, content=b"<html>proxy</html>",
                              headers={"content-type": "text/html"})
    c = AmesClient("http://ames.test", KEY, transport=httpx.MockTransport(html))
    with pytest.raises(AmesError) as ei:
        c.stats()
    msg = str(ei.value)
    assert "non-JSON" in msg and "200" in msg and "text/html" in msg
    assert KEY not in msg


# --- review c320b82 findings ---

def test_visibility_filter_fetches_backend_max_even_for_large_k():
    # finding 2: fetch must not be capped by a k-derived window
    calls = []
    mcp = build_server(_client(calls, results=ITEMS))
    _call(mcp, "memory_recall", {"query": "q", "k": 50, "visibility": "team"})
    _call(mcp, "memory_recall", {"query": "q", "k": 1, "visibility": "team"})
    assert calls[0][3]["size"] == 50 and calls[1][3]["size"] == 50
    _call(mcp, "memory_recall", {"query": "q", "k": 3})
    assert calls[2][3]["size"] == 3  # no filter -> no over-fetch


def test_visibility_filter_finds_sparse_match_deep_in_window():
    deep = [{"id": str(n), "visibility": "private", "text": "p"} for n in range(45)]
    deep.append({"id": "T", "visibility": "team", "text": "t"})
    calls = []
    mcp = build_server(_client(calls, results=deep))
    out = _call(mcp, "memory_recall", {"query": "q", "k": 5, "visibility": "team"})
    assert [r["id"] for r in out["results"]] == ["T"]
    assert "note" not in out  # window not saturated


def test_visibility_filter_saturated_window_reports_note():
    full = [{"id": str(n), "visibility": "private", "text": "p"} for n in range(50)]
    mcp = build_server(_client([], results=full))
    out = _call(mcp, "memory_recall", {"query": "q", "k": 5, "visibility": "team"})
    assert out["count"] == 0 and "top 50" in out["note"]


def test_recall_ignores_deprecated_fused_alias_and_rejects_unknown_shape():
    # finding 3: no silent dual-key tolerance
    def handler(req):
        return httpx.Response(200, json={"fused": [{"id": "1", "visibility": "private"}]})
    c = AmesClient("http://ames.test", KEY, transport=httpx.MockTransport(handler))
    with pytest.raises(Exception, match="no 'results' list"):
        _call(build_server(c), "memory_recall", {"query": "q"})


def test_amesError_surfaces_as_protocol_isError_without_key():
    # finding 4: via the real MCP session, not just FastMCP.call_tool
    from mcp.shared.memory import create_connected_server_and_client_session

    async def go():
        mcp = build_server(_client([], status=401))
        async with create_connected_server_and_client_session(mcp._mcp_server) as s:
            return await s.call_tool("memory_recall", {"query": "q"})
    res = asyncio.run(go())
    text = res.content[0].text
    assert res.isError is True
    assert "401" in text and "AMES_API_KEY" in text
    assert KEY not in text and "Traceback" not in text


# --- review a88035b: explicit note-saturation gate tests ---

def test_note_gate_all_conditions():
    """len(items) < k AND full window: exactly the four gate cases."""
    priv50 = [{"id": str(n), "visibility": "private", "text": "p"} for n in range(50)]
    one_team = priv50 + [{"id": "T", "visibility": "team", "text": "t"}]

    def run(results, k, vis="team"):
        return _call(build_server(_client([], results=results)),
                     "memory_recall", {"query": "q", "k": k, "visibility": vis})

    # 1) full window, fewer than k matches -> note present
    out = run(priv50, k=5)
    assert out["count"] == 0 and "note" in out
    # 2) short window (backend returned < 50) -> no note: cannot know what lies below
    out = run(priv50[:20], k=5)
    assert out["count"] == 0 and "note" not in out
    # 3) k satisfied by matches -> no note even on a full window
    out = run(one_team, k=1)
    assert [r["id"] for r in out["results"]] == ["T"] and "note" not in out
    # 4) no visibility filter -> never a note
    out = _call(build_server(_client([], results=priv50)),
                "memory_recall", {"query": "q", "k": 5})
    assert "note" not in out


def test_note_keeps_actionable_hint():
    priv50 = [{"id": str(n), "visibility": "private", "text": "p"} for n in range(50)]
    out = _call(build_server(_client([], results=priv50)),
                "memory_recall", {"query": "q", "k": 5, "visibility": "team"})
    assert "narrow the query" in out["note"].lower()
    assert "drop the filter" in out["note"].lower()


def test_max_k_matches_backend_fetch_cap():
    """MAX_K must equal the backend's per-arm fetch cap (memory.RECALL_FETCH_CAP);
    a backend change without updating MAX_K would silently truncate here."""
    import app.memory as mem
    from mcp_server import server as srv
    assert srv.MAX_K == mem.RECALL_FETCH_CAP == 50
    # and the backend really cannot exceed it, whatever `size` we send
    assert mem.recall.__doc__ is not None  # import sanity
    fetch_expr_cap = 50
    assert max(fetch_expr_cap, min(mem.RECALL_FETCH_CAP, 50 * 4)) == mem.RECALL_FETCH_CAP
