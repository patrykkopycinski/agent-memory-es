"""Live smoke over a REAL MCP stdio handshake against a running ames-backend.
Skipped unless AMES_SMOKE_KEY is set. Use a scratch owner key, never a live bank.
Recall/stats only; no writes. AMES_SMOKE_URL defaults to http://localhost:8123."""
import asyncio
import json
import os
import pathlib
import sys

import pytest

mcp_types = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

KEY = os.environ.get("AMES_SMOKE_KEY")
pytestmark = pytest.mark.skipif(not KEY, reason="AMES_SMOKE_KEY not set")
BACKEND_DIR = str(pathlib.Path(__file__).resolve().parents[2])


async def _session_run(fn, extra_env=None):
    env = {"PATH": os.environ.get("PATH", ""), "AMES_API_KEY": KEY,
           "AMES_URL": os.environ.get("AMES_SMOKE_URL", "http://localhost:8123"),
           **(extra_env or {})}
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"],
                                   env=env, cwd=BACKEND_DIR)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            return await fn(s, init)


def test_handshake_list_recall_stats():
    async def go(s, init):
        tools = {t.name for t in (await s.list_tools()).tools}
        rec = await s.call_tool("memory_recall", {"query": "mcp smoke", "k": 3})
        st = await s.call_tool("memory_stats", {})
        return init, tools, rec, st
    init, tools, rec, st = asyncio.run(_session_run(go))
    assert init.serverInfo.name == "agent-memory-es"
    assert tools == {"memory_recall", "memory_reflect", "memory_stats"}
    assert not rec.isError and not st.isError
    body = json.loads(rec.content[0].text)
    assert body["query"] == "mcp smoke" and isinstance(body["results"], list)
    assert "counts" in json.loads(st.content[0].text)
    assert KEY not in rec.content[0].text + st.content[0].text


def test_retain_absent_without_allow_write():
    async def go(s, init):
        return {t.name for t in (await s.list_tools()).tools}
    assert "memory_retain" not in asyncio.run(_session_run(go))
