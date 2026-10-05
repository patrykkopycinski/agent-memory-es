"""Print a real MCP client transcript (stdio): initialize, tools/list, recall, stats.
Usage: AMES_API_KEY=... python -m mcp_server.scripts.handshake_demo [query]  (run from backend/)"""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(query: str) -> None:
    env = {"PATH": os.environ["PATH"], "AMES_API_KEY": os.environ["AMES_API_KEY"],
           "AMES_URL": os.environ.get("AMES_URL", "http://localhost:8123")}
    p = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"], env=env)
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            print("initialize ->", init.serverInfo.name, init.serverInfo.version,
                  "protocol", init.protocolVersion)
            for t in (await s.list_tools()).tools:
                print("tool:", t.name, "| args:", list(t.inputSchema.get("properties", {})))
            for name, args in [("memory_recall", {"query": query, "k": 3}),
                               ("memory_stats", {})]:
                res = await s.call_tool(name, args)
                print(f"call {name} {json.dumps(args)} isError={res.isError}")
                print(res.content[0].text[:800])

asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "hello"))
