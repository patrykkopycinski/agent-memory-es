"""Standalone MCP (stdio JSON-RPC) server exposing agent-memory-es tools.

Works with any MCP client (Claude Code, Cursor, ...):

  ames-mcp --api-key KEY --url http://localhost:8123

Tools: ames_recall, ames_retain, ames_reflect. The HTTP API must be
X-API-Key authenticated; keys are minted via POST /admin/keys.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TOOLS = [
    {
        "name": "ames_recall",
        "description": "Search durable agent memory (episodic/semantic/procedural, "
                       "private/team/common visibility). Call before re-reading a "
                       "codebase or re-deriving known context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "natural-language search"},
                "kinds": {"type": "array", "items": {"type": "string",
                          "enum": ["episodic", "semantic", "procedural"]}},
                "size": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "ames_retain",
        "description": "Store a durable memory worth reusing across sessions "
                       "(decisions, procedures, hard-won facts). Not for transient state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                "text": {"type": "string"},
                "visibility": {"type": "string", "enum": ["private", "team", "common"],
                               "default": "private"},
            },
            "required": ["kind", "text"],
        },
    },
    {
        "name": "ames_reflect",
        "description": "Answer a question from memory with citations.",
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]


def http(url: str, key: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def dispatch(url: str, key: str, name: str, args: dict) -> dict:
    if name == "ames_recall":
        return http(url, key, "/memory/recall",
                    {"query": args["query"], "kinds": args.get("kinds"),
                     "size": args.get("size", 8)})
    if name == "ames_retain":
        return http(url, key, "/memory/retain",
                    {"kind": args["kind"], "text": args["text"],
                     "visibility": args.get("visibility", "private")})
    if name == "ames_reflect":
        return http(url, key, "/memory/reflect", {"question": args["question"]})
    raise ValueError(f"unknown tool {name}")


def rpc(url, key):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, method = msg.get("id"), msg.get("method")
        if method == "initialize":
            out = {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agent-memory-es", "version": "0.1.0"}}}
        elif method == "notifications/initialized" or method == "initialized":
            continue
        elif method == "tools/list":
            out = {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            try:
                res = dispatch(url, key, msg["params"]["name"],
                               msg["params"].get("arguments", {}))
                out = {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": json.dumps(res)[:60000]}]}}
            except Exception as e:
                out = {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"error: {e}"}],
                    "isError": True}}
        elif method == "ping":
            out = {"jsonrpc": "2.0", "id": mid, "result": {}}
        elif mid is not None:
            out = {"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32601, "message": f"unknown method {method}"}}
        else:
            continue
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(prog="ames-mcp")
    p.add_argument("--url", default=os.environ.get("AMES_SERVICE_URL", "http://localhost:8123"))
    p.add_argument("--api-key", default=os.environ.get("AMES_SERVICE_KEY", ""))
    a = p.parse_args()
    if not a.api_key:
        print("ames-mcp: AMES_SERVICE_KEY or --api-key required", file=sys.stderr)
        sys.exit(2)
    rpc(a.url, a.api_key)


if __name__ == "__main__":
    main()
