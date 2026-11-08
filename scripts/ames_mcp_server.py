"""Standalone MCP (stdio JSON-RPC) server exposing agent-memory-es tools.

Works with any MCP client (Claude Code, Cursor, ...):

  ames-mcp --api-key KEY --url http://localhost:8123

Tools: ames_recall, ames_retain, ames_reflect, memory_proposals (read-only). Approving or
rejecting a decision is deliberately not a tool: it is a human act (see scripts/ames_approve.py).
The HTTP API must be
X-API-Key authenticated; keys are minted via POST /admin/keys.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Tracing reuses the backend's telemetry module (same gate, exporter and privacy
# scrubber) when it is importable, i.e. this script runs from a repo checkout or
# the backend package is on PYTHONPATH. Otherwise the server runs untraced.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "backend"))
    from app import telemetry as _otel
except Exception:  # pragma: no cover - minimal install
    _otel = None

MAX_RECALL_SIZE = 50   # keep equal to backend app.memory.MAX_RECALL_SIZE (a test pins it)

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
                "size": {"type": "integer", "default": 8, "minimum": 1,
                         "maximum": MAX_RECALL_SIZE},
                "include_inactive_decisions": {"type": "boolean", "default": False,
                                               "description": "also return rejected, expired "
                                                              "and superseded decisions"},
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
        "name": "memory_proposals",
        "description": "List open team decision proposals (read-only): text, author, who approved, "
                       "approvals still needed. Approval is done by a human, not by an agent.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 50, "minimum": 1,
                                     "maximum": 100}},
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


def _headers(base: dict) -> dict:
    """Add W3C traceparent for the active MCP span so the backend trace joins it."""
    return _otel.inject_headers(base) if _otel is not None else base


def http(url: str, key: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
        headers=_headers({"Content-Type": "application/json", "X-API-Key": key}))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def http_get(url: str, key: str, path: str) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, method="GET",
                                 headers=_headers({"X-API-Key": key}))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


PROPOSED_LABEL = "PROPOSED, not in force"


def label_proposals(res: dict) -> dict:
    """A proposed decision is not in force: say so in the hit text itself, so a reader that only
    looks at `text` cannot mistake it for a standing decision."""
    seen = set()

    def mark(hit):
        if isinstance(hit, dict) and hit.get("decision_status") == "proposed" and id(hit) not in seen:
            seen.add(id(hit))
            hit["label"] = PROPOSED_LABEL
            hit["text"] = f"[{PROPOSED_LABEL}] {hit.get('text', '')}"

    for key in ("results", "evidence", "fused"):
        for hit in res.get(key) or []:
            mark(hit)
    for hits in (res.get("by_kind") or {}).values():
        for hit in hits:
            mark(hit)
    return res


def dispatch(url: str, key: str, name: str, args: dict) -> dict:
    """Run one tool call inside an `mcp.tool_call` span (tool name, latency, result
    counts; never arguments or result text)."""
    if _otel is None:
        return _dispatch(url, key, name, args)
    known = name in {t["name"] for t in TOOLS}
    t0 = time.perf_counter()
    with _otel.span("mcp.tool_call", {"mcp.tool.name": name if known else "unknown",
                                       "mcp.transport": "stdio"},
                    otel_kind="server"):
        try:
            res = _dispatch(url, key, name, args)
        finally:
            _otel.set_attrs({"mcp.latency_ms": round((time.perf_counter() - t0) * 1000.0, 3)})
        _otel.set_attrs(_otel.mcp_result_attrs(res))
        return res


def _dispatch(url: str, key: str, name: str, args: dict) -> dict:
    if name == "ames_recall":
        body = {"query": args["query"], "kinds": args.get("kinds"),
                "size": min(max(int(args.get("size", 8)), 1), MAX_RECALL_SIZE)}
        if args.get("include_inactive_decisions") is True:
            body["include_inactive_decisions"] = True
        return label_proposals(http(url, key, "/memory/recall", body))
    if name == "memory_proposals":
        limit = min(max(int(args.get("limit", 50)), 1), 100)
        out = http_get(url, key, f"/governance/proposals?limit={limit}")
        if out.get("override_conflicts"):
            out["warning"] = ("override conflict: an active override could not supersede its target; "
                              "both decisions are in force until an operator resolves it")
        return out
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
    if _otel is not None:
        _otel.configure_for_cli("ames-mcp")
    rpc(a.url, a.api_key)


if __name__ == "__main__":
    main()
