"""FastMCP server for agent-memory-es.

Env:
  AMES_API_KEY   required. Identity == scope: the key decides which private bank
                 and which team/common memories this server can see.
  AMES_URL       backend base URL (default http://localhost:8123)
  ALLOW_WRITE    "1"/"true"/"yes"/"on" registers memory_retain (default off)
  AMES_TIMEOUT   HTTP timeout seconds (default 30)

Transports: stdio (default) or streamable-http (--transport streamable-http).
"""
import argparse
import os
import sys
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP

from .client import AmesClient, AmesError

Visibility = Literal["private", "team", "common"]
Kind = Literal["episodic", "semantic", "procedural"]

MAX_K = 50
_TRUE = {"1", "true", "yes", "on"}

# fields of a recalled item worth sending to the model (drops embeddings, rrf internals)
_ITEM_FIELDS = ("id", "kind", "score", "visibility", "text", "occurred_at",
                "entities", "doc_group")


def write_enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("ALLOW_WRITE", "")).strip().lower() in _TRUE


def _slim(item: dict) -> dict:
    return {k: item[k] for k in _ITEM_FIELDS if k in item}


def build_server(client: AmesClient, allow_write: bool = False,
                 **fastmcp_kwargs) -> FastMCP:
    mcp = FastMCP("agent-memory-es", **fastmcp_kwargs)

    @mcp.tool()
    def memory_recall(query: str, k: int = 8, visibility: Optional[Visibility] = None,
                      kinds: Optional[list[Kind]] = None,
                      as_of: Optional[str] = None) -> dict:
        """Search long-term memory (hybrid BM25 + vector, time-aware).

        query: natural-language question or keywords.
        k: max results (1-50, default 8).
        visibility: only return memories with this visibility (private/team/common).
          Scope is determined by the API key identity, not by this filter: it can
          only narrow what the key may already see.
        kinds: restrict to episodic / semantic / procedural.
        as_of: ISO-8601 instant the question is asked; resolves 'recent'/'last week'.
        """
        k = max(1, min(int(k), MAX_K))
        # backend has no visibility param: filter client-side. With a filter, always
        # fetch the backend maximum (not a multiple of k) so a sparse visibility is not
        # starved by the k-sized window; then cut to k.
        fetch = k if visibility is None else MAX_K
        resp = client.recall(query, fetch, list(kinds) if kinds else None, as_of)
        items = resp.get("results") if isinstance(resp, dict) else None
        if not isinstance(items, list):
            raise AmesError("AMES recall response has no 'results' list "
                            "(backend/API version mismatch?)")
        scanned = len(items)
        if visibility is not None:
            items = [i for i in items if i.get("visibility") == visibility]
        items = items[:k]
        out = {"query": query, "count": len(items),
               "results": [_slim(i) for i in items],
               "abstained": bool(resp.get("abstained"))}
        if visibility is not None and len(items) < k and scanned >= MAX_K:
            out["note"] = (f"visibility filter applied to the top {scanned} backend hits "
                           f"only; more '{visibility}' memories may exist below that. "
                           "Narrow the query or drop the filter.")
        if resp.get("mental_model"):
            out["mental_model"] = resp["mental_model"]
        return out

    @mcp.tool()
    def memory_reflect(question: str) -> dict:
        """Answer a question from memory (multi-round recall + synthesis, cited
        by memory id). Slower than memory_recall and may use an LLM."""
        return client.reflect(question)

    @mcp.tool()
    def memory_stats() -> dict:
        """Counts of memories visible to this key per kind/visibility, recent
        semantic memories, active mental models and knowledge pages."""
        return client.stats()

    if allow_write:
        @mcp.tool()
        def memory_retain(text: str, kind: Kind = "semantic",
                          visibility: Visibility = "private",
                          occurred_at: Optional[str] = None) -> dict:
            """Store a memory (WRITE; enabled because ALLOW_WRITE is set).
            kind: episodic (event) / semantic (fact) / procedural (how-to).
            visibility defaults to private; team/common are shared with others."""
            return client.retain(kind, text, visibility, occurred_at)

    return mcp


def make_server_from_env(env=None, **fastmcp_kwargs) -> FastMCP:
    env = os.environ if env is None else env
    allow = write_enabled(env)
    client = AmesClient(env.get("AMES_URL", "http://localhost:8123"),
                        env.get("AMES_API_KEY", ""),
                        float(env.get("AMES_TIMEOUT", "30")))
    return build_server(client, allow, **fastmcp_kwargs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="mcp_server")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8124)
    a = p.parse_args(argv)
    try:
        mcp = make_server_from_env(host=a.host, port=a.port)
    except AmesError as e:
        print(f"ames-mcp: {e}", file=sys.stderr)  # stdout is the MCP channel
        sys.exit(2)
    mcp.run(transport=a.transport)
