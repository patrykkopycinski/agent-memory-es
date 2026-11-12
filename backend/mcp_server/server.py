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
import contextlib
import os
import sys
import time
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP

from .client import AmesClient, AmesError, _otel

Visibility = Literal["private", "team", "common"]
Kind = Literal["episodic", "semantic", "procedural"]

# Client-side ceiling on k and the filtered over-fetch. Verified against the backend:
# memory.recall caps every arm at RECALL_FETCH_CAP = 50, so asking for more than 50 can
# never return more. If the backend cap changes, this must track it.
MAX_K = 50
_TRUE = {"1", "true", "yes", "on"}

# fields of a recalled item worth sending to the model (drops embeddings, rrf internals)
_ITEM_FIELDS = ("id", "kind", "score", "confidence", "visibility", "text", "occurred_at",
                "entities", "doc_group")


@contextlib.contextmanager
def _tool_span(name: str):
    """`mcp.tool_call` span: tool name, latency and result counts. Never the
    arguments or the result text. Untraced (yields a no-op) without telemetry."""
    if _otel is None:
        yield lambda res: res
        return
    t0 = time.perf_counter()
    with _otel.span("mcp.tool_call", {"mcp.tool.name": name,
                                       "mcp.transport": "fastmcp"},
                    otel_kind="server"):
        def record(res):
            _otel.set_attrs({"mcp.latency_ms":
                             round((time.perf_counter() - t0) * 1000.0, 3)})
            _otel.set_attrs(_otel.mcp_result_attrs(res))
            return res
        try:
            yield record
        finally:
            _otel.set_attrs({"mcp.latency_ms":
                             round((time.perf_counter() - t0) * 1000.0, 3)})


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
        with _tool_span("memory_recall") as record:
            return record(_recall(client, query, k, visibility, kinds, as_of))

    @mcp.tool()
    def memory_reflect(question: str) -> dict:
        """Answer a question from memory (multi-round recall + synthesis, cited
        by memory id). Slower than memory_recall and may use an LLM."""
        with _tool_span("memory_reflect") as record:
            return record(client.reflect(question))

    @mcp.tool()
    def memory_stats() -> dict:
        """Counts of memories visible to this key per kind/visibility, recent
        semantic memories, active mental models and knowledge pages."""
        with _tool_span("memory_stats") as record:
            return record(client.stats())

    @mcp.tool()
    def memory_find_related(doc_id: str, kind: str = "semantic",
                            threshold: float = 0.3, k: int = 5) -> dict:
        """Find similar-but-unlinked memories for one memory id (hidden
        connections): hypotheses from the embedding space, never auto-linked.

        doc_id: a memory id from memory_recall/memory_reflect results.
        kind: the memory's kind (episodic/semantic/procedural).
        threshold: cosine floor (0..1). Lower finds more, noisier candidates.
          Default 0.3, lower than the REST default (0.7): this bank's local
          embedding model scores sibling topics 0.30-0.50, so 0.7 surfaces
          nothing for agent exploration.
        k: max candidates (1-20).
        Use memory_links_list(status='proposed') to see filed link proposals
        from the scheduled pass instead.
        """
        with _tool_span("memory_find_related") as record:
            try:
                return record(client.links_find_related(doc_id, kind, threshold, k))
            except AmesError as e:
                if "404" in str(e):
                    raise AmesError(f"no visible memory with id {doc_id!r} "
                                    f"of kind {kind!r}") from None
                raise

    @mcp.tool()
    def memory_links_list(status: Optional[str] = None,
                          edge_source: Optional[str] = None,
                          size: int = 50) -> dict:
        """List stored memory-to-memory edges (links) visible to this key.

        status: active | proposed | rejected (omit for all).
        edge_source: human | near_dup | co_occurrence | proposal (omit for all).
        size: 1-100.
        Proposed edges await an owner decision (memory_links_approve/reject).
        """
        with _tool_span("memory_links_list") as record:
            return record(client.links_list(status, edge_source,
                                            max(1, min(int(size), 100))))

    @mcp.tool()
    def memory_links_structure(kind: str = "semantic") -> dict:
        """Structure snapshot of this key's memory bank: active docs, docs with
        at least one active edge, orphans and orphan rate. High orphan rate =
        many disconnected memories (candidates for linking or consolidation)."""
        with _tool_span("memory_links_structure") as record:
            return record(client.links_structure(kind))

    if allow_write:
        @mcp.tool()
        def memory_retain(text: str, kind: Kind = "semantic",
                          visibility: Visibility = "private",
                          occurred_at: Optional[str] = None) -> dict:
            """Store a memory (WRITE; enabled because ALLOW_WRITE is set).
            kind: episodic (event) / semantic (fact) / procedural (how-to).
            visibility defaults to private; team/common are shared with others."""
            with _tool_span("memory_retain") as record:
                return record(client.retain(kind, text, visibility, occurred_at))

        @mcp.tool()
        def memory_links_link(src: str, dst: str, kind: str = "semantic",
                              reason: Optional[str] = None) -> dict:
            """Link two memories with a human edge (WRITE). Both must be visible
            to this key. Use when a relationship the automatic pass cannot see
            (cause/effect, same incident, same decision) matters for recall."""
            with _tool_span("memory_links_link") as record:
                return record(client.links_link(kind, src, dst, reason))

        @mcp.tool()
        def memory_links_approve(edge_id: str) -> dict:
            """Activate a proposed edge from the hidden-connections pass (WRITE).
            Only the owner of the bank may approve."""
            with _tool_span("memory_links_approve") as record:
                return record(client.links_approve(edge_id))

        @mcp.tool()
        def memory_links_reject(edge_id: str) -> dict:
            """Reject a proposed edge (WRITE): the pair stays unlinked and the
            pass will not re-propose it. Only the owner of the bank may reject."""
            with _tool_span("memory_links_reject") as record:
                return record(client.links_reject(edge_id))

    return mcp


def _recall(client, query, k, visibility, kinds, as_of) -> dict:
    k = max(1, min(int(k), MAX_K))
    # The backend has no visibility param, so filtering happens client-side in this
    # server. With a filter set, fetch MAX_K (this server's ceiling, not a multiple
    # of k) so a sparse visibility is not starved by a k-sized window; then cut to k.
    fetch = k if visibility is None else MAX_K
    resp = client.recall(query, fetch, list(kinds) if kinds else None, as_of)
    items = resp.get("results") if isinstance(resp, dict) else None
    if not isinstance(items, list):
        raise AmesError("AMES recall response has no 'results' list "
                        "(backend/API version mismatch?)")
    scanned = len(items)
    if visibility is not None:
        items = [i for i in items if i.get("visibility") == visibility]
    # Backend order is the answer order: confidence-descending, or the cross-encoder's
    # when hits carry rerank_rank. The visibility filter only drops items, so this sort
    # just restores that order for the cut to k.
    if not any("rerank_rank" in i for i in items):
        items = sorted(items, key=lambda i: (i.get("confidence") is None,
                                             -(i.get("confidence") or 0.0)))
    items = items[:k]
    out = {"query": query, "count": len(items),
           "results": [_slim(i) for i in items],
           "abstained": bool(resp.get("abstained"))}
    if visibility is not None and len(items) < k and scanned >= MAX_K:
        out["note"] = (f"best effort: visibility filter applied to the top {scanned} "
                       f"backend hits; more '{visibility}' matches may exist below that. "
                       "Narrow the query, or drop the filter and filter client-side.")
    if resp.get("mental_model"):
        out["mental_model"] = resp["mental_model"]
    return out


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
    if _otel is not None:
        _otel.configure_for_cli("ames-mcp")
    mcp.run(transport=a.transport)
