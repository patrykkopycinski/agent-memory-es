# AMES MCP server (v1)

Exposes agent-memory-es as an MCP memory tool (Claude Code, Agent Builder, Hermes, any MCP client).
Thin HTTP wrapper over the AMES REST API; adds no ranking/dedup logic and does not touch `backend/app/`.

## Tools
| tool | notes |
|---|---|
| `memory_recall(query, k=8, visibility?, kinds?, as_of?)` | read-only. k clamped 1-50. |
| `memory_reflect(question)` | read-only; backend may call an LLM, slower. |
| `memory_stats()` | read-only; counts/models/pages visible to the key. |
| `memory_retain(text, kind="semantic", visibility="private", occurred_at?)` | WRITE. Only registered when `ALLOW_WRITE=1`; otherwise it does not exist in `tools/list`. |

## Config (env)
- `AMES_API_KEY` (required): the server refuses to start without it. Only sent as `X-API-Key`; never logged or put in errors.
- `AMES_URL` (default `http://localhost:8123`), `AMES_TIMEOUT` (default 30s), `ALLOW_WRITE` (`1/true/yes/on`, default off).

## Visibility and scope
Scope is the key identity: the key decides the private bank (`owner_id`) and which team/common
memories are visible. `visibility` on `memory_recall` only *narrows* results to private/team/common.
The backend recall API has no visibility parameter, so the filter is applied in this server after
over-fetching (min(50, max(4k, 20)) candidates); a narrow filter on a sparse bank can return fewer than k.
`memory_retain` passes `visibility` through (default `private`); team/common are shared with others,
so enable writes deliberately.

## Run
    cd backend && pip install -r mcp_server/requirements.txt
    AMES_API_KEY=... python -m mcp_server                                  # stdio
    AMES_API_KEY=... python -m mcp_server --transport streamable-http --port 8124   # http://127.0.0.1:8124/mcp

The HTTP transport has NO auth of its own and uses the server's single key: keep it on loopback
(default) or behind an authenticating proxy. Prefer stdio, one server per identity.

Claude Code:  `claude mcp add ames -e AMES_API_KEY=... -e AMES_URL=http://localhost:8123 -- python -m mcp_server` (cwd `backend/`)
Hermes (`config.yaml`): `mcp_servers: {ames: {command: python3, args: ["-m","mcp_server"], cwd: .../backend, env: {AMES_API_KEY: ..., ALLOW_WRITE: "0"}}}`

Docker: `docker build -f backend/mcp_server/Dockerfile -t ames-mcp backend/` then
`docker run -i --rm --network host -e AMES_API_KEY ames-mcp` (stdio).

## Tests
    cd backend && python -m pytest mcp_server/tests            # unit (mocked backend)
    AMES_SMOKE_KEY=<scratch-owner key> python -m pytest mcp_server/tests/test_live_smoke.py   # live, recall/stats only
    AMES_API_KEY=... python -m mcp_server.scripts.handshake_demo "query"   # prints a real client transcript

Design: official SDK `FastMCP` (`mcp>=1.2,<2`; 2.x renamed it) for stdio + streamable-http and schemas from type hints.
