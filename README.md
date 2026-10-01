# agent-memory-es

A self-owned, Elasticsearch-native long-term memory service for AI agents: persistent memory across sessions for personal assistants and coding-agent fleets, with shared team knowledge plus strictly private per-user banks.

```
Hermes Agent ─┐                                   ┌─ episodic  (raw events)
MCP clients ──┼─ FastAPI :8123 ── Elasticsearch ──┼─ semantic  (distilled facts)
HTTP / curl ──┘   (auth · ops · worker)           └─ procedural (playbooks)
```

## Why

Agents forget everything between sessions. This service gives every agent (or human teammate) a memory bank: raw experience, distilled facts, and procedural playbooks — retrievable with hybrid search, isolated per owner, and shareable only on explicit promotion.

## Demo tour

- **[Architecture overview](docs/ARCHITECTURE.md)** — every component, grounded in source
- **[System topology diagram](docs/diagrams/architecture.svg)** · **[recall pipeline diagram](docs/diagrams/recall-pipeline.svg)**
- **[Interactive explainer](docs/explainer.html)** — click through retain → recall → reflect
- **[Landing page](docs/index.html)** — the concept in one page

## Design highlights

- **Three indices**: episodic (raw events), semantic (facts), procedural (playbooks/skills)
- **Recall**: hybrid BM25 + dense (`dense_vector` kNN) fused with RRF; token-budget cut
- **Tenancy**: every doc carries `owner_id` + `visibility ∈ {private, team, common}`; the recall filter `owner_id == me OR visibility != private` is applied server-side — owner identity comes from the API key, never from a request argument
- **Promotion guard**: deterministic sensitive-marker regex blocks private→shared promotion; user rejection tombstones block re-learning forgotten values
- **Consolidation**: background worker dedups and supersedes semantic facts (propose-then-apply); audit history preserved
- **Surfaces**: native Hermes memory-provider plugin + MCP endpoint (Claude Code, Cursor, Codex, any MCP client) + plain HTTP
- **Graceful degrade**: no embeddings backend → BM25-only recall
- **Target cluster**: Elasticsearch 9.6.0-SNAPSHOT (self-hosted docker)

## Quick start

```bash
cd backend
export AMES_ADMIN_TOKEN=change-me
docker compose up -d          # ames-backend :8123, ames-worker, Elasticsearch

# mint a key (owner identity is bound to it)
curl -s -X POST "localhost:8123/admin/keys?owner_id=alice" -H "X-Admin-Token: $AMES_ADMIN_TOKEN"

# remember something, then recall it
curl -s -X POST localhost:8123/memory/retain -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"kind":"semantic","text":"Deploys go through OmniRoute catchup branches","visibility":"private"}'
curl -s -X POST localhost:8123/memory/recall -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"query":"how do deploys work"}'
```

Hermes users: install the [standalone plugin](https://github.com/patrykkopycinski/hermes-plugin-agent-memory-es) — it turns the API into `agent_memory_recall` / `agent_memory_retain` / `agent_memory_reflect` agent tools.

## Layout

```
backend/    FastAPI app: memory ops, consolidation worker, MCP endpoint
backend/hermes_plugin/    reference copy of the Hermes provider plugin
docs/       architecture, decision records, diagrams
scripts/    phase-0 spike, seed, isolation tests
```

## Status

Phase 0 spike — see docs/PHASE0_PLAN.md. API surface is small and stable; consolidation and reflect are evolving.

## License

MIT
