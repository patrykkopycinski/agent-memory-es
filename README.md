# agent-memory-es

A self-owned, Elasticsearch-native long-term memory service for AI agents: persistent memory across sessions for personal assistants and coding-agent fleets, with shared team knowledge plus strictly private per-user banks.

```
Hermes Agent ─┐                                   ┌─ episodic  (raw events)
MCP clients ──┼─ FastAPI :8123 ── Elasticsearch ──┼─ semantic  (distilled facts)
HTTP / curl ──┘   (auth · ops · worker)           └─ procedural (playbooks)
```

## Quickstart (60 seconds)

```bash
# 1. everything up (ES + API + worker, named volumes):
cd backend && docker compose -f docker-compose.quickstart.yml up -d
# 1b. host-side tools (importer, doctor) talk to the compose ES — NEVER to
#     :9268, which may be an ssh tunnel to someone's PROD cluster:
export AMES_ES_URL=http://localhost:19200
# 2. mint a key:
curl -X POST -H 'X-Admin-Token: dev-admin' \
  'http://localhost:8123/admin/keys?owner_id=you'
# 3. preflight:
python ../scripts/ames_doctor.py
# 4. import your existing memory (governance stores / Hindsight / CLAUDE.md):
python ../scripts/import_memory.py ~/my-governance-store --dry-run
python ../scripts/import_memory.py ~/my-governance-store
# 5. recall from any MCP client (Claude Code, Cursor):
python ../scripts/ames_mcp_server.py   # reads AMES_SERVICE_URL / AMES_SERVICE_KEY
```

Preflight trouble → `ames_doctor.py` prints the exact fix for each failure.
No Docker? Any ES 8.x works — set `AMES_ES_URL`. Importers & MCP bridge:
**[docs/IMPORTERS.md](docs/IMPORTERS.md)** · usage patterns:
**[docs/RECIPES.md](docs/RECIPES.md)** (recall-before-read, session-end
retention hooks, what deserves the bank).

## Why

Agents forget everything between sessions. This service gives every agent (or human teammate) a memory bank: raw experience, distilled facts, and procedural playbooks — retrievable with hybrid search, isolated per owner, and shareable only on explicit promotion.

## Architecture

```
Hermes Agent ─┐                                   ┌─ episodic  (raw events)
MCP clients ──┼─ FastAPI :8123 ── Elasticsearch ──┼─ semantic  (distilled facts)
HTTP / curl ──┘   (auth · ops · worker)           └─ procedural (playbooks)
```

API key → owner identity, server-side visibility filter, hybrid BM25 + kNN recall fused with RRF, a guarded private→shared promotion path, and a propose-then-apply consolidation worker. Full detail with per-file source references: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — diagrams: [topology](docs/diagrams/architecture.svg) · [recall pipeline](docs/diagrams/recall-pipeline.svg) · interactive: [explainer](docs/explainer.html).

## Capabilities (v0.2)

Beyond the hybrid core:

- **Graph arm** — entity two-hop expansion in recall: facts sharing entities with your query's entities surface even with zero lexical overlap. Fused at low RRF weight (0.01) so hop-2 evidence never outvotes lexical+kNN agreement (measured: MRR 0.906→0.928 with hop-2 intact).
- **Knowledge pages** — `am_pages`: per-topic living documents regenerated from active semantic facts, evidence-linked (`source_ids`), never self-citing. `POST /memory/pages/refresh`, `GET /memory/pages[/{scope}]`.
- **Temporal arm** — time expressions in queries ("last 3 months", "in 2024") parse into a window, filled spread across equal buckets so results aren't all from one end.
- **Reranker** — ES-native `_inference` rerank endpoint; honest fallback (`reranked: false`) when model/license unavailable — never faked.
- **Reflect multi-round** — bounded LLM query rewrites extend evidence before synthesis; mental-model tier consulted first.
- **Mental models** — curated summaries for frequent questions, matched by BM25 on question pattern, surfaced as a priority tier in recall and reflect. `POST /memory/models`. Worker draft proposals are off by default; set `AMES_MODEL_DRAFTS=1` (or `true`) to enable. Drafts are idempotent per (owner, pattern) — one draft, refreshed in place, never auto-promoted. `AMES_DRAFT_MAX_ENTITY_DF` (default `0.3`) caps cluster entities at that fraction of the owner’s active facts — entities appearing on more facts (extraction artifacts like `memory`/`facts`/`conventions`) are skipped as cluster keys.

## Landing page

**https://patrykkopycinski.github.io/agent-memory-es/** — concept, architecture, quick start, interactive archify diagrams.

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
scripts/    doctor, importers, MCP bridge, seed, isolation tests
```

## Mental-model drafts

Drafts are gated by two knobs: a cluster-key entity needs at least
`min_facts` (default 3) active semantic facts, and it must stay under the
max document frequency `AMES_DRAFT_MAX_ENTITY_DF` (default 0.3) — i.e. the
entity may appear on at most 30% of the owner's active semantic facts.

Because of the DF cutoff, 3 cluster facts need at least 7 unrelated filler
facts to stay under the 30% bar, so **an owner needs >= 10 active semantic
facts before any model draft can appear** (3 cluster facts + 7 fillers);
the first draft shows up at the next consolidation pass after that.

## Status

Deployed and in daily use (self-hosted): Hermes memory provider swapped in, MCP farm workers onboarded, consolidation worker running. API surface is small and stable; consolidation and reflect are evolving. History: docs/PHASE0_PLAN.md, docs/AUDIT.md.

## License

MIT

