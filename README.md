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
- **Mental models** — curated summaries for frequent questions, matched by BM25 on question pattern, surfaced as a priority tier in recall and reflect. `POST /memory/models`.
- **Write-time fact extraction** — every episodic `retain` queues an async job; a worker (`python -m app.worker --facts`) extracts up to 12 dated, source-linked facts into `semantic`. Contradicted facts get `valid_to`/`superseded_by` (kept for audit, excluded from recall after expiry). Durable lease-based queue with retries; `GET /memory/facts/drain`, `POST /memory/facts/backfill` (+ `--backfill` worker) for existing memory; `"extract_facts": false` opts out on bulk imports.
- **Label filters** — caller tags + optional LLM label extraction on retain; recall `filter` (`all`/`any`/`none`/`narrow_any`) abstains instead of returning off-topic memories.

## Benchmarks

Measured against [Hindsight](https://github.com/vectorize-io/hindsight) with the same harness, answer model and judge for both. Full method, per-type tables, development history and follow-ups: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

| Benchmark | AMES | Hindsight | Verdict |
|---|---|---|---|
| **LongMemEval-S holdout 100** (current release) | **84/100** | **84/100** | **Parity** (McNemar p=1.00) |
| LongMemEval-S anchor 100 (current release) | 83/100 | 76/100 | Parity, AMES ahead (p=0.21) |
| PersonaMem 32k, 589 Qs (previous release) | 352 (59.8%) | 372 (63.2%) | Parity, Hindsight ahead (p=0.10) |
| PrecisionMemBench, 77 cases (previous release) | 58/77 | 66/77 | Hindsight ahead (precision) |

AMES sends less context than Hindsight on every holdout question (median 17.3k vs 20.0k chars; recall p95 53 ms). Wins come from questions about what the *assistant* said (1.00 vs 0.40); Hindsight leads on knowledge-update and multi-session. Next: latest-value-only recall for knowledge-update, date arithmetic in code for temporal questions — see [follow-ups](docs/BENCHMARKS.md#follow-ups).

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
backend/    FastAPI app: memory ops, consolidation + fact-extraction workers, MCP endpoint
backend/hermes_plugin/    reference copy of the Hermes provider plugin
docs/       architecture, decision records, diagrams
scripts/    doctor, importers, MCP bridge, seed, isolation tests
```

## Status

Deployed and in daily use (self-hosted) as the Hermes memory provider, replacing Hindsight: MCP farm workers onboarded, consolidation and fact-extraction workers running. Benchmark parity with Hindsight on the LongMemEval-S holdout ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)). API surface is small and stable; consolidation and reflect are evolving. History: docs/PHASE0_PLAN.md, docs/AUDIT.md.

## License

MIT
