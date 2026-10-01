# agent-memory-es

A self-owned, Elasticsearch-native long-term memory service for AI agents: persistent memory across sessions for personal assistants and coding-agent fleets, with shared team knowledge plus strictly private per-user banks.

## Why

Agents forget everything between sessions. This service gives every agent (or human teammate) a memory bank: raw experience, distilled facts, and procedural playbooks — retrievable with hybrid search, isolated per owner, and shareable on explicit promotion.

## Design highlights

- **Three indices**: episodic (raw events), semantic (facts), procedural (playbooks/skills)
- **Recall**: hybrid BM25 + dense (`semantic_text`) fused with RRF; token-budget cut; cross-encoder rerank optional
- **Tenancy**: every doc carries `owner_id` + `visibility ∈ {private, team, common}`; recall filter `owner_id == me OR visibility != private` (DLS when the cluster license allows, app-level API-key→filter middleware otherwise)
- **Consolidation**: background worker dedups and supersedes semantic facts; audit history preserved
- **Surfaces**: native provider API + MCP endpoint (Claude Code, Cursor, Codex, any MCP client)
- **Target cluster**: Elasticsearch 9.6.0-SNAPSHOT (self-hosted docker)

## Layout

```
backend/    FastAPI app: memory ops, consolidation worker, MCP endpoint
docs/       internal design + decision records
scripts/    seed, isolation tests, parity harness
```

## Status

Phase 0 spike — see docs/PHASE0_PLAN.md.

## License

MIT
