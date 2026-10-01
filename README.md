# agent-memory-es

Self-owned, ES-native agent memory — a Hindsight replacement for Hermes and a shared memory layer for a coding farm. Original implementation; architecture informed by Elastic's public agent-memory research (see ATTRIBUTION).

## Design (see docs/DECISION_RECORD.md)

- **Three indices**: episodic (raw events), semantic (facts), procedural (playbooks/skills)
- **Recall**: hybrid BM25 + dense (`semantic_text`) fused with RRF; token-budget cut; per-tenant visibility filter
- **Tenancy**: every doc carries `owner_id` + `visibility ∈ {private, team, common}`; recall filter `owner_id == me OR visibility != private` (DLS when the cluster license allows, app-level API-key→filter middleware otherwise)
- **Consolidation**: background worker dedups and supersedes semantic facts, audit history preserved
- **Surfaces**: Hermes memory-provider (native) + MCP endpoint (coding farm, teammates)
- **Target cluster**: Elasticsearch 9.6.0-SNAPSHOT (self-hosted, m1max docker) — verified to exist in the elastic/elasticsearch snapshots

## Layout

```
backend/    FastAPI app: memory ops, consolidation worker, MCP endpoint, Hermes provider adapter
docs/       decision record, design notes
scripts/    seed, leak tests, parity harness
```

## Attribution

Architecture informed by Elastic Search-Labs' public agent-memory blog post
(https://www.elastic.co/search-labs/blog/agent-memory-elasticsearch) and its
MIT-licensed reference demo noamschwartz/atlas-memory-demo. No code copied;
this is an independent implementation under our own repo and license (MIT).
