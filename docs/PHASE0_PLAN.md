# Phase 0 spike plan (owner: Patryk)

Goal: prove the core mechanics on ES 9.6.0-SNAPSHOT before writing the full backend.

## Steps

1. Boot ES 9.6.0-SNAPSHOT (docker, m1max or local) — snapshot image from docker.elastic.co
2. Index templates: `am_episodic`, `am_semantic`, `am_procedural` — semantic_text field, owner_id, visibility, timestamps, superseded_by
3. Recall: hybrid BM25+dense RRF + visibility filter; token-budget cut app-side
4. Leak test: owner A retains private doc; owner B recall must not return it (both DLS and middleware paths)
5. Consolidation spike: supersede a stale fact, verify audit chain kept
6. MCP endpoint skeleton: retain/recall/reflect tools

## Exit gates

- cross-owner leak test: 0 hits for B on A's private docs
- retain→recall round-trip via script
- supersession preserves history
