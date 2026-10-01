# Decision Audit Trail

Append-only. Every autonomous decision logged with rationale, so any entry can be challenged retroactively.

## Format
`[DATE] AREA — decision :: rationale :: alternative rejected`

## 2026-10-01

- [2026-10-01] PROCUREMENT — Build on local Mac (this host), ES on m1max docker (`am-es-spike` :9268, 9.6.0-SNAPSHOT) :: cloud-vm-dispatch PAT lacks access to the new private repo; user directed "build complete solution on your own" so blocking on PAT plumbing serves nothing; ES execution stays remote per standing doctrine, code authoring is local :: waiting for PAT repo-access update

- [2026-10-01] SCOPE — Autonomous through Phase 1-3, no per-phase approvals :: user explicit instruction "I don't want to approve each of the phases manually… build complete solution on your own", decisions tracked here :: per-phase approval gates

- [2026-10-01] BASE — Independent implementation, atlas consulted as reference only (no code copied) :: user: "I want to make sure it's my own repo" :: forking/copying MIT code wholesale

- [2026-10-01] PRIVACY — Public README describes product on its own terms; internal doc keeps the competitive framing :: user: "fine to mention [the successor framing] internally, but don't make it so obvious publicly" :: public attribution section naming the incumbent
- [2026-10-01] WORKER — Consolidation worker iterates owners derived from API-key store (`worker.known_owners`), 600s default interval, errors logged not fatal :: owners are exactly key-holders; no separate registry needed yet :: separate owner registry table
- [2026-10-01] HERMES — In-process provider adapter (`hermes_provider.py`) calling memory ops directly, no HTTP hop :: latency on Hermes hot path; HTTP adds nothing when co-located :: always-HTTP adapter
- [2026-10-01] TESTS — Session-wide API-key store via conftest; auth keys file resolved per-call (env at call time, not import time) :: pytest imports all modules pre-run, per-module key files cross-contaminated → 401s; import-time env capture caused the earlier "leak" misdiagnosis :: per-module AMES_API_KEYS_FILE
- [2026-10-01] OPS — Recreated indices after spike-script mapping drift (owner_id was text in old indices from prototype) :: keyword term-filter correctness is the isolation contract :: leave old indices, filter in code
- [2026-10-01] EMBEDDINGS — External embeddings via OmniRoute (openrouter/google/gemini-embedding-2, 3072-dim dense_vector) instead of ES inference API :: snapshot license is basic; _inference + semantic_text 403 "non-compliant"; dense_vector+kNN works on basic/trial. Started 30-day enterprise TRIAL on the spike cluster for RRF retriever (also gated) :: wait for enterprise license / rebuild RRF fusion app-side
- [2026-10-01] LICENSE — 30-day trial activated on am-es-spike :: unblocks RRF + inference for dev; prod decision needed before expiry (app-side RRF fallback exists in recall() BM25-only branch) :: stay basic, hand-fuse
- [2026-10-01] API SHAPE — knn retriever inside rrf requires "k" param (9.6.0-SNAPSHOT enforces); rank_window_size required :: discovered via 400 "Required [k]" against live cluster
- [2026-10-01] RESILIENCE — Embedding failures degrade to BM25-only recall (try/except pass in retain+recall) :: memory must never be unavailable because the embedding gateway is down :: hard-fail on embed errors
- [2026-10-01] EMBEDDINGS-2 — ES-native inference first (.multilingual-e5-small-elasticsearch, 384d, cluster-side, no external gateway), OmniRoute fallback behind AMES_EMBED_BACKEND :: user directive: leverage ES-native over OmniRoute; e5-small ships with cluster, works under basic+trial; 3x faster suite (5.4s vs 15s) :: OmniRoute-only
- [2026-10-01] JINA — parked until recall eval harness exists (atlas adoption #5); if e5-small saturates eval set, Jina adds only an external dependency; switching = index rebuild (dims differ) :: switch now on vibes
- [2026-10-01] TOMBSTONE (atlas finding #1, 64/704 systems) — value-keyed rejection in am_rejected index, consulted before every semantic write; 80% term-overlap match blocks reworded re-assertion; per-owner :: hides-by-row correction cannot stop a writer recreating the value :: record-keyed supersession only
- [2026-10-01] TOMBSTONE-IMPL — salient-term overlap (stopword-stripped tokens), not exact hash; strict mapping am_rejected{value_key,terms,owner_id,reason,source_id,rejected_at}
- [2026-10-01] EVAL — Deterministic recall harness (tests/eval_recall.py): 30 facts / 3 owners / paraphrase queries, Recall@k+MRR by doc id, cross-owner private hit = hard fail. ES-native e5-small: Recall@5=1.000 MRR=0.907. OmniRoute gemini-embedding-2: Recall@5=1.000 MRR=0.901 :: equal quality on this corpus; e5-small wins on latency + no external dep + license-free; keep ES-native, Jina parked (corpus not saturating — nothing to measure a better embedder on) :: switch to Jina/gemini now
- [2026-10-01] EVAL-DIM — dims are backend-pinned (384 vs 3072); eval runs must rebuild indices when switching; store.py default stays 384 (ES-native default)
- [2026-10-01] CONSOLIDATE-2 — dry_run=True default (propose-only), apply explicit; keep_both disposition (coexistence ≠ contradiction) at 0.75–0.85 similarity band; _supersede owner-checks both ends (atlas finding vs Atlas's id-only supersede update) :: agentmemory-style silent supersession on a threshold is unreviewable; Memora dry_run + Memanto keep_both adopted
- [2026-10-01] OPS-TRAP — eval dim-switch left a 3072d index vs 384d store default → 400s; index rebuild is part of any backend switch, encoded in eval flow now
- [2026-10-01] EVAL-IDEMPOTENT — eval_recall wipes its own ame_eval_* corpus slice per run (delete_by_query on owner prefix) :: re-runs stacked duplicate golds, MRR collapsed 0.907→0.436 on stale corpus; deterministic eval must own its data :: append-only corpus
- [2026-10-01] WORKER-DRYRUN — worker test updated for dry_run default: run_once() proposes, apply via consolidate(dry_run=False) :: old test asserted mutation on the default path, broke after hardening
- [2026-10-01] REFLECT — OmniRoute chat completion (auto/best-chat, temp 0), evidence-only synthesis, id citations clipped to retrieved set, INSUFFICIENT_EVIDENCE abstention on empty recall :: atlas finding: retrieval abstention beats bending answers with irrelevant memory; citations restricted to retrieved ids per Hindsight reflect contract
- [2026-10-01] REFLECT-TEST — out-of-domain question (capital of France) must abstain; PASS
- [2026-10-01] HERMES-PLUGIN — new MemoryProvider `agent_memory_es` (backend/hermes_plugin/), HTTP to FastAPI service, per-profile API key, sync_turn stores user content only (evidence tier, anti-self-capture), on_memory_write mirrors builtin writes to semantic tier, 3 tools (recall/retain/reflect) :: Hermes plugin contract (MemoryProvider ABC) is the native integration; atlas anti-self-capture: assistant prose never retained. Verified live: retain/recall/prefetch/RecallStatus round-trip against running service + ES 9.6.0-SNAPSHOT :: in-process adapter only
- [2026-10-01] PLUGIN-STATUS — plugin authored + canary-verified in repo, NOT yet installed into a Hermes profile (install = copy to ~/.hermes/profiles/<p>/plugins/ + config provider swap); install/canary on one profile is the next gate before Hindsight retirement
