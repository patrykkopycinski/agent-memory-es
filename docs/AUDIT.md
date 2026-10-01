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
