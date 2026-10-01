# Decision Audit Trail

Append-only. Every autonomous decision logged with rationale, so any entry can be challenged retroactively.

## Format
`[DATE] AREA — decision :: rationale :: alternative rejected`

## 2026-10-01

- [2026-10-01] PROCUREMENT — Build on local Mac (this host), ES on m1max docker (`am-es-spike` :9268, 9.6.0-SNAPSHOT) :: cloud-vm-dispatch PAT lacks access to the new private repo; user directed "build complete solution on your own" so blocking on PAT plumbing serves nothing; ES execution stays remote per standing doctrine, code authoring is local :: waiting for PAT repo-access update

- [2026-10-01] SCOPE — Autonomous through Phase 1-3, no per-phase approvals :: user explicit instruction "I don't want to approve each of the phases manually… build complete solution on your own", decisions tracked here :: per-phase approval gates

- [2026-10-01] BASE — Independent implementation, atlas consulted as reference only (no code copied) :: user: "I want to make sure it's my own repo" :: forking/copying MIT code wholesale

- [2026-10-01] PRIVACY — Public README describes product on its own terms; internal doc keeps the competitive framing :: user: "fine to mention [the successor framing] internally, but don't make it so obvious publicly" :: public attribution section naming the incumbent
