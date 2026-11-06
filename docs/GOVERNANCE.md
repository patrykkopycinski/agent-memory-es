# Decision governance

A shared (team) memory bank needs a way to stop one writer, human or agent, from silently
rewriting what everyone else relies on. Governance puts a review step in front of **decisions**.

## What is governed

A *governed entry* is a `semantic` memory with visibility `team` and the tag `kind:decision`.
Private decisions, and team memories without that tag (how-tos, notes), behave as before.

## Lifecycle

```
retain ──> proposed ──> active ──> superseded      (an approved override replaces it)
              ├──────> rejected                     (veto, or withdrawn by the author)
              └──────> expired                      (no decision within the TTL)
```

- A new decision needs **1** approval from someone other than its author.
- An override (`supersedes_decision=<id>` on retain; the target must be an *active* governed
  decision) needs **2** distinct non-author approvals. On activation the old decision becomes
  `superseded` (`superseded_by` / `supersedes` link both ends).
- The author's approval never counts; the same person approving twice counts once. "Person" is
  the key's `person` attribute (falling back to its owner id), so two keys of one person are one
  approver, and the author is excluded whichever of their keys approves.
- Two overrides of the same active decision cannot both activate. An approval that would activate
  an override first moves the target to `superseded` (compare-and-set); only if that succeeded does
  the override activate. The loser stays `proposed` (it is not rejected) and its approval returns
  `409 override target is no longer an active decision`; if the activation then fails, the target
  is restored to `active`.
- Approvals are bound to the text hash. Editing a proposal (author only) resets them but keeps the
  original `proposal_expires_at`: an edit never extends the deadline. The edit writes the new
  text first and removes the old passages only afterwards, so a collision with another entry or a
  failure partway leaves the original proposal intact.
- A rejection by a *veto owner* rejects the proposal outright; any other rejection is an
  objection and does not block approval.
- The author deleting a proposal withdraws it (`rejected`, reason `withdrawn`); it is not erased.
- An **active** decision is read-only. Update, delete and promote return `409`
  (`governed decision: propose an override`), for its creator too. The check covers every doc of
  the decision's `doc_group`, so a sibling doc is no way around it.
- A governed write ignores the caller's `doc_group` and gets a server-generated one (so does a
  promoted private decision). Status changes touch only the passages of the approved text.
- Consolidation and near-duplicate linking never touch governed entries, on either side.
- Promoting a private `kind:decision` memory to the team bank creates a proposal.

## Who can approve

Approval is a human act. Keys carry a role from a fixed allowlist (`member`, `owner`,
`approver`; anything else is `422` at mint time) and an optional `person`:
`POST /admin/keys?owner_id=..&role=approver&person=alice`. Only the `approver` role may call
`/governance/approve` and `/governance/reject` (`member` and `owner` keys get `403`), and an
approver key without a `person` cannot approve. Give the approver key and the person's agent keys
the same `person`, so the author check works across all of them.

An approver key is accepted on `/governance/*` only: every other route, including recall and
`/mcp/tools/*`, answers `403`. `GET /governance/proposals` stays open to every key. No MCP tool maps to approve or reject (`/mcp/tools/approve` is a 404), and the MCP servers
expose only a read-only `memory_proposals` tool.

`scripts/ames_approve.py` is the reference client. It reads the key from an interactive prompt
only, and refuses to run without a TTY: never from an argument, environment variable or file.
Before asking for the key it shows the target URL and asks for confirmation, and it refuses
plain `http://` to anything but a loopback host.

```
scripts/ames_approve.py list
scripts/ames_approve.py show <doc_id>
scripts/ames_approve.py approve <doc_id> [--note TEXT]
scripts/ames_approve.py reject <doc_id> --reason TEXT
```

## API

| Route | Who | Notes |
|---|---|---|
| `POST /memory/retain` | any key | optional `supersedes_decision`; response carries `decision_status`, `approvals_needed` |
| `GET /governance/proposals` | any key | open proposals with approvals so far |
| `POST /governance/approve` `{doc_id, note?}` | `approver` | `403` author / non-approver; `409` not open or already approved |
| `POST /governance/reject` `{doc_id, reason}` | `approver` | veto owner rejects; others object |
| `POST /admin/governance/backfill` | admin token | marks existing team decisions `active` (`approved_via: backfill`, `proposed_by` = the owner); compare-and-set per doc, idempotent |
| `POST /memory/recall` | any key | `include_inactive_decisions: true` also returns rejected / expired / superseded |

Recall returns active decisions as before; proposals carry `decision_status: proposed` (the MCP
script renders them as "PROPOSED, not in force"); closed decisions are hidden unless asked for.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AMES_GOV_NEW_APPROVALS` | `1` | approvals to activate a new decision |
| `AMES_GOV_OVERRIDE_APPROVALS` | `2` | approvals to activate an override (never below the new-decision value) |
| `AMES_GOV_PROPOSAL_TTL_DAYS` | `14` | a proposal older than this expires; `0` disables expiry |
| `AMES_GOV_VETO_OWNERS` | empty | comma-separated person ids whose rejection is final |

## Guarantees

- **Audit first.** Every propose, approve, reject, withdraw, activate, supersede and expire writes
  an `am_mutations` row (actor, text hash) *before* the state changes. If the audit write fails,
  the transition does not happen. A compare-and-set attempt that loses its race keeps its row,
  marked `detail.outcome: conflict`; rows without that mark describe transitions that happened.
- **Concurrency.** Transitions are compare-and-set on the head passage (`if_seq_no` /
  `if_primary_term`) with retry, so simultaneous approvals both count and activation happens once.
  Status fields are mirrored to the other passages of a long decision.
- **Expiry.** The worker pass expires stale proposals (every page, not just the first) and
  completes any override whose old decision was not yet marked superseded. It also restores a
  target that was claimed by an override that never activated (older than five minutes).
- **Fail closed on schema.** If the governance field mapping cannot be applied at startup the
  error is logged and the service refuses to serve retain/recall rather than run without it.
- **Existing banks.** Run the backfill once after upgrading so earlier team decisions are in
  force; until then they are still excluded from consolidation.
