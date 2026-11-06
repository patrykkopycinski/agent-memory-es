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
- The author's approval never counts; the same person approving twice counts once.
- Approvals are bound to the text hash. Editing a proposal (author only) resets them.
- A rejection by a *veto owner* rejects the proposal outright; any other rejection is an
  objection and does not block approval.
- The author deleting a proposal withdraws it (`rejected`, reason `withdrawn`); it is not erased.
- An **active** decision is read-only. Update, delete and promote return `409`
  (`governed decision: propose an override`), for its creator too.
- Consolidation and near-duplicate linking never touch governed entries, on either side.
- Promoting a private `kind:decision` memory to the team bank creates a proposal.

## Who can approve

Approval is a human act. Keys carry a role; only the `approver` role may call
`/governance/approve` and `/governance/reject` (`member` and `owner` keys get `403`). Mint the
approver key for the same person id as that person's agent key, so the author check works across
both. No MCP tool maps to approve or reject (`/mcp/tools/approve` is a 404), and the MCP servers
expose only a read-only `memory_proposals` tool.

`scripts/ames_approve.py` is the reference client. It reads the key from an interactive prompt
only, and refuses to run without a TTY: never from an argument, environment variable or file.

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
| `POST /admin/governance/backfill` | admin token | marks existing team decisions `active` (`approved_via: backfill`); idempotent |
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
  the transition does not happen.
- **Concurrency.** Transitions are compare-and-set on the head passage (`if_seq_no` /
  `if_primary_term`) with retry, so simultaneous approvals both count and activation happens once.
  Status fields are mirrored to the other passages of a long decision.
- **Expiry.** The worker pass expires stale proposals and completes any override whose old
  decision was not yet marked superseded.
- **Existing banks.** Run the backfill once after upgrading so earlier team decisions are in
  force; until then they are still excluded from consolidation.
