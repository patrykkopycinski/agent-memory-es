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
  the key's `person` attribute (a legacy key without one falls back to its owner id), so two keys
  of one person are one approver. The author is excluded however they approve: an approve or
  reject is refused (`403`) when the approver key's `person` is the proposal's `proposed_by` or the
  person of any key of the proposal's owner, **or when the approver key belongs to the same
  `owner_id` as the proposal**, whatever person it names.
- Two overrides of the same active decision cannot both activate. An approval that would activate
  an override first moves the target to `superseded` (compare-and-set); only if that succeeded does
  the override activate. The loser stays `proposed` (it is not rejected) and its approval returns
  `409 override target is no longer an active decision`; if the activation then fails, the target
  is restored to `active`.
  Concurrent approvers of one override share the claim, but only the call that *created* it may
  release it. After the override is written the target is checked once more: a claim released in
  between is taken again, and a target held by another decision is reported as a conflict (below).
  A rejected (veto), withdrawn or expired override gives its claim back.
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
`approver`; anything else is `422` at mint time) and a **required** `person`, the human behind
the key (`422` without one): `POST /admin/keys?owner_id=..&role=approver&person=alice`. Keys
minted before `person` was required keep working as agent keys; name their person with
`POST /admin/keys/person?owner_id=..&person=alice` (admin token; only keys with no person are
changed). The audit row `key_set_person` (what was asked: owner and person) is written first, in
strict mode, and a failed audit write answers `503` with nothing changed. A second row,
`key_set_person_done`, records the outcome (`keys_changed`) on a best-effort basis, so the first row
means "intended", the second "done". The call returns `updated`, the number of keys changed.
Persons already stored in the key file are lowercased when it is read, so a hand-edited `Alice`
is the person `alice`.

**One human, one person string.** Persons are trimmed and lowercased at mint time, in
`set-person` and when the key file is read. Every comparison (author check, one approval per
person, objections, withdraw, veto list) is case-insensitive, so `Alice` and `alice` are one person.
Use the same string for every key of a human.

**Governed writes need a person.** A key without one (minted before `person` was required) can
still read, recall and write private or non-governed memories, but a team `kind:decision` retain,
an override, or promoting a private decision to team answers `403` ("name the person first via
/admin/keys/person"). The proposal also stores the author set at propose time
(`proposed_by_persons`: the author and every person named on a key of the owner then), and the
author check uses that stored set together with the current one, so revoking a sibling key later
cannot make the author an eligible approver. An owner can edit or withdraw its earlier proposals
whether `proposed_by` holds its person or its owner id. Only the `approver` role may call
`/governance/approve` and `/governance/reject` (`member` and `owner` keys get `403`), and an
approver key without a `person` cannot approve. Give the approver key and the person's agent keys
the same `person`, so the author check works across all of them; an approver key must not share
an `owner_id` with the agent keys whose proposals it approves.

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

`list` prints the open proposals and then a `WARNING:` line for every entry of
`override_conflicts`. Exit codes: `0` ok, `1` error, `2` usage error (argparse), `3` the approval
activated an override whose target another decision holds (a `WARNING:` line is printed).

## API

| Route | Who | Notes |
|---|---|---|
| `POST /memory/retain` | any key | optional `supersedes_decision`; response carries `decision_status`, `approvals_needed` |
| `GET /governance/proposals` | any key | open proposals with approvals so far, plus `override_conflicts`: active overrides that could not supersede their target (still active in recall; an operator decides). The `memory_proposals` MCP tool shows both and adds a `warning` when there are conflicts |
| `POST /governance/approve` `{doc_id, note?}` | `approver` | `403` author / non-approver; `409` not open or already approved. An override that activated but whose target another decision holds answers `200` with `activated: true, conflict: true` |
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
| `AMES_GOV_VETO_OWNERS` | empty | comma-separated person ids whose rejection is final (case-insensitive) |
| `AMES_GOV_RECONCILE_DAYS` | `7` | the worker re-checks linked overrides decided within this many days (unlinked ones are always checked) |
| `AMES_GOV_FULL_RECONCILE_HOURS` | `24` | once per this many hours the worker scans every approved override whatever its age (minimum 1 minute); the marker is per worker process, so a restart triggers one full scan |

## Guarantees

- **Audit first.** Every propose, approve, reject, withdraw, activate, supersede and expire writes
  an `am_mutations` row (actor, text hash) *before* the state changes. If the audit write fails,
  the transition does not happen. A compare-and-set attempt that loses its race keeps its row,
  marked `detail.outcome: conflict`; rows without that mark describe transitions that happened.
- **Concurrency.** Transitions are compare-and-set on the head passage (`if_seq_no` /
  `if_primary_term`) with retry, so simultaneous approvals both count and activation happens once.
  Status fields are mirrored to the other passages of a long decision.
- **Expiry and repair.** The worker pass expires stale proposals (every page) and then:
  restores a target claimed by an override that never activated, once the claim is older than five
  minutes and with no upper age limit. A target is restored only when its claimant is `proposed`,
  `rejected` or `expired` with no `approved_via`: never when the claimant is active or was itself
  superseded (A -> B -> C leaves A superseded). After a restore the successor is read again: if it
  became active in between, the target is claimed again. It also reconciles the active overrides
  that went live through approvals and were decided within `AMES_GOV_RECONCILE_DAYS`, plus any
  that lack the `supersedes` link; once per `AMES_GOV_FULL_RECONCILE_HOURS` the window is dropped
  and every approved override is checked, so an old one whose claim was lost is found too. If that
  full scan fails, the next one is due in an hour rather than on the next pass. An override whose
  target is not superseded by it (or that lacks the `supersedes` link) is completed. One that
  cannot be completed (the target is held by another decision or is gone) is logged as a warning, audited as `decision_override_conflict` with `outcome: conflict`
  and marked `approved_via: approvals_conflict`, which takes it out of later passes; an operator
  decides. Both passes read every page.
- **Fail closed on schema.** At startup the index mapping is read first (every index behind an
  alias is checked): when every governance
  field is already mapped with the right type nothing is written, so a role without
  `put_mapping` still starts. When a field is missing or has the wrong type the mapping is applied;
  if that fails the error is logged and the service refuses to serve retain/recall rather than run
  without it.
- **Existing banks.** Run the backfill once after upgrading so earlier team decisions are in
  force; until then they are still excluded from consolidation.
