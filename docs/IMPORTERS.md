# Importers

Migrate existing memory corpora into agent-memory-es with one command — every
entry goes through the normal `retain` path (embeddings, entity extraction,
tombstone guards all apply).

```
python scripts/import_memory.py <source> [--dry-run] [--owner you]
                                   [--format governance|hindsight|markdown]
                                   [--visibility private|team|common]
                                   [--include-changelog]
```

Always `--dry-run` first: it parses and previews without retaining.

## Governance stores (per-repo markdown, Claude Code style)

A directory containing `profile.md`, `decisions-log.md`, `design-decisions.md`,
`changelog.md`, `refactoring-log.md`, `session-handoff.md` (any subset).

| Source file | ames kind | visibility | Notes |
|---|---|---|---|
| `profile.md` | procedural | team | repo overview, paths, cardinal rules — one entry |
| `decisions-log.md` | semantic | team | **each `### DATE -- Title` decision = one entry** |
| `design-decisions.md` | semantic | team | same per-decision split |
| `refactoring-log.md` | procedural | private | structural changes, regression tracing |
| `changelog.md` | procedural | private | **skipped by default** (re-derivable from git); `--include-changelog` imports per-day summaries |
| `session-handoff.md` | — | — | **never imported** — ephemeral session state, not bank material |

The per-decision split matters: his 108 KB decisions-log that only ever shows
its last 120 lines becomes individually retrievable entries — that is the
point of moving files into a bank.

## Hindsight export

`.json` (array) or `.jsonl` (one object per line). Field mapping is tolerant:
`text|content|body`, `kind|type` (`fact`→semantic, `procedure`→procedural,
`note|observation`→episodic), `occurred_at|created_at|timestamp`, `visibility`.

## Plain markdown (`CLAUDE.md`, `AGENTS.md`, notes)

One entry per `##` section (preamble becomes its own entry). Procedural /
private by default — these files usually describe one machine's setup.

## Recompression caveats

- `--visibility` overrides the per-format mapping for **all** entries — use
  only when you know the corpus is shareable.
- Importing the same source twice duplicates entries (no content-hash dedup
  yet); drop/re-create the indices or use a fresh owner_id for re-imports.
- The promotion guard still blocks team/common retention of texts carrying
  secret markers, same as interactive retain.
