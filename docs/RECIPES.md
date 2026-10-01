# Recipes

## Recall-before-read (the token-saving loop)

Measured on real OmniRoute traffic: recall-first gating addresses ~3–9% of
input tokens on execution-heavy workloads, several times more on
onboarding/QA-heavy ones. The order matters:

1. **recall first** — before any repo exploration, call `ames_recall` with the
   task's topic. ~200–800 tokens.
2. **on HIT** — skip the exploration; the distilled memory replaces a
   multi-call read/grep episode (10k–100k+ tokens each).
3. **on MISS** — explore normally, then **retain** the distilled answer
   (`procedural` for how-to, `semantic` for decisions/facts) so the next
   session hits.

Paying recall on every turn *after* reading the repo is the wrong order —
that buys nothing. Recall gates the read; the read feeds the retain.

## Session-end retention (closing the weakest link)

Saving relies on agent discipline by default. Enforce visibility instead:

- **Claude Code**: install `scripts/session-stop-hook.sh` as a `Stop` hook
  (instructions in the file header). It nudges — never silently writes —
  so a missed retain becomes visible.
- **Hermes**: the plugin exposes `agent_memory_retain`; add it to your
  session-handoff checklist.

## MCP clients (Claude Code / Cursor)

The FastAPI service speaks plain REST; any MCP client can use the thin stdio
bridge:

```
python scripts/ames_mcp_server.py --url http://localhost:8123 --api-key KEY
```

Tools: `ames_recall`, `ames_retain`, `ames_reflect`. Claude Code config:

```json
{ "mcpServers": { "ames": {
    "command": "python",
    "args": ["/abs/path/scripts/ames_mcp_server.py"],
    "env": { "AMES_SERVICE_URL": "http://localhost:8123",
             "AMES_SERVICE_KEY": "your-key" } } } }
```

## What deserves the bank (and what doesn't)

Retain when a fact is **(a)** expensive to re-derive (multi-step exploration),
**(b)** stable across days/weeks, and **(c)** useful to more than one session
or person. Skip transient state, one-off facts, and anything one grep away.
Changelogs are re-derivable from git — retain the *interpretation*
("this refactor moved X to Y, because Z"), not the diff log.
