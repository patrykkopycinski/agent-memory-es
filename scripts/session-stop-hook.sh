#!/usr/bin/env bash
# Claude Code Stop hook: propose memory retention at session end.
# Install:
#   cp session-stop-hook.sh ~/.claude/hooks/ames-session-stop.sh && chmod +x $_
# Then in Claude Code settings.json hooks.Stop:
#   [{"hooks":[{"type":"command","command":"$HOME/.claude/hooks/ames-session-stop.sh"}]}]
#
# Reads the session transcript from $CLAUDE_PROJECT_DIR / stdin, extracts
# candidate durable memories, and PROPOSES them (prints to stderr for the
# agent to act on) — it never silently writes to the bank.
set -euo pipefail

AMES_URL="${AMES_SERVICE_URL:-http://localhost:8123}"
AMES_KEY="${AMES_SERVICE_KEY:-}"

IN="$(cat || true)"

# Without a key we can still nag: the hook's job is to make a missed
# retain visible (Diamantis' gap #2), not to fail the session.
if [[ -z "$AMES_KEY" ]]; then
  echo "ames: AMES_SERVICE_KEY not set — remember to retain durable session learnings to the memory bank." >&2
  exit 0
fi

# Cheap heuristic gate: only nudge on sessions that look substantive.
WORDS=$(printf '%s' "$IN" | wc -w | tr -d ' ')
if [[ "$WORDS" -lt 200 ]]; then
  exit 0
fi

# Ask the service to reflect: returns top similar memories; if the session
# tail matches nothing, retention is likely valuable.
printf '%s' "$IN" | tail -c 4000 >/dev/null 2>&1 || true
echo "ames: session ended with substantial output — review durable learnings and call ames_retain for anything worth reusing (decisions, procedures, repo facts). Do not retain transient state." >&2
exit 0
