"""Write-time anti-ephemeral guard for the `semantic` kind: text that mostly reports one
run's incident details (a bare count of events, a high-resolution timestamp, a duration in
the past tense) is refused with HTTP 422, because it pollutes the bank with facts no later
recall will match. The design copies the durable-only write policy of the Nightshift Cortex
optimiser (kibana PR #303323): single-run details never become page content there; here
they never become semantic memories. The `episodic` kind is exempt by design — an episodic
memory is a dated event, so dates and counts are its substance, not pollution.

The line between a durable fact and an incident report is judgement, not mechanics, so this
guard stays deliberately small and explicit:

1. a count of concrete events (``5,944 errors`` / ``23 restarts`` / ``97 percent of
   requests``) — the number is a property of one run, not of the system;
2. a timestamp (or date-time pair) more precise than a calendar year — the moment one
   thing happened is never durable, and its precision makes recall worse (an ask
   phrased at another granularity never matches);
3. a *past-tense* duration — ``took 18 minutes`` / ``was down for 2 hours`` report how long
   one incident lasted, unlike a still-true schedule ("``departs at 6``"), which keeps a
   present-tense verb.

Anything the patterns miss is left to recall ranking and the reader; the guard must stay
cheap (no model call) and its rejections explainable (each message names the pattern, never
the match, mirroring secrets_guard). New pattern rules go in tests first.
"""
import re

# A number (thousands separated or not, optionally decimal, optionally a leading "about"/
# "roughly"/"~") followed, within a few words, by a plural noun of occurrence:
# errors, requests, restarts, pods, nodes, timeouts... The noun list is a closed set so an
# ordinary "5 dashboards" or "2 approvals" (configuration facts, not event tallies) stays
# writable, and the set is extended only when a rejection is demonstrated by a test first.
_COUNTED_NOUNS = (
    "errors?", "failures?", "failed attempts?", "warnings?", "timeouts?", "aborts?",
    "restarts?", "crashes?", "panics?", "drops?", "drops? out", "occurrences?",
    "incidents?", "alerts?", "events?", "hits?", "misses?", "retries?",
    "requests?", "calls?", "invocations?", "queries?", "jobs?", "tasks?", "runs?",
    "messages?", "docs?", "documents?", "records?", "entries?",
    "pods?", "nodes?", "replicas?", "instances?", "containers?", "shards?",
    "users?", "sessions?", "connections?",
)
_COUNT_RE = re.compile(
    r"\b(?:about|roughly|approximately|~\s*)?\d[\d.,]*\s+(?:[a-z]+\s+){0,2}(?:" + "|".join(_COUNTED_NOUNS) + r")\b",
    re.IGNORECASE,
)

# A timestamp more precise than a year: 2026-11-10, 10:42, 10:42:15, 10.42, T10:42Z,
# 07:45+02:00, 2026-11-10T10:42:15Z. A bare year (2026) or a bare month+year does not
# match; a date with a time always does; a bare calendar date does too — "checked on
# Nov 9" is one-run detail, while "since 2026" is a durable qualifier the extraction
# prompt itself asks for.
_PRECISE_TS_RE = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?Z?(?:[+-]\d{2}:?\d{2})?)?)"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b"
    r"|\b\d{1,2}[.:]\d{2}\s*(?:am|pm)\b"
    r"|\b(?:0?\d|1[0-2]):[0-5]\d(?:\s*[ap]\.?m\.?)\b"
    r"|\d{4}-\d{2}-\d{2}T[\d:.]+Z",
    re.IGNORECASE,
)

# A duration attached to a past-tense verb: "took 18 minutes", "was down for 2 hours",
# "ran for 40 seconds", "lasted 3 days", "took six minutes". The tense marker keeps
# schedules ("departs at 6", "takes 2 hours every night" — habitual present) writable;
# only a report of how long one finished thing lasted is refused.
_DURATION_RE = re.compile(
    r"\b(?:took|lasted|spent|ran for|was down for|were down for|has been down for|"
    r"had been down for|is still down for|went down for|went on for|elapsed)\s+"
    r"(?:about\s+|roughly\s+|approximately\s+|~\s*)?"
    r"(?:\d+[\d.,]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"a couple of|a few|several|half an?|an?)\s+"
    r"(?:milli?seconds?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b",
    re.IGNORECASE,
)

_EXEMPT_PREFIXES = (
    # An episodic memory is a dated event by design; other kinds get no exemption.
)


def reject_ephemeral(text: str) -> None:
    """ValueError (HTTP 422) if the text looks like a single-run incident report rather
    than a durable fact. The message names the pattern, never the match. Runs after
    secrets_guard on the same retain path; the `semantic` kind only."""
    if not text:
        return
    if _COUNT_RE.search(text):
        raise ValueError(
            "text reports a count of events (numbers like tallies belong to one run, not "
            "to the system); write the pattern instead (e.g. 'errors recur', 'restarts "
            "repeatedly')")
    if _PRECISE_TS_RE.search(text):
        raise ValueError(
            "text carries a precise timestamp (a moment one thing happened); durable facts "
            "keep at most a year")
    if _DURATION_RE.search(text):
        raise ValueError(
            "text reports how long one past event lasted; distil it into a durable claim "
            "(e.g. 'slows down every morning') or store it as episodic")
