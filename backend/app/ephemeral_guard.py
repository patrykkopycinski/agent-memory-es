"""Write-time anti-ephemeral guard for the `semantic` kind: text that mostly reports one
run's incident details (a bare count of events, a high-resolution timestamp, a duration in
the past tense) is refused with HTTP 422, because it pollutes the bank with facts no later
recall will match. The design copies the durable-only write policy of the Nightshift Cortex
optimiser (kibana PR #303323): single-run details never become page content there; here
they never become semantic memories. The `episodic` kind is exempt by design — an episodic
memory is a dated event, so dates and counts are its substance, not pollution.

The line between a durable fact and an incident report is judgement, not mechanics, so this
guard stays deliberately small and explicit:

1. a count of concrete events (``5,944 errors`` / ``23 restarts``) that is not a rate or
   size limit of the system — ``at most 100 requests per second`` / ``capped at 5000
   documents`` are durable configuration; ``5944 errors`` is a property of one run;
2. a timestamp more precise than a calendar year — the moment one thing happened is never
   durable, and its precision makes recall worse (an ask phrased at another granularity
   never matches);
3. a past-tense duration — ``took 18 minutes`` / ``was down for 2 hours`` report how long
   one incident lasted, unlike a still-true schedule ("``departs at 6``", "``starts at
   02:00``"), which keeps a present-tense verb.

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

# A rate or size limit — or a possession/retention statement — names a durable property of
# the system, not of one run. It is recognised either by a limiting or possessive verb
# carrying the same number ("accepts at most 100 requests", "capped at 5000 documents",
# "kafka retains 7 days of messages", "the index has 3 shards") or by a per-unit
# denominator right after the counted noun ("100 requests per second", "500 docs a day",
# "2 connections per node"). Present tense only: "had 23 restarts" stays a tally. A
# per-unit rate on a past-tense incident ("dropped 5944 errors per hour that night") also
# frames a limit-shaped number; separating those needs judgement the guard defers to
# recall ranking (see the module docstring).
_LIMIT_FRAME_RE = re.compile(
    r"\b(?:accepts?|allows?|handles?|supports?|serves?|rate.?limits?|limits?|"
    r"holds?|contains?|stores?|retains?|has|have|"
    r"caps?\s+(?:at\s+)?|capped\s+(?:at\s+)?|max(?:imum)?|min(?:imum)?|"
    r"at most|at least|up to|no more than|no fewer than)"
    r"(?:\s*(?:of|is|are|to|at|:)\s*|\s+)"
    r"(?:about\s+|roughly\s+|approximately\s+|~\s*)?\d[\d.,]*",
    re.IGNORECASE,
)
_PER_UNIT_RE = re.compile(
    r"\s*(?:per\s+|an?\s+|each\s+)(?:milli?seconds?|ms|seconds?|secs?|minutes?|mins?|"
    r"hours?|hrs?|days?|weeks?|months?|pods?|nodes?|instances?|shards?|indices?|"
    r"indexes?|users?|workers?|hosts?|cores?|calls?)\b",
    re.IGNORECASE,
)

# A timestamp more precise than a year, in shapes that are moments by themselves: a full
# calendar date (2026-11-10, Nov 9, November 9th — "may" is excluded because the modal
# verb makes "may 5" unreadable), a clock time with seconds (10:42:15 — schedules are
# never stated to the second), or a clock time with a timezone (10:42Z, 07:45+02:00 —
# an offset marks a moment). A bare year (2026) or month+year stays writable. A bare
# HH:MM — with or without am/pm — is writable too: "starts at 02:00" and "departs at
# 6:30 pm nightly" are schedules, so their moment reading ("finished at 10:42 am") is
# caught by the past-tense verb in _CLOCK_MOMENT_RE, not here.
_PRECISE_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"|\b\d{1,2}:\d{2}:\d{2}\b"
    r"|\b\d{1,2}:\d{2}(?:Z\b|[+-]\d{2}:?\d{2})"
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}"
    r"(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)

# A bare clock time reads as a moment only through a verb that names a completed event
# ("the run finished at 10:42"). Tense-ambiguous verbs (was, started, began, ran) are
# excluded on purpose: they also carry habitual schedules ("the maintenance window was at
# 02:00 every night", "the sync started at 02:00 nightly"), and those misses are left to
# recall ranking — the same deliberate-smallness trade as everywhere else in this module.
_CLOCK_MOMENT_RE = re.compile(
    r"\b(?:finished|ended|completed|stopped|halted|crashed|failed|occurred|happened|"
    r"dropped|aborted|timed out|went down)\s+"
    r"(?:at|by|around|about)\s+(?:about\s+|roughly\s+)?\d{1,2}:\d{2}"
    r"(?:\s*(?:am|pm)\b|\s*[ap]\.?\s*m\.?)?",
    re.IGNORECASE,
)

# A duration attached to a completed-event verb: "took 18 minutes", "was down for 2 hours",
# "ran for 40 seconds", "lasted 3 days", "took twenty minutes". The completed-event marker
# keeps schedules ("departs at 6", "takes 2 hours every night" — habitual present) writable;
# only a report of how long one finished thing lasted is refused. Still-ongoing states
# ("is still down for maintenance") are deliberately absent: they are not reports of a
# finished event, and their staleness is a recall-ranking problem, not a write-gate one.
_DURATION_RE = re.compile(
    r"\b(?:took|lasted|spent|ran for|was down for|were down for|"
    r"had been down for|went down for|went on for|elapsed)\s+"
    r"(?:about\s+|roughly\s+|approximately\s+|~\s*)?"
    r"(?:\d+[\d.,]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|a couple of|a few|several|half an?|an?)\s+"
    r"(?:milli?seconds?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b",
    re.IGNORECASE,
)


def _limit_frames(text: str, count_match: "re.Match") -> bool:
    """True when a rate/size limit or possession statement frames this counted noun
    (durable configuration, not a one-run tally): a per-unit denominator right after
    the noun, or a framing verb whose number overlaps the counted span. Overlap (not
    just co-occurrence) keeps a mixed sentence honest — 'accepts at most 100 requests;
    dropped 5944 requests' still rejects."""
    if _PER_UNIT_RE.match(text, count_match.end()):
        return True
    return any(frame.start() < count_match.end() and frame.end() > count_match.start()
               for frame in _LIMIT_FRAME_RE.finditer(text))


def reject_ephemeral(text: str) -> None:
    """ValueError (HTTP 422) if the text looks like a single-run incident report rather
    than a durable fact. The message names the pattern, never the match. Runs after
    secrets_guard on the same retain path; the `semantic` kind only."""
    if not text:
        return
    if any(m for m in _COUNT_RE.finditer(text) if not _limit_frames(text, m)):
        raise ValueError(
            "text reports a count of events (numbers like tallies belong to one run, not "
            "to the system); write the pattern instead (e.g. 'errors recur', 'restarts "
            "repeatedly')")
    if _PRECISE_TS_RE.search(text):
        raise ValueError(
            "text carries a precise timestamp (a moment one thing happened); durable facts "
            "keep at most a year")
    if _CLOCK_MOMENT_RE.search(text):
        raise ValueError(
            "text pins a clock time to a finished event (a moment one thing happened); "
            "keep the time on the episodic memory or drop it from the durable claim")
    if _DURATION_RE.search(text):
        raise ValueError(
            "text reports how long one past event lasted; distil it into a durable claim "
            "(e.g. 'slows down every morning') or store it as episodic")
