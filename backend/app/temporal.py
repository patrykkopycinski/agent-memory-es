"""Temporal retrieval arm: parse time expressions into a window, fill by relevance
spread across the range (not all from one end).

Deterministic lightweight parser: 'last N days/weeks/months', 'in YYYY', 'YYYY-MM',
'this|last year/month/week'. Falls back to None (arm skipped) like Hindsight's
'no date in query · skipped'.
"""
import re
import datetime

_PATTERNS = [
    (re.compile(r"(?i)\blast\s+(\d+)\s+(day|week|month)s?\b"), "last_n"),
    (re.compile(r"(?i)\b(?:in\s+)?(\d{4})-(\d{2})\b"), "ym"),
    (re.compile(r"(?i)\b(?:in\s+)?(\d{4})\b"), "year"),
    (re.compile(r"(?i)\bthis\s+(week|month|year)\b"), "this"),
    (re.compile(r"(?i)\blast\s+(week|month|year)\b"), "prev"),
]


def parse_window(query: str, now: datetime.datetime = None):
    """Returns (start_iso, end_iso) or None."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    for pat, kind in _PATTERNS:
        m = pat.search(query)
        if not m:
            continue
        if kind == "last_n":
            n, unit = int(m.group(1)), m.group(2)
            days = n * {"day": 1, "week": 7, "month": 30}[unit]
            return _iso(now - datetime.timedelta(days=days)), _iso(now)
        if kind == "ym":
            y, mo = int(m.group(1)), int(m.group(2))
            start = datetime.datetime(y, mo, 1, tzinfo=datetime.timezone.utc)
            end = _month_end(y, mo)
            return _iso(start), _iso(end)
        if kind == "year":
            y = int(m.group(1))
            return _iso(datetime.datetime(y, 1, 1, tzinfo=datetime.timezone.utc)), \
                   _iso(datetime.datetime(y, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc))
        if kind == "this":
            unit = m.group(1)
            if unit == "week":
                s = now - datetime.timedelta(days=now.weekday())
                return _iso(s), _iso(now)
            if unit == "month":
                s = now.replace(day=1)
                return _iso(s), _iso(now)
            return _iso(now.replace(month=1, day=1)), _iso(now)
        if kind == "prev":
            unit = m.group(1)
            if unit == "week":
                s = now - datetime.timedelta(days=now.weekday() + 7)
                return _iso(s), _iso(s + datetime.timedelta(days=6, hours=23))
            if unit == "month":
                first = now.replace(day=1)
                prev_end = first - datetime.timedelta(days=1)
                return _iso(prev_end.replace(day=1)), _iso(prev_end)
            return _iso(now.replace(year=now.year - 1, month=1, day=1)), \
                   _iso(now.replace(year=now.year - 1, month=12, day=31))
    return None


def spread_buckets(start_iso: str, end_iso: str, n: int = 4) -> list:
    """Split the window into n equal buckets so results spread across the range."""
    s, e = _dt(start_iso), _dt(end_iso)
    if e <= s:
        return [(start_iso, end_iso)]
    step = (e - s) / n
    out = []
    for i in range(n):
        bs, be = s + step * i, s + step * (i + 1) - datetime.timedelta(seconds=1) \
            if i < n - 1 else e
        out.append((_iso(bs), _iso(be)))
    return out


def _month_end(y: int, mo: int) -> datetime.datetime:
    nxt = datetime.datetime(y + (mo == 12), (mo % 12) + 1, 1, tzinfo=datetime.timezone.utc)
    return nxt - datetime.timedelta(seconds=1)


def _iso(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _dt(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
