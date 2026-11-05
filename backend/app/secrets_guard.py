"""Write-time secret guard for retain: text that looks like a credential is rejected (HTTP 422).

The patterns are a Python port of KEYWORD_RE and SHAPE_RES in
opencode-harness/plugin/autolearn/redact.ts. Keep the two in sync: a pattern added there
should be added here, or autolearn redacts what the service would still accept.
"""
import re
from typing import Optional

# key-ish word, separator, optional scheme word, 8+ char value (redact.ts KEYWORD_RE).
# Deliberately stricter than redact.ts, which only masks: a 422 on prose like "auth migration"
# or "token estimator" would reject ordinary facts. Here the separator must contain ':' or '=',
# the value must be 16+ chars mixing letters and digits, and it must not be a lowercase slug
# (model ids, version strings, flow names such as "gpt-4o-2024-08-06" or "deepseek-v3.2-flash").
KEYWORD_RE = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|authorization|credentials?|auth)"
    r"([\"'\s]*[:=][\"'\s:=]*)([A-Za-z]+\s+)?([A-Za-z0-9_\-/.+=]{8,})", re.IGNORECASE)

# Well-known token shapes that appear without a keyword in front of them (redact.ts SHAPE_RES).
SHAPE_RES = (
    ("shape_sk", re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")),
    ("shape_gh_classic", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("shape_gh_fine", re.compile(r"\bgithub_" r"pat_[A-Za-z0-9_]{20,}")),
    ("shape_xox", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("shape_akia", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("shape_aiza", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("shape_jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("shape_bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE)),
    # Header only. redact.ts also masks the key body up to the END line (it rewrites text);
    # a guard that rejects the write needs just the header to refuse it.
    ("shape_pem_header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)

KEYWORD_MIN_VALUE_CHARS = 16
_SLUG_SEP = re.compile(r"[-_.+/=]+")
# A lowercase word with an optional short numeric suffix ("client2", "oauth2"), or a version
# token ("v3", "2024", "4o", "08").
_SLUG_WORD = re.compile(r"^(?:[a-z]{2,}[0-9]{0,4}[a-z]?|v?[0-9]{1,8}[a-z]?)$")


def _is_slug(value: str) -> bool:
    parts = [p for p in _SLUG_SEP.split(value) if p]
    return len(parts) >= 2 and all(_SLUG_WORD.match(p) for p in parts)


def _keyword_value_is_secret(value: str) -> bool:
    return (len(value) >= KEYWORD_MIN_VALUE_CHARS
            and re.search(r"[A-Za-z]", value) is not None
            and re.search(r"[0-9]", value) is not None
            and not _is_slug(value))


def find_secret(text: Optional[str]) -> Optional[str]:
    """Name of the first secret pattern `text` matches, or None. Never returns the match."""
    if not text:
        return None
    for name, rx in SHAPE_RES:
        if rx.search(text):
            return name
    for m in KEYWORD_RE.finditer(text):
        if _keyword_value_is_secret(m.group(4)):
            return "keyword_value"
    return None


def find_secret_in_tags(tags) -> Optional[str]:
    """Like find_secret over each caller-supplied tag, on the RAW strings: norm_tags lowercases
    and turns '_'/'-' into spaces, which would hide a token shape. Non-strings are left for
    norm_tags to reject."""
    if not isinstance(tags, (list, tuple)):
        return None
    for t in tags:
        if isinstance(t, str):
            hit = find_secret(t)
            if hit:
                return hit
    return None
