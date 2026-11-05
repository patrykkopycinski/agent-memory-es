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
# and the value must pass _keyword_value_is_secret (16+ chars, letters and digits, not a lowercase
# slug or a commit hash).
# Keywords: `api key` with a space or `_`/`-`, and any `<name>_KEY` / `<name>-key` suffix form
# (SERVICE_KEY=...). The suffix alternative is one alnum char plus `_key`: no nested quantifier,
# so it cannot backtrack. The other quantifiers are on single character classes.
KEYWORD_RE = re.compile(
    r"(api[_\- ]?key|[A-Za-z0-9][_-]key|token|secret|password|passwd|authorization|credentials?|auth)"
    r"([\"'\s]*[:=][\"'\s:=]*)([A-Za-z]+\s+)?([A-Za-z0-9_\-/.+=~]{8,})", re.IGNORECASE)

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
# Hash exemption: 7-12 hex or exactly 40 (a git SHA-1). 32-hex values (Datadog, Sentry, npm keys)
# and 64-hex values are never exempt, and neither are upper-case hex strings.
_HEX_HASH = re.compile(r"^(?:[0-9a-f]{7,12}|[0-9a-f]{40})$")
_COMMIT_WORDS = frozenset({"commit", "sha", "rev", "hash"})
# There is deliberately NO path / model-id exemption: a value with a '/' is judged like any other
# (only the lowercase slug above is exempt). Such values ("meta-llama/Llama-3.3-70B-Instruct",
# "src/plugins2/auth/README.md") are rejected as false positives (see KNOWN_FALSE_POSITIVES in
# the tests); every path exemption tried was bypassable with random chunks shaped like it.


def _is_slug(value: str) -> bool:
    parts = [p for p in _SLUG_SEP.split(value) if p]
    return len(parts) >= 2 and all(_SLUG_WORD.match(p) for p in parts)


def _is_commit_hash(value: str, scheme_word: str) -> bool:
    """A 7-12 or 40 char hex string is also the shape of a hex API key, so it is exempt only when
    the scheme word right after the separator says it is a hash ("auth: commit 1470bfd…")."""
    return _HEX_HASH.match(value) is not None and scheme_word.strip().lower() in _COMMIT_WORDS


def _keyword_value_is_secret(value: str, scheme_word: str = "") -> bool:
    return (len(value) >= KEYWORD_MIN_VALUE_CHARS
            and re.search(r"[A-Za-z]", value) is not None
            and re.search(r"[0-9]", value) is not None
            and not _is_slug(value)
            and not _is_commit_hash(value, scheme_word))


def find_secret(text: Optional[str]) -> Optional[str]:
    """Name of the first secret pattern `text` matches, or None. Never returns the match."""
    if not text:
        return None
    for name, rx in SHAPE_RES:
        if rx.search(text):
            return name
    for m in KEYWORD_RE.finditer(text):
        if _keyword_value_is_secret(m.group(4), m.group(3) or ""):
            return "keyword_value"
    return None


def find_secret_in_tags(tags) -> Optional[str]:
    """Like find_secret over each caller-supplied tag, on the RAW strings: norm_tags lowercases
    and turns '_'/'-' into spaces, which would hide a token shape. Also tries the lowercased tag,
    since the shapes are case-sensitive and an UPPER-cased `GHP_...` tag is still a secret
    attempt. That retry is for tags ONLY: upper-case shapes (`GHP_…`, `SK-…`) in retained TEXT
    are not matched. Non-strings are left for norm_tags to reject."""
    if not isinstance(tags, (list, tuple)):
        return None
    for t in tags:
        if isinstance(t, str):
            hit = find_secret(t) or find_secret(t.lower())
            if hit:
                return hit
    return None
