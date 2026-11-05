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
# and the value must pass _keyword_value_is_secret (16+ chars, letters and digits, not a slug,
# path, version or commit hash).
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
_SLUG_SEP = re.compile(r"[-_+=]+")
# One slug part: dot-joined pieces, each a lowercase word with a short numeric suffix
# ("client2", "a3b", "35b"), or a version number ("v3", "3.8", "9.6.0"). A part that is purely
# alphabetic may also be Capitalized or UPPER ("SNAPSHOT", "Users"); a part with digits must be
# lowercase, so random mixed-case/digit runs ("Zx81QwQw") never qualify.
_SLUG_PIECE = re.compile(r"^(?:[a-z]{1,}[0-9]{0,4}[a-z]{0,2}|v?[0-9]{1,8}[a-z]{0,2})$")
_HEX_HASH = re.compile(r"^[0-9a-f]{7,40}$")
_COMMIT_WORDS = {"commit", "sha", "rev", "revision", "hash", "ref", "merge", "merged", "at", "pr", "head"}
_COMMIT_CONTEXT = re.compile(r"\b(?:commit|sha1?|rev|revision|hash|merge[d]?|cherry-pick|head)\b[^\n]{0,24}$",
                             re.IGNORECASE)


def _is_slug_piece(piece: str) -> bool:
    if piece.isalpha() and (piece.islower() or piece.isupper() or piece == piece.capitalize()):
        return True
    return _SLUG_PIECE.match(piece) is not None


def _is_slug(value: str) -> bool:
    parts = [p for p in _SLUG_SEP.split(value) if p]
    if len(parts) < 2:
        return False
    return all(all(_is_slug_piece(x) for x in p.split(".") if x) for p in parts)


def _is_path(value: str) -> bool:
    """A filesystem path or owner/repo reference, not a credential: either rooted ('/', '~/',
    './') with 2+ slashes, or every '/'-segment is itself a slug. Base64 secrets that merely
    contain '/' (AWS-style) have mixed-case/digit segments, so they stay rejected."""
    if "/" not in value:
        return False
    if re.match(r"^(?:~|\.{1,2})?/", value) and value.count("/") >= 2:
        return True
    segs = [x for x in value.split("/") if x]
    return len(segs) >= 2 and all(_is_slug(x) or _is_slug_piece(x) for x in segs)


def _is_commit_hash(value: str, scheme_word: str, before: str) -> bool:
    """7-40 hex chars is also the shape of a hex API key, so it is exempt only with commit
    context: a commit-ish scheme word ("auth: commit 1470bfd…") or one just before the keyword."""
    if not _HEX_HASH.match(value):
        return False
    return scheme_word.strip().lower() in _COMMIT_WORDS or bool(_COMMIT_CONTEXT.search(before))


def _keyword_value_is_secret(value: str, scheme_word: str = "", before: str = "") -> bool:
    return (len(value) >= KEYWORD_MIN_VALUE_CHARS
            and re.search(r"[A-Za-z]", value) is not None
            and re.search(r"[0-9]", value) is not None
            and not _is_slug(value)
            and not _is_path(value)
            and not _is_commit_hash(value, scheme_word, before))


def find_secret(text: Optional[str]) -> Optional[str]:
    """Name of the first secret pattern `text` matches, or None. Never returns the match."""
    if not text:
        return None
    for name, rx in SHAPE_RES:
        if rx.search(text):
            return name
    for m in KEYWORD_RE.finditer(text):
        if _keyword_value_is_secret(m.group(4), m.group(3) or "", text[max(0, m.start() - 40):m.start()]):
            return "keyword_value"
    return None


def find_secret_in_tags(tags) -> Optional[str]:
    """Like find_secret over each caller-supplied tag, on the RAW strings: norm_tags lowercases
    and turns '_'/'-' into spaces, which would hide a token shape. Also tries the lowercased tag,
    since the shapes are case-sensitive and an UPPER-cased `GHP_...` tag is still a secret
    attempt. Non-strings are left for norm_tags to reject."""
    if not isinstance(tags, (list, tuple)):
        return None
    for t in tags:
        if isinstance(t, str):
            hit = find_secret(t) or find_secret(t.lower())
            if hit:
                return hit
    return None
