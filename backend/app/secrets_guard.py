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
# Hash exemption: 7-12 hex or exactly 40 (a git SHA-1). 32-hex values (Datadog, Sentry, npm keys)
# and 64-hex values are never exempt, and neither are upper-case hex strings.
_HEX_HASH = re.compile(r"^(?:[0-9a-f]{7,12}|[0-9a-f]{40})$")
_COMMIT_WORDS = frozenset({"commit", "sha", "rev", "hash"})
# Path pieces. A lone path segment ("Hqlmx7", "xk9mq") is an ordinary word, a number, or a word plus
# a 1-2 digit suffix ("plugins2"); mixed letter/digit chunks are accepted only INSIDE a '-'/'_'
# joined segment, where model ids live ("Qwen3-Coder-30B-A3B-Instruct", "Mixtral-8x7B-v0.1"), so
# short random chunks joined by '/' never qualify.
_PATH_NUM = re.compile(r"^(?:[0-9]{8}T[0-9]{6}|[0-9]{1,12})$")        # id, GCP project number, timestamp
_WORD_NUM = re.compile(r"^[a-z]{3,}[0-9]{1,2}$")                      # "plugins2", "weekly2"
_JOINED_PIECE = re.compile(
    r"^(?:[a-z]{1,2}[0-9]{1,4}[a-z]{1,2}[0-9]{1,4}"                   # short git-style id "b400b54"
    r"|[A-Z][a-z]{2,}[0-9]{0,2}|(?:[A-Z][a-z]{2,}){2,3}"             # "Qwen3", "DeepSeek"
    r"|V[0-9]{1,2}|[0-9]{1,3}[A-Z]|[A-Z][0-9]{1,2}[A-Z]|[0-9]{1,2}x[0-9]{1,3}[A-Z])$")  # "V3", "70B", "A3B", "8x7B"
_PATH_ROOT = re.compile(r"^(?:~|\.{1,2})?/")
_ALNUM_RUN = re.compile(r"[A-Za-z0-9]{12,}")


def _is_slug_piece(piece: str) -> bool:
    if piece.isalpha() and (piece.islower() or piece.isupper() or piece == piece.capitalize()):
        return True
    return _SLUG_PIECE.match(piece) is not None


def _is_slug(value: str) -> bool:
    parts = [p for p in _SLUG_SEP.split(value) if p]
    if len(parts) < 2:
        return False
    return all(all(_is_slug_piece(x) for x in p.split(".") if x) for p in parts)


def _has_random_run(value: str) -> bool:
    """A 12+ char alphanumeric run mixing upper case, lower case and digits is key material."""
    return any(re.search(r"[a-z]", r) and re.search(r"[A-Z]", r) and re.search(r"[0-9]", r)
               for r in _ALNUM_RUN.findall(value))


def _dot_pieces(part: str) -> list:
    return [x for x in part.split(".") if x]


def _is_path_segment(seg: str) -> bool:
    parts = [p for p in _SLUG_SEP.split(seg) if p]
    if not parts:
        return False
    if len(parts) > 1:
        return all(_is_slug_piece(x) or _PATH_NUM.match(x) is not None or _JOINED_PIECE.match(x) is not None
                   for p in parts for x in _dot_pieces(p))
    return all((x.isalpha() and _is_slug_piece(x)) or _PATH_NUM.match(x) is not None
               or _WORD_NUM.match(x) is not None or re.match(r"^v?[0-9]{1,8}$", x) is not None
               for x in _dot_pieces(parts[0]))


def _path_pieces_look_random(segs: list) -> bool:
    """No real word (4+ letters with a vowel) anywhere, or most pieces mixing letters and digits."""
    pieces = [x for seg in segs for p in _SLUG_SEP.split(seg) for x in _dot_pieces(p)]
    if not any(x.isalpha() and len(x) >= 4 and re.search(r"[aeiouAEIOU]", x) for x in pieces):
        return True
    mixed = [x for x in pieces if re.search(r"[A-Za-z]", x) and re.search(r"[0-9]", x) and not re.match(r"^v?[0-9]+$", x)]
    return len(mixed) * 2 > len(pieces)


def _is_path(value: str) -> bool:
    """A filesystem path, owner/repo, model id or resource name, not a credential: 2+ '/'-segments
    (after an optional '/', '~/', './' or '../' root), every one made of ordinary words, ids or
    sizes, at least one real word, fewer than half the pieces mixing letters and digits, and no
    12+ char mixed-case-and-digit run anywhere. Rooted values get no special treatment:
    '~/Xk9mQ2pL7vR4tN8w/B3zH' fails on its first segment like a relative path would.
    Base64 secrets that merely contain '/' (AWS-style) have mixed-case/digit segments."""
    if "/" not in value or _has_random_run(value):
        return False
    segs = [x for x in _PATH_ROOT.sub("", value).split("/") if x]
    return (len(segs) >= 2 and all(_is_path_segment(x) for x in segs)
            and not _path_pieces_look_random(segs))


def _is_commit_hash(value: str, scheme_word: str) -> bool:
    """A 7-12 or 40 char hex string is also the shape of a hex API key, so it is exempt only when
    the scheme word right after the separator says it is a hash ("auth: commit 1470bfd…")."""
    return _HEX_HASH.match(value) is not None and scheme_word.strip().lower() in _COMMIT_WORDS


def _keyword_value_is_secret(value: str, scheme_word: str = "") -> bool:
    return (len(value) >= KEYWORD_MIN_VALUE_CHARS
            and re.search(r"[A-Za-z]", value) is not None
            and re.search(r"[0-9]", value) is not None
            and not _is_slug(value)
            and not _is_path(value)
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
