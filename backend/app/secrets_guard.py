"""Write-time secret guard for retain: text that looks like a credential is rejected (HTTP 422).

The patterns are a Python port of KEYWORD_RE and SHAPE_RES in
opencode-harness/plugin/autolearn/redact.ts. Keep the two in sync: a pattern added there
should be added here, or autolearn redacts what the service would still accept.
"""
import re
from typing import Optional

# key-ish word, separator, optional scheme word, 8+ char value (redact.ts KEYWORD_RE).
# Deliberately stricter than redact.ts, which only masks: a 422 on prose like "auth migration"
# or "token estimator" would reject ordinary facts. Here the separator must contain ':' or '='
# and the value must mix letters and digits.
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
    ("shape_pem_header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


def find_secret(text: Optional[str]) -> Optional[str]:
    """Name of the first secret pattern `text` matches, or None. Never returns the match."""
    if not text:
        return None
    for name, rx in SHAPE_RES:
        if rx.search(text):
            return name
    for m in KEYWORD_RE.finditer(text):
        value = m.group(4)
        if re.search(r"[A-Za-z]", value) and re.search(r"[0-9]", value):
            return "keyword_value"
    return None
