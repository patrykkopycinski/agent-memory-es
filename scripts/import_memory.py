"""Importers: migrate existing memory corpora into agent-memory-es.

Supported sources:
  - governance  : Diamantis-style per-repo governance stores
                  (profile/decisions-log/design-decisions/changelog/
                   refactoring-log/session-handoff markdown files)
  - hindsight   : JSONL/JSON export of Hindsight memory entries
  - markdown    : plain CLAUDE.md / AGENTS.md / *.md knowledge files

Every importer maps source content onto (kind, visibility, text, occurred_at)
and retains through the normal memory.retain path so embeddings, entity
extraction and tombstone guards all apply.
"""
import argparse
import datetime as _dt
import json
import os
import re
import sys
from typing import Callable, Iterable, Optional

# --- locate the backend app regardless of cwd -------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app import memory  # noqa: E402

# --- governance store parsing ------------------------------------------------

GOV_FILES = {
    "profile.md": ("procedural", "team"),
    "decisions-log.md": ("semantic", "team"),
    "design-decisions.md": ("semantic", "team"),
    "changelog.md": ("procedural", "private"),
    "refactoring-log.md": ("procedural", "private"),
    "session-handoff.md": (None, None),  # ephemeral: skipped by default
}

_DECISION_RE = re.compile(
    r"^#{2,3}\s+(\d{4}-\d{2}-\d{2})\s+(?:--|—|–)\s+(.+?)\s*$", re.MULTILINE
)


def _split_entries(md_text: str) -> list:
    """Split a decisions-log style file into (date, title, body) entries."""
    matches = list(_DECISION_RE.finditer(md_text))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[m.end():end].strip().strip("-").strip()
        out.append((m.group(1), m.group(2).strip(), body))
    return out


def _split_changelog(md_text: str) -> list:
    """Split a dated changelog into (date, body) chunks."""
    day_re = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})", re.MULTILINE)
    matches = list(day_re.finditer(md_text))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        out.append((m.group(1), md_text[m.end():end].strip()))
    return out


def parse_governance_store(store_dir: str) -> Iterable[dict]:
    """Yield retain-ready entries from one governance store directory.

    Mapping (documented in docs/IMPORTERS.md):
      profile.md           -> procedural, team   (repo overview, paths, rules)
      decisions-log.md     -> semantic,  team    (each decision = one entry)
      design-decisions.md  -> semantic,  team    (each decision = one entry)
      refactoring-log.md   -> procedural, private
      changelog.md         -> skipped by default (re-derivable from git);
                              --include-changelog retains per-day summaries
      session-handoff.md   -> skipped (ephemeral session state, not bank material)
    """
    store = os.path.abspath(store_dir)
    slug = os.path.basename(store)
    include_changelog = _FLAGS.get("include_changelog", False)

    for fname, (kind, vis) in GOV_FILES.items():
        path = os.path.join(store, fname)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        if kind is None:  # session-handoff
            continue
        if fname == "profile.md":
            yield {
                "kind": kind, "visibility": vis,
                "text": text, "occurred_at": None,
                "source": f"{slug}/{fname}",
            }
        elif fname in ("decisions-log.md", "design-decisions.md"):
            for date, title, body in _split_entries(text):
                yield {
                    "kind": kind, "visibility": vis,
                    "text": f"{title}\n\n{body}",
                    "occurred_at": f"{date}T00:00:00Z",
                    "source": f"{slug}/{fname}#{date}",
                }
        elif fname == "refactoring-log.md":
            for date, title, body in _split_entries(text):
                yield {
                    "kind": kind, "visibility": vis,
                    "text": f"{title}\n\n{body}",
                    "occurred_at": f"{date}T00:00:00Z",
                    "source": f"{slug}/{fname}#{date}",
                }
        elif fname == "changelog.md" and include_changelog:
            for date, body in _split_changelog(text):
                yield {
                    "kind": kind, "visibility": vis,
                    "text": body[:4000],
                    "occurred_at": f"{date}T00:00:00Z",
                    "source": f"{slug}/{fname}#{date}",
                }
        else:
            if fname == "changelog.md":
                continue


_FLAGS: dict = {}


def parse_hindsight_export(path: str) -> Iterable[dict]:
    """Parse a Hindsight export (JSONL, one entry per line, or a JSON array).

    Fields used (tolerant): text/content/body, kind/type, visibility,
    occurred_at/created_at/timestamp. Unknown kinds default to 'episodic'.
    """
    raw = open(path, encoding="utf-8", errors="replace").read().strip()
    entries = []
    if raw.startswith("["):
        entries = json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    kind_map = {"fact": "semantic", "note": "episodic",
                "procedure": "procedural", "observation": "episodic"}
    for e in entries:
        text = e.get("text") or e.get("content") or e.get("body") or ""
        if not text.strip():
            continue
        raw_kind = (e.get("kind") or e.get("type") or "").lower()
        kind = kind_map.get(raw_kind, "episodic")
        yield {
            "kind": kind,
            "visibility": e.get("visibility", "private"),
            "text": text,
            "occurred_at": e.get("occurred_at") or e.get("created_at") or e.get("timestamp"),
            "source": f"hindsight:{path}",
        }


def parse_markdown(path: str) -> Iterable[dict]:
    """Import a plain markdown knowledge file (CLAUDE.md, AGENTS.md, notes).

    One entry per '##' section; preamble becomes its own entry. Procedural,
    private by default (the file's rules usually describe one machine's setup).
    """
    text = open(path, encoding="utf-8", errors="replace").read()
    sec_re = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(sec_re.finditer(text))
    base = os.path.basename(path)
    if not matches:
        if text.strip():
            yield {"kind": "procedural", "visibility": "private", "text": text,
                   "occurred_at": None, "source": f"markdown:{base}"}
        return
    preamble = text[:matches[0].start()].strip()
    if preamble:
        yield {"kind": "procedural", "visibility": "private", "text": preamble,
               "occurred_at": None, "source": f"markdown:{base}#preamble"}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        if body:
            yield {"kind": "procedural", "visibility": "private", "text": body,
                   "occurred_at": None, "source": f"markdown:{base}#{m.group(1)[:40]}"}


PARSERS: dict[str, Callable[[str], Iterable[dict]]] = {
    "governance": parse_governance_store,
    "hindsight": parse_hindsight_export,
    "markdown": parse_markdown,
}


def _detect(source: str) -> str:
    if os.path.isdir(source) and os.path.exists(os.path.join(source, "profile.md")):
        return "governance"
    if source.endswith(".json") or source.endswith(".jsonl"):
        return "hindsight"
    if source.endswith(".md"):
        return "markdown"
    raise SystemExit(f"cannot detect source type for {source}; pass --format explicitly")


def run(source: str, format: Optional[str], owner: str, visibility: Optional[str],
        include_changelog: bool, dry_run: bool) -> int:
    _FLAGS["include_changelog"] = include_changelog
    fmt = format or _detect(source)
    entries = list(PARSERS[fmt](source))
    print(f"source={source} format={fmt} parsed_entries={len(entries)}")
    if dry_run:
        for e in entries[:5]:
            preview = e["text"][:120].replace("\n", " ")
            print(f"  [dry-run] {e['kind']}/{e['visibility']} {e['source']}: {preview}")
        print(f"  ... {len(entries)} entries total (dry-run, nothing retained)")
        return 0
    ok, failed = 0, 0
    for e in entries:
        try:
            vis = visibility or e["visibility"]
            memory.retain(owner, e["kind"], e["text"], vis, e["occurred_at"])
            ok += 1
        except Exception as ex:  # tombstone rejects, ES down, etc.
            failed += 1
            print(f"  ! {e['source']}: {ex}", file=sys.stderr)
    print(f"retained={ok} failed={failed}")
    return 1 if failed and not ok else 0


def main() -> None:
    p = argparse.ArgumentParser(
        prog="ames import",
        description="Import existing memory corpora into agent-memory-es")
    p.add_argument("source", help="governance store dir | hindsight .json/.jsonl | markdown file")
    p.add_argument("--format", choices=sorted(PARSERS), default=None,
                   help="override source-type detection")
    p.add_argument("--owner", default="local", help="owner_id to retain as (default: local)")
    p.add_argument("--visibility", choices=["private", "team", "common"], default=None,
                   help="override entry visibility (default: per-format mapping)")
    p.add_argument("--include-changelog", action="store_true",
                   help="also import dated changelog summaries (skipped by default)")
    p.add_argument("--dry-run", action="store_true",
                   help="parse and preview, retain nothing")
    a = p.parse_args()
    raise SystemExit(run(a.source, a.format, a.owner, a.visibility,
                         a.include_changelog, a.dry_run))


if __name__ == "__main__":
    main()
