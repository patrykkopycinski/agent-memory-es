#!/usr/bin/env python3
"""One-shot cleanup: remove duplicate status=draft mental models.

The pre-2026-10-06 worker re-created the same draft every
AMES_CONSOLIDATE_INTERVAL because propose_draft() was not idempotent
(_already_covered only counted active models). This script collapses each
(owner_id, question_pattern) group of drafts to its NEWEST member
(updated_at desc, tie-break: newest _seq_no as proxy for insert order).

Dry-run by default; pass --apply to actually delete. Honors
AMES_ES_URL / AMES_INDEX_PREFIX like the rest of the app.

Usage:
  python3 scripts/dedupe_model_drafts.py            # dry-run, prints counts
  python3 scripts/dedupe_model_drafts.py --apply    # delete duplicates
"""
import argparse
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.store import es, PREFIX, _es_url  # noqa: E402

INDEX = f"{PREFIX}am_models"


def fetch_drafts(size: int = 1000) -> list:
    """All drafts with their metadata, newest first."""
    r = es("POST", f"/{INDEX}/_search", {
        "size": size,
        "query": {"term": {"status": "draft"}},
        "sort": [{"updated_at": {"order": "desc"}},
                 {"_seq_no": {"order": "desc"}}],
        "_source": ["owner_id", "question_pattern", "updated_at"],
    })
    return r["hits"]["hits"]


def plan_dedupe(hits: list) -> dict:
    """Group by (owner_id, question_pattern); keep newest, rest to delete."""
    groups: dict = {}
    for h in hits:
        s = h["_source"]
        key = (s["owner_id"], s["question_pattern"])
        groups.setdefault(key, []).append(h)
    to_delete = []
    for key, members in groups.items():
        # members are newest-first (sort above); keep the first
        to_delete.extend(m["_id"] for m in members[1:])
    return {"groups": {k: len(v) for k, v in groups.items()},
            "total_drafts": len(hits),
            "keep": len(groups),
            "delete": to_delete}


def apply_deletes(ids: list) -> int:
    if not ids:
        return 0
    body = {"query": {"ids": {"values": ids}}}
    r = es("POST", f"/{INDEX}/_delete_by_query?refresh=true&conflicts=proceed", body)
    return r.get("deleted", 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete duplicates (default: dry-run)")
    ap.add_argument("--owner", default=None,
                    help="restrict to one owner_id (default: all)")
    args = ap.parse_args()

    hits = fetch_drafts()
    if args.owner:
        hits = [h for h in hits if h["_source"]["owner_id"] == args.owner]
    p = plan_dedupe(hits)

    print(f"ES: {_es_url()}  index: {INDEX}")
    print(f"drafts scanned: {p['total_drafts']}  distinct (owner, pattern): {p['keep']}  "
          f"duplicates: {len(p['delete'])}")
    for (owner, pattern), n in sorted(p["groups"].items()):
        if n > 1:
            print(f"  {owner!r} / {pattern!r}: {n} drafts -> keep 1, delete {n - 1}")

    if not args.apply:
        print("DRY-RUN: no changes made. Re-run with --apply to delete "
              f"{len(p['delete'])} duplicates.")
        return 0

    n = apply_deletes(p["delete"])
    print(f"APPLIED: deleted {n} drafts")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as e:
        print(f"ES unreachable: {e}", file=sys.stderr)
        sys.exit(2)
