"""Consolidation worker loop: periodic dedup/supersede + mental-model draft
proposals (never auto-promoted) across all owners."""
import os
import time

from . import auth, memory, mental_models as mm
from .store import es, idx


def known_owners() -> list:
    keys = auth._load()
    # digest entries only — the file also keeps name-keyed plaintext copies
    # (legacy migration) which would blow up v["owner_id"].
    return sorted({v["owner_id"] for v in keys.values()
                   if isinstance(v, dict)})


def _already_covered(owner: str, pattern: str) -> bool:
    """Active model (own or shared) already matching this pattern."""
    try:
        return mm.match_model(owner, pattern) is not None
    except Exception:
        return True  # fail closed: never propose on uncertainty


def _candidate_clusters(owner: str, min_facts: int = 3) -> list:
    """Entity clusters with >= min_facts active semantic facts and no covering
    model → draft proposals. Cluster key = most frequent shared entity."""
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": 0,
        "query": {"bool": {"filter": [{"term": {"owner_id": owner}},
                                       {"term": {"active": True}}]}},
        "aggs": {"ents": {"terms": {"field": "entities", "size": 20,
                                    "min_doc_count": min_facts}}}})
    proposals = []
    for b in r["aggregations"]["ents"]["buckets"]:
        ent = b["key"]
        pattern = f"{ent.replace('_', ' ')} facts and conventions"
        if _already_covered(owner, pattern):
            continue
        fr = es("POST", f"/{idx('semantic')}/_search", {
            "size": 10,
            "query": {"bool": {"filter": [{"term": {"owner_id": owner}},
                                             {"term": {"active": True}},
                                             {"term": {"entities": ent}}]}},
            "sort": [{"occurred_at": {"order": "desc"}}]})
        facts = [{"id": h["_id"], "text": h["_source"]["text"]}
                 for h in fr["hits"]["hits"]]
        if len(facts) < min_facts:
            continue
        summary = f"DRAFT (auto-proposed, {len(facts)} facts about '{ent}'): " + "; ".join(
            f["text"][:80] for f in facts[:3])
        proposals.append(mm.propose_draft(owner, pattern, summary,
                                          [f["id"] for f in facts]))
    return proposals


def run_once() -> dict:
    """Propose then apply: dry_run pass for visibility, apply pass for supersessions
    (keep_both stays untouched — surfaced only). Mental-model drafts proposed per
    owner; promotion is manual (upsert_model via the owner's key)."""
    stats = {}
    for owner in known_owners():
        plan = memory.consolidate(owner, dry_run=True)
        applied = memory.consolidate(owner, dry_run=False)
        try:
            drafts = _candidate_clusters(owner)
        except Exception:
            drafts = []
        stats[owner] = {"proposals": len(plan.get("proposals", [])),
                        "superseded": applied.get("superseded", 0),
                        "model_drafts": len(drafts)}
    return stats


def run_forever(interval_s: int = 0):
    interval_s = interval_s or int(os.environ.get("AMES_CONSOLIDATE_INTERVAL", "600"))
    while True:
        try:
            print(time.strftime("%H:%M:%S"), run_once(), flush=True)
        except Exception as e:  # noqa: BLE001 — worker must survive
            print("consolidate error:", e, flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    import sys
    if "--facts" in sys.argv:
        from .facts import loop
        scope_owner = os.environ.get("AMES_FACT_OWNER")
        loop(scope_owner)
    else:
        run_forever()
