"""Consolidation worker loop: periodic dedup/supersede + mental-model draft
proposals (never auto-promoted) across all owners."""
import os
import time

from . import auth, memory, mental_models as mm
from .store import es, idx

# '1'/'true' enable worker model-draft proposals; anything else disables.
def _drafts_enabled() -> bool:
    return os.environ.get("AMES_MODEL_DRAFTS", "").strip().lower() in ("1", "true")

# Entities that appear on nearly every fact (extraction artifacts) never make
# good cluster keys.
_DRAFT_ENTITY_DENYLIST = {"memory", "facts", "conventions"}

# Max document frequency (fraction of the owner's active semantic facts) for a
# cluster-key entity; above it the entity is treated as noise.
_DRAFT_MAX_ENTITY_DF = float(os.environ.get("AMES_DRAFT_MAX_ENTITY_DF", "0.3"))


def known_owners() -> list:
    keys = auth._load()
    # digest entries only — the file also keeps name-keyed plaintext copies
    # (legacy migration) which would blow up v["owner_id"].
    return sorted({v["owner_id"] for v in keys.values()
                   if isinstance(v, dict)})


def _already_covered(owner: str, pattern: str) -> bool:
    """Exact normalized match against the owner's OWN active models.

    An existing DRAFT is not coverage: the pattern must stay in the run so
    propose_draft() refreshes it in place. Deliberately NOT mm.match_model
    (BM25): every generated pattern shares the 'facts and conventions' tail,
    so any visible active model would BM25-match any pattern and suppress
    all proposals. Any lookup error fails CLOSED (never propose on
    uncertainty)."""
    try:
        return bool(mm.find_models(owner, pattern, status="active"))
    except Exception:
        return True  # fail closed: never propose on uncertainty


def _candidate_clusters(owner: str, min_facts: int = 3) -> list:
    """Entity clusters with >= min_facts active semantic facts and no covering
    model → draft proposals. Cluster key = most frequent shared entity.

    Entities whose document frequency exceeds _DRAFT_MAX_ENTITY_DF (of the
    owner's active facts) or that are on the denylist are skipped as noise —
    cheap junk filter, so the terms agg size is generous (200)."""
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": 0,
        "track_total_hits": True,
        "query": {"bool": {"filter": [{"term": {"owner_id": owner}},
                                       {"term": {"active": True}}]}},
        "aggs": {"ents": {"terms": {"field": "entities", "size": 200,
                                    "min_doc_count": min_facts}}}})
    total = r.get("hits", {}).get("total", {}).get("value", 0) or 1
    proposals = []
    for b in r["aggregations"]["ents"]["buckets"]:
        ent = b["key"]
        if ent in _DRAFT_ENTITY_DENYLIST:
            continue
        if _DRAFT_MAX_ENTITY_DF > 0 and b["doc_count"] / total > _DRAFT_MAX_ENTITY_DF:
            continue
        pattern = f"{ent.replace('_', ' ')} facts and conventions"
        if _already_covered(owner, pattern):
            continue  # active model (or lookup error): never propose.
            # NOTE: an existing DRAFT is deliberately not terminal coverage —
            # the entity falls through to propose_draft(), which updates the
            # draft in place (new cluster facts refresh it, no duplicates).
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
        r = mm.propose_draft(owner, pattern, summary,
                             [f["id"] for f in facts])
        proposals.append(r)
    return proposals


def run_once() -> dict:
    """Propose then apply: dry_run pass for visibility, apply pass for supersessions
    (keep_both stays untouched — surfaced only). Mental-model drafts proposed per
    owner; promotion is manual (upsert_model via the owner's key)."""
    stats = {}
    for owner in known_owners():
        plan = memory.consolidate(owner, dry_run=True)
        applied = memory.consolidate(owner, dry_run=False)
        if _drafts_enabled():
            try:
                drafts = _candidate_clusters(owner)
            except Exception:
                drafts = []
        else:
            drafts = []
        # Count only NEW drafts: existing drafts are re-proposed every pass
        # and updated in place (propose_draft returns created=False for
        # them); non-draft returns (promotion raced us) don't count either.
        stats[owner] = {"proposals": len(plan.get("proposals", [])),
                        "superseded": applied.get("superseded", 0),
                        "model_drafts": sum(
                            1 for d in drafts
                            if d.get("status") == "draft" and d.get("created"))}
    return stats


def run_forever(interval_s: int = 0):
    interval_s = interval_s or int(os.environ.get("AMES_CONSOLIDATE_INTERVAL", "600"))
    while True:
        try:
            print(time.strftime("%H:%M:%S"), run_once(), flush=True)
        except Exception as e:  # noqa: BLE001 — worker must never die
            print(time.strftime("%H:%M:%S"), "worker error:", e, flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    # docker-compose.yml / docker-compose.quickstart.yml run
    # `python -m app.worker`: without this entry the module imports, exits 0,
    # and the container crash-loops under `restart: unless-stopped`.
    run_forever()
