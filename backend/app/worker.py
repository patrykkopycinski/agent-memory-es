"""Consolidation worker loop: periodic dedup/supersede + mental-model draft
proposals (never auto-promoted) across all owners."""
import logging
import os
import time

from . import auth, embeddings, memory, mental_models as mm
from .store import KINDS, es, idx

log = logging.getLogger(__name__)

# Max vectorless docs the re-embed sweep repairs per tick (oldest first).
REEMBED_BATCH = int(os.environ.get("AMES_REEMBED_BATCH", "50"))

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
    keys = auth.load_readonly()  # the key volume is mounted :ro here; never migrate-write
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


def reembed_sweep(batch: int = 0) -> dict:
    """Embed active docs that were stored without a vector (the embedder was down or still
    loading at write time) and update them in place. Bounded: at most `batch` docs per call,
    oldest first. Skipped while the embedder is degraded or loading. Each update carries
    if_seq_no/if_primary_term, so a doc edited or deleted since the search is skipped (409)."""
    batch = batch or REEMBED_BATCH
    try:
        embeddings.probe()
    except Exception as exc:  # ModelLoading, ModelUnavailable, a dead backend: all mean "not now"
        return {"skipped": embeddings.describe_error(exc), "embedded": 0, "conflicts": 0}
    found = []
    for kind in KINDS:
        r = es("POST", f"/{idx(kind)}/_search", {
            "size": batch, "seq_no_primary_term": True, "_source": ["text", "occurred_at"],
            "sort": [{"occurred_at": "asc"}],
            "query": {"bool": {"filter": [{"term": {"active": True}}],
                               "must_not": [{"exists": {"field": "embedding"}}]}}})
        for h in r["hits"]["hits"]:
            if h["_source"].get("text"):
                found.append((h["_source"].get("occurred_at") or "", kind, h))
    found.sort(key=lambda t: t[0])
    found = found[:batch]
    result = {"found": len(found), "embedded": 0, "conflicts": 0}
    if not found:
        return result
    vectors = embeddings.embed([h["_source"]["text"] for _, _, h in found])
    for (_, kind, h), vec in zip(found, vectors):
        try:
            es("POST", f"/{idx(kind)}/_update/{h['_id']}"
                       f"?if_seq_no={h['_seq_no']}&if_primary_term={h['_primary_term']}",
               {"doc": {"embedding": vec}})
            result["embedded"] += 1
        except RuntimeError as exc:
            if "-> 409:" not in str(exc):
                raise
            result["conflicts"] += 1
    log.info("reembed sweep: %s", result)
    return result


def run_forever(interval_s: int = 0):
    interval_s = interval_s or int(os.environ.get("AMES_CONSOLIDATE_INTERVAL", "600"))
    while True:
        try:
            print(time.strftime("%H:%M:%S"), run_once(), flush=True)
        except Exception as e:  # noqa: BLE001 — worker must never die
            print(time.strftime("%H:%M:%S"), "worker error:", e, flush=True)
        try:
            print(time.strftime("%H:%M:%S"), "reembed", reembed_sweep(), flush=True)
        except Exception as e:  # noqa: BLE001 — a failed sweep must not stop consolidation
            print(time.strftime("%H:%M:%S"), "reembed error:", e, flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    # docker-compose.yml / docker-compose.quickstart.yml run
    # `python -m app.worker`: without this entry the module imports, exits 0,
    # and the container crash-loops under `restart: unless-stopped`.
    import sys
    if "--backfill" in sys.argv:
        from .facts import backfill_loop
        backfill_loop()
    elif "--facts" in sys.argv:
        from .facts import loop
        scope_owner = os.environ.get("AMES_FACT_OWNER")
        loop(scope_owner)
    else:
        run_forever()
