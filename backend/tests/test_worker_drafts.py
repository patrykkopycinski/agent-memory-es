"""Worker draft proposals: flag gating, exact-only cover, DF cutoff, draft-only
counting. (Ported from fix/model-draft-dedup-and-flag, adapted to main's
_find_existing_draft in-place-update idempotency.)

Fixtures need filler facts: the DF cutoff (AMES_DRAFT_MAX_ENTITY_DF=0.3)
correctly refuses entities that appear on most of the owner's facts, so a
cluster entity must stay under 30% doc frequency to be proposed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, mental_models as mm, auth
from app.store import es, idx, PREFIX
from app.worker import _candidate_clusters, run_once

O = "drafttest"
O2 = "draftcov"

RIVENDELL_FACTS = [
    "Rivendell deploys via GitHub Actions on main merge",
    "Rivendell uses conventional commits scope config",
    "Rivendell release workflow tags from CHANGELOG.md",
    "Rivendell staging environment runs on Fly.io",
]


def _wipe(owner):
    es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true",
       {"query": {"term": {"owner_id": owner}}})
    mm.ensure_models_index()
    es("POST", f"/{PREFIX}am_models/_delete_by_query?refresh=true",
       {"query": {"term": {"owner_id": owner}}})


FILLER_FACTS = [
    "Kafka consumer lag alerts page the on-call rotation",
    "The design system uses tokens for spacing and color",
    "Quarterly planning happens in a shared roadmap doc",
    "Postgres backups are verified by restore drills",
    "Onboarding includes a pairing week with a buddy",
    "Feature flags roll out gradually by cohort",
    "The CI pipeline caches pnpm stores between builds",
    "Incident reviews are blameless and written up",
    "Mobile releases go through a staged rollout",
    "The style guide bans magic numbers in layouts",
    "Service meshes route internal traffic with mTLS",
    "Data quality dashboards track freshness SLAs",
    "Hiring loops include a structured rubric",
    "Documentation lives next to the code it describes",
    "Chaos drills run monthly in the staging cluster",
    "Cost reports break down spend per team",
    "Accessibility audits gate every major release",
    "The API versioning policy forbids breaking changes",
    "Runbooks are rehearsed before game days",
    "Logs are structured JSON with trace ids",
]


def _filler(owner, n=20):
    for t in FILLER_FACTS[:n]:
        memory.retain(owner, "semantic", t)


def _seed(owner, ents_n=4, filler=20):
    _wipe(owner)
    for t in RIVENDELL_FACTS[:ents_n]:
        memory.retain(owner, "semantic", t)
    _filler(owner, filler)


# --- flag off → run_once proposes zero model drafts -------------------------
auth.create_key(O)
_seed(O)
os.environ.pop("AMES_MODEL_DRAFTS", None)
stats = run_once()
assert O in stats, stats
assert stats[O]["model_drafts"] == 0, stats[O]
print("FLAG-OFF-ZERO-DRAFTS: PASS")

# --- flag on → draft proposed, idempotent (in-place update, no dupes) ------
os.environ["AMES_MODEL_DRAFTS"] = "1"
d1 = _candidate_clusters(O, min_facts=3)
assert d1, "no draft proposed with flag on"
d = [x for x in d1 if "rivendell" in x["question_pattern"]]
assert d, [x["question_pattern"] for x in d1]
assert d[0]["status"] == "draft" and d[0]["source_ids"], d[0]
print("DRAFT-PROPOSED: PASS", d[0]["question_pattern"])

_candidate_clusters(O, min_facts=3)
cnt = es("POST", f"/{PREFIX}am_models/_count",
         {"query": {"bool": {"filter": [
             {"term": {"owner_id": O}},
             {"match_phrase": {"question_pattern": "rivendell"}}]}}})["count"]
assert cnt == 1, f"expected exactly 1 rivendell draft, got {cnt}"
print("DRAFT-IDEMPOTENT: PASS")

# --- draft must NOT surface in recall priority tier -------------------------
res = memory.recall(O, "rivendell facts and conventions")
assert "mental_model" not in res, res.get("mental_model")
print("DRAFT-NOT-IN-TIER: PASS")

# --- list_drafts surfaces it for review -------------------------------------
assert any("rivendell" in x["question_pattern"] for x in mm.list_drafts(O))
print("DRAFT-LISTED: PASS")

# --- exact-only cover: 'Vellum' active model must NOT cover 'rivendell' ----
# (old BM25 match_model covered everything via the shared
#  'facts and conventions' tail)
_wipe(O)
for t in RIVENDELL_FACTS:
    memory.retain(O, "semantic", t)
for t in ["Vellum indexes memory into elasticsearch shards",
          "Vellum compaction runs nightly on the primary node",
          "Vellum access control is enforced at the gateway",
          "Vellum backups land in cold storage daily"]:
    memory.retain(O, "semantic", t)
_filler(O)
mm.upsert_model(O, "vellum facts and conventions", "Vellum summary.")
pats = [x["question_pattern"] for x in _candidate_clusters(O, min_facts=3)]
assert "rivendell facts and conventions" in pats, pats
assert "vellum facts and conventions" not in pats, pats
print("COVER-EXACT-ONLY: PASS")

# --- normalization: case/whitespace variant of an active model covers -------
mm.upsert_model(O, "  Rivendell   Facts AND Conventions ", "normalized cover.")
pats = [x["question_pattern"] for x in _candidate_clusters(O, min_facts=3)]
assert "rivendell facts and conventions" not in pats, pats
print("COVER-NORMALIZED: PASS")

# --- superset pattern is NOT covered by a subset active model ---------------
# active 'rivendell facts' must not cover 'rivendell facts and conventions'
# (exact normalized equality only)
_seed(O)
mm.upsert_model(O, "rivendell facts", "subset pattern.")
pats = [x["question_pattern"] for x in _candidate_clusters(O, min_facts=3)]
assert "rivendell facts and conventions" in pats, pats
print("COVER-SUPERSET-NOT-COVERED: PASS")

# --- shared (other-owner) active model does not cover ----------------------
auth.create_key(O2)
_seed(O)
mm.upsert_model(O2, "rivendell facts and conventions", "someone else's model.")
pats = [x["question_pattern"] for x in _candidate_clusters(O, min_facts=3)]
assert "rivendell facts and conventions" in pats, pats
print("COVER-OWN-MODELS-ONLY: PASS")

# --- DF cutoff: entity on 100% of facts → no draft -------------------------
_wipe(O)
_filler(O, n=5)  # only 'filler/topic' entities, DF = 1.0
d = _candidate_clusters(O, min_facts=3)
assert not d, d
print("DF-100PCT-NO-DRAFT: PASS")

# --- DF cutoff: 3-of-20 facts → draft still proposed -----------------------
_wipe(O)
for t in RIVENDELL_FACTS[:3]:
    memory.retain(O, "semantic", t)
_filler(O, n=17)
pats = [x["question_pattern"] for x in _candidate_clusters(O, min_facts=3)]
assert "rivendell facts and conventions" in pats, pats
assert not any("facts and conventions" in p and "rivendell" not in p for p in pats), pats  # every other entity <3 docs or >30% DF with dedup residue
print("DF-3OF20-DRAFT: PASS")

# --- denylist entity never clusters ----------------------------------------
_wipe(O)
for i in range(4):
    memory.retain(O, "semantic", f"memory fact {i}")
pats = [x["question_pattern"] for x in _candidate_clusters(O, min_facts=3)]
assert "memory facts and conventions" not in pats, pats
print("DENYLIST: PASS")

# --- run_once counts only status=='draft' returns --------------------------
_seed(O)


class _FakeMM:
    def propose_draft(self, *a, **k):
        return {"status": "active"}  # pretend promotion raced us


_orig = mm.propose_draft
mm.propose_draft = _FakeMM().propose_draft
try:
    stats = run_once()
finally:
    mm.propose_draft = _orig
assert stats[O]["model_drafts"] == 0, stats[O]
print("ACTIVE-RETURN-NOT-COUNTED: PASS")

# --- manual promotion → now in tier ----------------------------------------
_seed(O)
_candidate_clusters(O, min_facts=3)
mm.upsert_model(O, "rivendell facts and conventions",
                "Rivendell: GH Actions deploys, conventional commits, CHANGELOG tags, Fly.io staging.")
res = memory.recall(O, "rivendell facts and conventions")
assert "mental_model" in res, res.get("mental_model")
print("PROMOTED-IN-TIER: PASS")

_wipe(O)
_wipe(O2)
print("test_worker_drafts.py: ALL PASS")
