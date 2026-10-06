"""Regression (2026-10-06 duplicate-draft bug): _candidate_clusters run
twice must yield exactly ONE draft per (owner, pattern); changed facts must
update that draft in place, not create a second one.

Root cause: worker._already_covered() -> mm.match_model() filters
status=active only, so existing drafts never counted as coverage and
propose_draft() POSTed a new doc every AMES_CONSOLIDATE_INTERVAL (600s).
Idempotency now lives in mm.propose_draft() itself (single write path).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script path guard; conftest imports it too)

from app import memory, mental_models as mm, auth
from app.store import es, idx, PREFIX
from app.worker import _candidate_clusters, run_once  # noqa: F401

O = "draftidem"


def _wipe(owner):
    es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true",
       {"query": {"term": {"owner_id": owner}}})
    mm.ensure_models_index()
    es("POST", f"/{PREFIX}am_models/_delete_by_query?refresh=true",
       {"query": {"term": {"owner_id": owner}}})


def _drafts(substr=""):
    return [d for d in mm.list_drafts(O) if substr in d["question_pattern"]]


# --- setup: clean owner, seed a rivendell cluster ---
_wipe(O)
# The owner must exist in the key store for run_once() to report it at all.
auth.create_key(O)
# Filler facts keep rivendell below the AMES_DRAFT_MAX_ENTITY_DF (0.3)
# cutoff: with only the cluster facts, rivendell DF=1.0 and is (correctly)
# skipped as noise. Idempotency is what this file tests, not clustering.
for t in ["Kafka consumer lag alerts page the on-call rotation",
          "The design system uses tokens for spacing and color",
          "Quarterly planning happens in a shared roadmap doc",
          "Postgres backups are verified by restore drills",
          "Onboarding includes a pairing week with a buddy",
          "Feature flags roll out gradually by cohort",
          "The CI pipeline caches pnpm stores between builds",
          "Incident reviews are blameless and written up",
          "Data quality dashboards track freshness SLAs",
          "Runbooks are rehearsed before game days"]:
    memory.retain(O, "semantic", t)
memory.retain(O, "semantic", "Rivendell deploys via GitHub Actions on main merge")
memory.retain(O, "semantic", "Rivendell uses conventional commits scope config")
memory.retain(O, "semantic", "Rivendell release workflow tags from CHANGELOG.md")

# --- run the worker path twice -> exactly one draft ---
d1 = _candidate_clusters(O, min_facts=3)
assert any("rivendell" in p["question_pattern"] for p in d1), d1
d2 = _candidate_clusters(O, min_facts=3)
ds = _drafts("rivendell")
assert len(ds) == 1, f"expected 1 draft after 2 passes, got {len(ds)}: {[x['id'] for x in ds]}"
riv_ids = [p["id"] for p in d2 if "rivendell" in p["question_pattern"]]
assert not riv_ids or riv_ids == [ds[0]["id"]], riv_ids
print("IDEMPOTENT-PASS2: PASS", ds[0]["id"], ds[0]["source_ids"])

# --- changed facts (new fact joins the cluster) -> one draft, updated ---
# explicit occurred_at: with local inference the seed retains can land in
# the same wall-clock second, so the occurred_at sort would not place the
# new fact first.
new_id = memory.retain(O, "semantic", "Rivendell staging environment runs on Fly.io",
                       occurred_at="2099-01-01T00:00:00Z")["_id"]
_candidate_clusters(O, min_facts=3)
ds = _drafts("rivendell")
assert len(ds) == 1, f"expected 1 draft after changed facts, got {len(ds)}"
assert new_id in ds[0]["source_ids"], (new_id, ds[0]["source_ids"])
assert len(ds[0]["source_ids"]) >= 4, ds[0]["source_ids"]
print("UPDATE-IN-PLACE: PASS", ds[0]["source_ids"])

# --- run_once twice end-to-end, still no new drafts ---
# run_once only proposes drafts when AMES_MODEL_DRAFTS is enabled — set it
# around the calls (restored even on assertion failure) so this actually
# exercises the worker's draft path and proves run_once reports the owner.
from app import worker as wk
_saved = os.environ.pop("AMES_MODEL_DRAFTS", None)
try:
    os.environ["AMES_MODEL_DRAFTS"] = "1"
    s1 = wk.run_once()
    wk.run_once()
finally:
    if _saved is None:
        os.environ.pop("AMES_MODEL_DRAFTS", None)
    else:
        os.environ["AMES_MODEL_DRAFTS"] = _saved
assert O in s1, f"run_once did not report seeded owner {O!r}: {sorted(s1)}"
assert s1[O].get("model_drafts", 0) >= 1, \
    f"run_once reported no drafts for {O} (flag not effective?): {s1[O]}"
ds = _drafts("rivendell")
assert len(ds) == 1, f"run_once duplicated drafts: {len(ds)}"
print("RUN-ONCE-IDEMPOTENT: PASS")
