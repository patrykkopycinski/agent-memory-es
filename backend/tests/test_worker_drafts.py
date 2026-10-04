"""Worker draft proposals + staleness: drafts never enter the priority tier."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, mental_models as mm, auth
from app.store import es, idx
from app.worker import _candidate_clusters

O = "drafttest"

# setup: wipe owner's semantic + models, seed 4 facts sharing an entity
es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true",
   {"query": {"term": {"owner_id": O}}})
mm.ensure_models_index()
from app.store import PREFIX
es("POST", f"/{PREFIX}am_models/_delete_by_query?refresh=true",
   {"query": {"term": {"owner_id": O}}})

memory.retain(O, "semantic", "Rivendell deploys via GitHub Actions on main merge")
memory.retain(O, "semantic", "Rivendell uses conventional commits scope config")
memory.retain(O, "semantic", "Rivendell release workflow tags from CHANGELOG.md")
memory.retain(O, "semantic", "Rivendell staging environment runs on Fly.io")

drafts = _candidate_clusters(O, min_facts=3)
assert drafts, "no draft proposed for rivendell cluster"
d = [x for x in drafts if "rivendell" in x["question_pattern"]]
assert d, [x["question_pattern"] for x in drafts]
assert d[0]["status"] == "draft" and d[0]["source_ids"], d[0]
print("DRAFT-PROPOSED: PASS", d[0]["question_pattern"])

# draft must NOT surface in recall priority tier
res = memory.recall(O, "rivendell facts and conventions")
assert "mental_model" not in res, res.get("mental_model")
print("DRAFT-NOT-IN-TIER: PASS")

# list_drafts surfaces it for review
assert any("rivendell" in x["question_pattern"] for x in mm.list_drafts(O))
print("DRAFT-LISTED: PASS")

# manual promotion → now in tier
mm.upsert_model(O, "rivendell facts and conventions",
                "Rivendell: GH Actions deploys, conventional commits, CHANGELOG tags, Fly.io staging.")
res = memory.recall(O, "rivendell facts and conventions")
assert "mental_model" in res and res["mental_model"]["status"] == "active", res.get("mental_model")
print("PROMOTE-IN-TIER: PASS")

# staleness: new fact AFTER model update → stale flag.
# The fact must be genuinely new: a near-duplicate would be collapsed by
# dedup-on-write (cos >= 0.92) and never land, so staleness never fires.
import time as _t
_t.sleep(1.1)
memory.retain(O, "semantic", "Rivendell now ships release artifacts through a Railway-hosted runner fleet")
res = memory.recall(O, "rivendell facts and conventions")
assert res["mental_model"].get("stale") is True, res["mental_model"]
print("STALENESS: PASS")

print("WORKER-DRAFTS: ALL PASS")
