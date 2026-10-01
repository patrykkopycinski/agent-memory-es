"""Consolidate hardening: dry_run default + explicit dispositions + supersede owner check.
dry_run=True (default) returns proposed pairs without mutating. apply=True performs them.
keep_both is a first-class disposition. Supersede asserts same-owner on both ends."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory

# dry_run default: propose only
p1 = memory.retain("c1", "semantic", "Vue is the frontend framework Patryk uses now")
p2 = memory.retain("c1", "semantic", "Vue is the frontend framework Patryk uses now")
plan = memory.consolidate("c1", dry_run=True)
assert plan["mutated"] == 0 and plan["proposals"], plan
assert all(p["disposition"] in ("supersede", "keep_both") for p in plan["proposals"]), plan
# apply
done = memory.consolidate("c1", dry_run=False)
assert done["superseded"] >= 1, done
# cross-owner supersede refused
a = memory.retain("c1", "semantic", "owner-c1 only fact for supersede guard")
b = memory.retain("c2", "semantic", "owner-c2 near duplicate for supersede guard")
try:
    memory._supersede("c2", a["_id"], b["_id"])
    raise AssertionError("cross-owner supersede should raise")
except PermissionError:
    pass
print("CONSOLIDATE-HARDENING: PASS", done)
