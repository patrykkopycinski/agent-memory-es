"""Promotion write-guard test: sensitive private memory must not promote to shared."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory

doc = memory.retain("g1", "semantic", "OmniRoute admin token rotates every 14 days")
try:
    memory.promote("g1", "semantic", doc["_id"], "common")
    raise AssertionError("promotion should have been blocked")
except ValueError as e:
    assert "sensitive marker" in str(e), e
ok = memory.retain("g1", "semantic", "Team convention: evals run via suite_sweep.py on Azure")
doc2 = memory.retain("g1", "semantic", "Team convention: evals run via suite_sweep.py on Azure VMs always")
p = memory.promote("g1", "semantic", doc2["_id"], "common")
assert "promoted_id" in p, p
print("PROMOTION-GUARD: PASS")
