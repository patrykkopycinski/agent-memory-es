"""Reflect LLM test against OmniRoute chat (needs tunnel 20128)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import memory
from app.store import es, idx

# owner-scoped clean slate: the throwaway cluster is shared across modules,
# and the abstention assert below must not see leftover r1 facts
es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true",
   {"query": {"term": {"owner_id": "r1"}}})

memory.retain("r1", "semantic", "The build-host runs the dogfood stack on port 5621 and Hindsight on 8888")
memory.retain("r1", "semantic", "The local workstation executes interactive sessions; all builds run remotely on build-host")

r = memory.reflect("r1", "Which services run on build-host and on which ports?")
assert "5621" in r["answer"] and "8888" in r["answer"], r
assert r["synthesized"] and r["sources"], r

r2 = memory.reflect("r1", "What is the capital of France?")
assert r2["answer"].strip().startswith("INSUFFICIENT_EVIDENCE"), r2
print("REFLECT-LLM: PASS")
