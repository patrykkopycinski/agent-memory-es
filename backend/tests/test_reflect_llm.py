"""Reflect LLM test against OmniRoute chat (needs tunnel 20128)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory

memory.retain("r1", "semantic", "The m1max host runs the VP dogfood stack on port 5621 and Hindsight on 8888")
memory.retain("r1", "semantic", "Local Mac executes interactive sessions; all builds run remotely on m1max")

r = memory.reflect("r1", "Which services run on m1max and on which ports?")
assert "5621" in r["answer"] and "8888" in r["answer"], r
assert r["synthesized"] and r["sources"], r

r2 = memory.reflect("r1", "What is the capital of France?")
assert r2["answer"].strip().startswith("INSUFFICIENT_EVIDENCE"), r2
print("REFLECT-LLM: PASS")
