"""Tombstone tests: value-keyed rejection blocks the write path, survives rewording."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory, tombstone

tombstone.reject("t1", "Alice works at Microsoft", "superseded: moved to Google Mar 2026")
# exact re-assert blocked
try:
    memory.retain("t1", "semantic", "Alice works at Microsoft")
    raise AssertionError("write should have been rejected")
except ValueError as e:
    assert "rejected value" in str(e)
# reworded re-assert (same salient terms) blocked
try:
    memory.retain("t1", "semantic", "Microsoft is where Alice works")
    raise AssertionError("reworded write should have been rejected")
except ValueError:
    pass
# different owner unaffected
memory.retain("t2", "semantic", "Alice works at Microsoft")
# different value unaffected
memory.retain("t1", "semantic", "Alice works at Google research team")
print("TOMBSTONE: PASS")
