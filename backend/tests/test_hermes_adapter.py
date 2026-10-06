"""Round-trip test: Hermes adapter (in-process) against live ES."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import auth
from app.hermes_provider import HermesMemoryProvider

key = auth.create_key("hermes-default")
p = HermesMemoryProvider("hermes-default", api_key=key)

p.retain("Kibana eval stacks must boot on dedicated per-worktree ports, never 5620 shared")
p.retain("OmniRoute heap watchdog: warn 9450MB restart 10200MB")

hits = p.recall("eval stack port shared")
assert any("5620" in h["text"] for h in hits), hits
answer = p.reflect("what ports must eval stacks use?")
assert "5620" in answer or "dedicated" in answer, answer
# wrong-key guard
try:
    HermesMemoryProvider("hermes-default", api_key="ame_deadbeef")
    raise AssertionError("should have raised")
except PermissionError:
    pass
print("HERMES ADAPTER: PASS")
