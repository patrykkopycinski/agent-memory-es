"""Worker test: consolidation across two owners, run_once paths."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import auth, memory  # noqa: E402
from app.worker import run_once  # noqa: E402

auth.create_key("w1")
auth.create_key("w2")
memory.retain("w1", "semantic", "Python toolchain for repo gates is python3.12 venv")
memory.retain("w1", "semantic", "Python toolchain for repo gates is python3.12 venv")
memory.retain("w2", "semantic", "Worker two note: git identity uses committer contact@patrykkopycinski.com")

stats = run_once()
assert "w1" in stats and "w2" in stats, stats
assert stats["w1"]["superseded"] >= 1, stats  # duplicate w1 fact superseded
hits = memory.recall("w2", "git identity committer email")["fused"]
assert any("contact@patrykkopycinski.com" in h["text"] for h in hits), hits
print("WORKER: PASS", stats)
