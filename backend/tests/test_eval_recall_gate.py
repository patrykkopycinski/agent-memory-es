"""pytest wrapper for the deterministic recall eval (tests/eval_recall.py),
so the remote runner can gate it: runs the corpus build + Recall@5/MRR +
the pre-registered graph on/off axis (Cell A). Direct script use unchanged:
    python tests/eval_recall.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "eval_recall.py")


def test_eval_recall_with_graph_axis():
    # -u: unbuffered so the axis lines reach the captured log even on failure
    r = subprocess.run([sys.executable, "-u", SCRIPT], capture_output=True,
                       text=True, timeout=600, cwd=HERE)
    print(r.stdout)
    assert r.returncode == 0, (
        f"eval_recall exited {r.returncode}:\n{r.stdout}\n{r.stderr}")
