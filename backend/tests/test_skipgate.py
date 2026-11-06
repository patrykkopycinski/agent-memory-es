"""Runs scripts/skipgate.py's offline fixture tests (no ES, no host) so every full runner pass covers them."""
import os
import subprocess
import sys

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "skipgate.py")


def test_skipgate_self_test_passes():
    r = subprocess.run([sys.executable, SCRIPT, "--self-test"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SELF-TEST OK" in r.stdout
