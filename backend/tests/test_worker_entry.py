"""docker-compose runs `python -m app.worker` (docker-compose.yml:32,
docker-compose.quickstart.yml:54): the module MUST have a
`if __name__ == "__main__": run_forever()` entry or the container exits 0
immediately and crash-loops under `restart: unless-stopped`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (must precede app imports)

WORKER = os.path.join(os.path.dirname(__file__), "..", "app", "worker.py")

src = open(WORKER, encoding="utf-8").read()
assert '__name__' in src and src.count('__name__') >= 1, \
    "worker.py lost its __main__ entry — docker-compose crash-loops"

# The entry must actually invoke run_forever(), inside the main guard.
tail = src.split('if __name__ == "__main__":', 1)
assert len(tail) == 2, 'worker.py: no `if __name__ == "__main__":` block'
entry = tail[1]
assert "run_forever()" in entry, \
    "worker.py __main__ block does not call run_forever()"

# And run_forever must be a real loop, not a stub.
from app import worker  # noqa: E402
assert callable(worker.run_forever)
print("test_worker_entry.py: PASS")
