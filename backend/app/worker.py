"""Consolidation worker loop: periodic dedup/supersede across all owners."""
import os
import time

from . import auth, memory


def known_owners() -> list:
    keys = auth._load()
    return sorted({v["owner_id"] for v in keys.values()})


def run_once() -> dict:
    stats = {}
    for owner in known_owners():
        stats[owner] = memory.consolidate(owner)
    return stats


def run_forever(interval_s: int = 0):
    interval_s = interval_s or int(os.environ.get("AMES_CONSOLIDATE_INTERVAL", "600"))
    while True:
        try:
            print(time.strftime("%H:%M:%S"), run_once(), flush=True)
        except Exception as e:  # noqa: BLE001 — worker must survive
            print("consolidate error:", e, flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    run_forever()
