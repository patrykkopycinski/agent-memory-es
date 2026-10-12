"""OpenAPPA-style write-guard spike (decision D3): deterministic pre-promotion check.

Scopes a simple sensitivity contract — private markers that must never reach a shared
visibility — as code, not a classifier. This is a SPIKE: measures block/false-block rates
on a seeded corpus to decide adopt/reject of a real APPA integration (repo is Preview/RFC).
"""
import re

# deterministic markers: if a private memory contains one, promotion to shared is blocked
PRIVATE_MARKERS = re.compile(
    r"(?i)\b(api[_ -]?key|token|password|passwd|secret|credential|ssh[_ -]?key|"
    r"private[_ -]?key|admin[_ -]?token)\b"
)


def guard_promotion(text: str, to_visibility: str) -> tuple:
    """Returns (allowed: bool, reason: str). Deterministic: same text, same decision."""
    if to_visibility in ("team", "common") and PRIVATE_MARKERS.search(text):
        return False, "sensitive marker present in private memory; refusing shared promotion"
    return True, "ok"


# Seeded corpus for the adopt/reject gate: >=95% block of true leaks, <10% false blocks
SHOULD_BLOCK = [
    "OmniRoute admin token rotates every 14 days",
    "The api_key for the gateway is in the env store",
    "Patryk's password vault code is 8841",
    "The deploy ssh key lives in the agent home",
    "Elastic credentials: user elastic pwd changeme-secret",
]
SHOULD_ALLOW = [
    "Team convention: evals run on Azure VMs via suite_runner.py",
    "Kibana dev boots with --no-base-path always",
    "Never force-push PR branches",
    "The deslop gate runs before every push",
    "Jest self-computes maxWorkers",
]


def measure():
    blocked = sum(1 for t in SHOULD_BLOCK if not guard_promotion(t, "common")[0])
    leaked = len(SHOULD_BLOCK) - blocked
    false_blocks = sum(1 for t in SHOULD_ALLOW if not guard_promotion(t, "common")[0])
    n, m = len(SHOULD_BLOCK), len(SHOULD_ALLOW)
    return {"leak_block_rate": blocked / n, "leaks": leaked,
            "false_block_rate": false_blocks / m, "false_blocks": false_blocks}


if __name__ == "__main__":
    stats = measure()
    print(stats)
    verdict = "ADOPT" if stats["leak_block_rate"] >= 0.95 and stats["false_block_rate"] < 0.10 else "REJECT"
    print(f"OPENAPPA-SPIKE VERDICT: {verdict}")
    assert verdict == "ADOPT", stats
    print("OPENAPPA SPIKE: PASS")
