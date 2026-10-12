"""Deterministic recall eval: Recall@k / MRR by document id, with a cross-owner
negative assertion (any cross-owner private hit = suite failure). Idempotent: wipes
and rebuilds its own corpus slice each run (owner prefix ame_eval_).
Run: AMES_ES_URL=... AMES_EMBED_BACKEND=es python tests/eval_recall.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory
from app.store import es, idx

OWNERS = ("ame_eval_e1", "ame_eval_e2", "ame_eval_e3")

CORPUS = [
    # (owner_idx, text, query)
    (0, "Patryk runs Hermes on a local workstation and executes builds on the remote build-host", "where do builds run"),
    (0, "Evals always run on Azure VMs via suite_runner.py, never the local workstation", "how are eval suites executed"),
    (0, "OmniRoute heap watchdog warns at 9450MB and restarts at 10200MB", "when does the gateway restart"),
    (0, "Kibana PR author email must be dev@example.com for the CLA", "what email for kibana commits"),
    (0, "Never force-push PR branches; rebase locally then regular push", "how to handle diverged PR branch"),
    (0, "Draft PRs do not auto-build; the /ci comment is the only Buildkite trigger", "how to trigger kibana PR CI"),
    (0, "The Kibana dev server always boots with --no-base-path", "kibana boot flag"),
    (0, "Jest self-computes maxWorkers; never override it per-worktree", "jest worker config rule"),
    (0, "Scout ES scores live on port 9220 with traces cached ES-direct", "where are eval scores stored"),
    (0, "deslop gate runs on the whole branch diff vs upstream main before push", "when does deslop run"),
    (1, "The dogfood stack is live at https://demo.example.com on build-host port 5621", "where is the dogfood stack deployed"),
    (1, "Hindsight daemon runs in docker on build-host port 8888 with a Postgres bank", "how is hindsight deployed"),
    (1, "ssh heredocs eat single quotes; ship scripts as files instead", "ssh script quoting trap"),
    (1, "Never rsync over a running script; ship to tmp then mv into place", "safe way to update a remote script"),
    (1, "PIPESTATUS is a bash-ism; zsh shells return empty for it", "why does pipestatus fail on zsh"),
    (1, "gh at /opt/homebrew/bin/gh is off the default PATH on build-host", "gh not found on build-host"),
    (1, "A VPN connects the local workstation to build-host", "how are the two hosts networked"),
    (1, "Docker mutation tests never run against the real daemon after the volume incident", "docker testing safety rule"),
    (1, "Disk reclaim on build-host is automated via a scheduled cron job", "what reclaims disk space"),
    (1, "The Golden eval index stores scores in evaluator.score with experiment_name colon format", "golden index score field"),
    (2, "Agent Builder skills live in the security_solution plugin scope", "where are agent builder skills"),
    (2, "AlertZero watches are wired into Kibana via the PND integration", "how does alertzero connect"),
    (2, "EIS connectors cache under ~/.elastic/connectors-cache.json", "eis connector cache location"),
    (2, "The judge model for evals is always gemini-3-1-pro", "which model judges evals"),
    (2, "kbn-evals conversations die at 300s headersTimeout on plain-http Kibana clients", "slow model timeout cause"),
    (2, "Elastic Defend fleet package is named endpoint not elastic_defend", "defend package name"),
    (2, "The Workflows plugin tools take one positional params dict plus injected task_id", "workflow tool call shape"),
    (2, "Regression suites must include a known-positive control before claiming zero counts", "zero-count query rule"),
    (2, "False-green verification requires mutation tests reverting a fix to red", "how to prove a gate is real"),
    (2, "Agent workflow gates on data, never on model judgment", "workflow gating principle"),
]


def wipe():
    for owner in OWNERS:
        es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": owner}}})


def build():
    ids = {}
    for i, text, _ in CORPUS:
        doc = memory.retain(OWNERS[i], "semantic", text)
        ids[text] = doc["_id"]
    return ids


def evaluate(ids, k=5):
    recalls, mrrs = [], []
    for i, text, query in CORPUS:
        owner = OWNERS[i]
        gold = ids[text]
        hits = memory.recall(owner, query, size=k)["fused"]
        rank = next((j + 1 for j, h in enumerate(hits) if h["id"] == gold), None)
        recalls.append(1 if rank else 0)
        mrrs.append(1.0 / rank if rank else 0.0)
        for h in hits:
            if h["visibility"] == "private" and h["owner_id"] != owner:
                raise SystemExit(f"CROSS-OWNER LEAK: {owner} saw {h['owner_id']}: {h['text'][:60]}")
    n = len(CORPUS)
    return sum(recalls) / n, sum(mrrs) / n, n


if __name__ == "__main__":
    wipe()
    ids = build()
    r, m, n = evaluate(ids)
    backend = os.environ.get("AMES_EMBED_BACKEND", "auto")
    print(f"backend={backend} n={n} Recall@5={r:.3f} MRR={m:.3f}")
