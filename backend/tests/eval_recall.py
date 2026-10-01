"""Deterministic recall eval: Recall@k / MRR by document id, with a cross-owner
negative assertion (any cross-owner private hit = suite failure). Corpus: 30 facts
across 3 owners with paraphrased queries. Run:
AMES_ES_URL=... AMES_EMBED_BACKEND=es python tests/eval_recall.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory

CORPUS = [
    # (owner, text, query)
    ("e1", "Patryk runs Hermes on a local Mac and executes builds on the remote m1max host", "where do builds run"),
    ("e1", "Evals always run on Azure VMs via suite_sweep.py, never the local Mac", "how are eval suites executed"),
    ("e1", "OmniRoute heap watchdog warns at 9450MB and restarts at 10200MB", "when does the gateway restart"),
    ("e1", "Kibana PR author email must be patryk.kopycinski@elastic.co for the CLA", "what email for kibana commits"),
    ("e1", "Never force-push PR branches; rebase locally then regular push", "how to handle diverged PR branch"),
    ("e1", "Draft PRs do not auto-build; the /ci comment is the only Buildkite trigger", "how to trigger kibana PR CI"),
    ("e1", "The Kibana dev server always boots with --no-base-path", "kibana boot flag"),
    ("e1", "Jest self-computes maxWorkers; never override it per-worktree", "jest worker config rule"),
    ("e1", "Scout ES scores live on port 9220 with traces cached ES-direct", "where are eval scores stored"),
    ("e1", "deslop gate runs on the whole branch diff vs upstream main before push", "when does deslop run"),
    ("e2", "The VP dogfood stack is live at https://vp.widzimysie.pl on m1max port 5621", "where is the vp stack deployed"),
    ("e2", "Hindsight daemon runs in docker on m1max port 8888 with a Postgres bank", "how is hindsight deployed"),
    ("e2", "ssh heredocs eat single quotes; ship scripts as files instead", "ssh script quoting trap"),
    ("e2", "Never rsync over a running script; ship to tmp then mv into place", "safe way to update a remote script"),
    ("e2", "PIPESTATUS is a bash-ism; zsh shells return empty for it", "why does pipestatus fail on zsh"),
    ("e2", "gh at /opt/homebrew/bin/gh is off the default PATH on m1max", "gh not found on m1max"),
    ("e2", "Tailscale tailnet connects the Mac workstation to m1max", "how are the two hosts networked"),
    ("e2", "Docker mutation tests never run against the real daemon after the volume incident", "docker testing safety rule"),
    ("e2", "Disk reclaim on m1max is automated via the disk-reclaim-auto cron", "what reclaims disk space"),
    ("e2", "The Golden eval index stores scores in evaluator.score with experiment_name colon format", "golden index score field"),
    ("e3", "Agent Builder skills live in the security_solution plugin scope", "where are agent builder skills"),
    ("e3", "AlertZero watches are wired into Kibana via the PND integration", "how does alertzero connect"),
    ("e3", "EIS connectors cache under ~/.elastic/eis-connectors-cache.json", "eis connector cache location"),
    ("e3", "The judge model for evals is always gemini-3-1-pro", "which model judges evals"),
    ("e3", "kbn-evals conversations die at 300s headersTimeout on plain-http Kibana clients", "slow model timeout cause"),
    ("e3", "Elastic Defend fleet package is named endpoint not elastic_defend", "defend package name"),
    ("e3", "The Workflows plugin tools take one positional params dict plus injected task_id", "workflow tool call shape"),
    ("e3", "Regression suites must include a known-positive control before claiming zero counts", "zero-count query rule"),
    ("e3", "False-green verification requires mutation tests reverting a fix to red", "how to prove a gate is real"),
    ("e3", "Agent workflow gates on data, never on model judgment", "workflow gating principle"),
]


def build():
    ids = {}
    for owner, text, _ in CORPUS:
        doc = memory.retain(owner, "semantic", text)
        ids[text] = doc["_id"]
    return ids


def evaluate(ids, k=5):
    recalls, mrrs = [], []
    for owner, text, query in CORPUS:
        gold = ids[text]
        hits = memory.recall(owner, query, size=k)["fused"]
        rank = next((i + 1 for i, h in enumerate(hits) if h["id"] == gold), None)
        recalls.append(1 if rank else 0)
        mrrs.append(1.0 / rank if rank else 0.0)
        # negative assertion: no cross-owner private hit ever
        for h in hits:
            if h["visibility"] == "private" and h["owner_id"] != owner:
                raise SystemExit(f"CROSS-OWNER LEAK: {owner} saw {h['owner_id']}: {h['text'][:60]}")
    n = len(CORPUS)
    return sum(recalls) / n, sum(mrrs) / n, n


if __name__ == "__main__":
    ids = build()
    r, m, n = evaluate(ids)
    backend = os.environ.get("AMES_EMBED_BACKEND", "auto")
    print(f"backend={backend} n={n} Recall@5={r:.3f} MRR={m:.3f}")
