"""Replay-based extraction eval (Cortex-optimiser style): replay source documents through
the real `_extract` path and score the facts it writes. No ES, no worker: extraction is
pure (prompt in, facts out), so the eval isolates what the extractor learns from a document
— mirroring how the Nightshift `cortex-optimize` dataset replays saved rounds through the
real optimiser and scores the wiki pages it writes (kibana PR #303323).

Both halves call the LLM gateway: extraction itself runs `facts._extract` -> `llm.chat_json`
before any scoring. With the gateway unreachable nothing is measurable, so the script probes
first (same probe as tests/_llm_gate.py), prints SKIP for both halves, and exits 0.

Run:  python tests/eval_extraction_replay.py           # needs the gateway; judged half included
      AMES_LLM_PROBE_ONLY=1 python tests/...          # skips only the judged half
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import _safety  # noqa: F401  (must precede app imports)

from app import facts, llm

# (document, [required substrings], [forbidden substrings])
# Required/forbidden match lowercased fact text; each required substring must appear in at
# least ONE extracted fact; each forbidden substring must appear in NONE.
CORPUS = [
    (
        "We decided to route all kbn-evals LLM traffic through EIS connectors. "
        "Local vLLM stacks are no longer supported for evals because the EIS cache "
        "gives stable latencies and the models match production. "
        "Renovate PRs are handled by the dependency-updater skill, never manually.",
        ["eis", "vllm"],
        [],
    ),
    (
        "The recall reranker must stay behind AMES_RERANK=1 in staging: enabling it "
        "globally once doubled p95 latency because every query paid the rerank hop. "
        "Rule: measure on new text before flipping a rerank default; cached text lies.",
        ["rerank", "latency"],
        [],
    ),
    (
        "Incident review: the ingest worker dropped 5944 errors during the night of "
        "Nov 9 while the queue was full; restarts repeated four times. Post-incident "
        "decision: the queue now has a hard cap and the worker sheds load instead of "
        "retrying forever.",
        ["queue"],
        ["5944", "four times"],
    ),
    (
        "How-to: boot a throwaway ES for AMES tests with the remote runner. The runner "
        "refuses ports 9200/9220/9268/9279 by design; every suite uses an amtest_ "
        "index prefix and deletes only its own slice.",
        ["amtest", "port"],
        [],
    ),
    (
        "Correction from the review: the earlier claim that recall fuses lexical and "
        "dense scores only when both backends are enabled was wrong. The fusion path "
        "runs whenever either backend returns hits; an empty side simply gets zero "
        "weight. Update your mental model accordingly.",
        # any phrasing of the corrected semantics; the extraction often rewords
        ["fusion", ("either the lexical or dense backend", "either backend", "lexical or dense")],
        [],
    ),
    (
        "Correction to the runbook: the runner talks to the throwaway ES on port 9291, "
        "not 9200 — the old number came from a stale note written before the port "
        "registry existed. Any doc still saying 9200 for review ES is wrong.",
        ["9291"],
        # the stale port must not survive even as a correcting aside: the current-state
        # prompt distils the corrected state only, so any "9200" in a fact is a leak
        ["9200"],
    ),
    (
        "Follow-up after re-testing: the rerank cache effect applies only to repeated "
        "identical texts. We first claimed it made rerank free in general; that was "
        "overbroad. New text always pays the full rerank cost, so budget for it.",
        ["new text", "identical"],
        [],
    ),
    (
        "Notes from the perf session: the pipeline took 18 minutes to drain and the "
        "collector crashed twice at 07:45. Durable conclusion: drain time scales with "
        "shard count, so cap shards per index at three for ingest-heavy workloads.",
        ["shard"],
        ["18 minutes", "07:45"],
    ),
]

JUDGE_PROMPT = (
    "You are scoring an extraction step. For each document, a set of candidate facts was "
    "extracted. Judge each fact on: durable (stays true while the system stays the same; "
    "no single-run details like timestamps, counts of events, or past-tense durations), "
    "self-contained (identifies its subject), and grounded (asserted by the document, no "
    "invented context). Return JSON: {\"scores\": [{\"doc\": 0, \"fact\": \"...\", "
    "\"durable\": true|false, \"self_contained\": true|false, \"grounded\": true|false}]}. "
    "Judge every fact; do not skip any."
)


def deterministic_report(all_facts):
    rows, failed = [], 0
    for (doc, required, forbidden), facts_for_doc in zip(CORPUS, all_facts):
        joined = " || ".join(f.lower() for f in facts_for_doc)
        # a required entry is a string (must appear) or a tuple of alternatives
        # (at least one must appear): extraction rewords, specs match semantics
        missing = [r for r in required
                   if (r.lower() not in joined if isinstance(r, str)
                       else not any(a.lower() in joined for a in r))]
        present = [f for f in forbidden if f.lower() in joined]
        ok = not missing and not present
        failed += 0 if ok else 1
        rows.append({"ok": ok, "required_missing": missing, "forbidden_present": present,
                     "n_facts": len(facts_for_doc)})
    return rows, failed


def judged_report(all_facts):
    payload = [{"doc": i, "document": doc[:1200], "facts": fs}
               for i, ((doc, _, _), fs) in enumerate(zip(CORPUS, all_facts))]
    raw = llm.chat_json(JUDGE_PROMPT, json.dumps(payload))
    scores = raw.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError("judge returned no scores")
    n_submitted = sum(len(fs) for fs in all_facts)
    if len(scores) != n_submitted:
        # a judge that skips or duplicates facts changes the denominator: refuse to score
        raise ValueError(f"judge scored {len(scores)} facts but {n_submitted} were submitted")
    # identity check: each score must name a submitted (doc, fact) pair — a judge that
    # attributes facts to the wrong document or invents text still changes the averages
    submitted = {}
    for i, fs in enumerate(all_facts):
        for f in fs:
            submitted[(i, f)] = submitted.get((i, f), 0) + 1
    seen = {}
    for s in scores:
        key = (s.get("doc"), s.get("fact"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > submitted.get(key, 0):
            raise ValueError(f"judge scored an unsubmitted or duplicated fact (doc={key[0]})")
    per_criterion = {"durable": 0, "self_contained": 0, "grounded": 0}
    for s in scores:
        for c in per_criterion:
            per_criterion[c] += 1 if s.get(c) else 0
    n = len(scores)
    return {c: round(v / n, 3) for c, v in per_criterion.items()}, n


# Judged floors (v4 measured 0.824/1.0/0.941 at 17 facts): a prompt or harness change that
# drops durable below 0.80, self_contained below 0.90, or grounded below 0.90 fails the
# run even when every substring check passes — the judged half gates the exit code.
JUDGED_FLOORS = {"durable": 0.80, "self_contained": 0.90, "grounded": 0.90}


def main():
    # Both halves call the gateway (extraction itself does), so probe before extracting —
    # even in probe-only mode: a down gateway prints SKIP for both halves instead of
    # crashing mid-loop.
    from _llm_gate import llm_unreachable_reason
    gateway_down = llm_unreachable_reason()
    probe_only = os.environ.get("AMES_LLM_PROBE_ONLY") == "1"

    if gateway_down:
        print(f"SKIP extraction+deterministic (llm gateway unreachable: {gateway_down}); "
              f"judged=skipped")
        return 0

    all_facts = []
    for i, (doc, _, _) in enumerate(CORPUS):
        # _extract takes the document date; use a fixed one so the corpus is reproducible
        # (the date is stamped into the fact prefix by the extraction path itself).
        try:
            doc_facts = facts._extract(doc, "2026-11-10")
        except Exception as exc:  # one flaky doc must not kill the corpus report
            print(f"doc {i}: EXTRACT-ERROR {type(exc).__name__}: {exc}")
            doc_facts = []
        all_facts.append(doc_facts)
        for f in doc_facts:
            print(f"  doc {i}: {f}")
        print(f"doc {i}: {len(doc_facts)} facts")

    # a doc whose extraction crashed is reported, never silently dropped: it has no
    # facts, so its required substrings are missing and the deterministic check fails
    crashed = [i for i, fs in enumerate(all_facts) if not fs]
    if crashed:
        print(f"WARNING: docs with zero facts (extraction crashed or empty): {crashed}")

    rows, failed = deterministic_report(all_facts)
    for i, r in enumerate(rows):
        status = "PASS" if r["ok"] else "FAIL"
        print(f"doc {i}: {status} n_facts={r['n_facts']}"
              + (f" missing={r['required_missing']}" if r["required_missing"] else "")
              + (f" forbidden={r['forbidden_present']}" if r["forbidden_present"] else ""))
    det_line = f"docs={len(CORPUS)} deterministic_failures={failed}"

    if probe_only:
        print(f"{det_line} judged=skipped (probe-only mode)")
        return 0 if failed == 0 else 1

    judged, n = judged_report(all_facts)
    print(f"{det_line} judged_facts={n} " + " ".join(f"{k}={v}" for k, v in judged.items()))
    below = [f"{k}={v:.3f}<{JUDGED_FLOORS[k]}" for k, v in judged.items() if v < JUDGED_FLOORS[k]]
    if below:
        print(f"JUDGED-FAIL: " + ", ".join(below))
        return 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
