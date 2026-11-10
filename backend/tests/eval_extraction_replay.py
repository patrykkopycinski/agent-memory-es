"""Replay-based extraction eval (Cortex-optimiser style): replay source documents through
the real `_extract` path and score the facts it writes. No ES, no worker: extraction is
pure (prompt in, facts out), so the eval isolates what the extractor learns from a document
— mirroring how the Nightshift `cortex-optimize` dataset replays saved rounds through the
real optimiser and scores the wiki pages it writes (kibana PR #303323).

Deterministic checks run always; LLM-judged scoring runs only when the gateway is up
(same probe as tests/_llm_gate.py — script mode, so a missing gateway prints SKIP for the
judged half and exits 0 with the deterministic half still reported).

Run:  python tests/eval_extraction_replay.py           # deterministic + judged if gateway up
      AMES_LLM_PROBE_ONLY=1 python tests/...          # deterministic checks only
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
        ["9200 for review"],
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
    per_criterion = {"durable": 0, "self_contained": 0, "grounded": 0}
    for s in scores:
        for c in per_criterion:
            per_criterion[c] += 1 if s.get(c) else 0
    n = len(scores)
    return {c: round(v / n, 3) for c, v in per_criterion.items()}, n


def main():
    all_facts = []
    for doc, _, _ in CORPUS:
        # _extract takes the document date; use a fixed one so the corpus is reproducible
        # (the date is stamped into the fact prefix by the extraction path itself).
        all_facts.append(facts._extract(doc, "2026-11-10"))
        for f in all_facts[-1]:
            print(f"  doc {len(all_facts) - 1}: {f}")
        print(f"doc {len(all_facts) - 1}: {len(all_facts[-1])} facts")

    rows, failed = deterministic_report(all_facts)
    for i, r in enumerate(rows):
        status = "PASS" if r["ok"] else "FAIL"
        print(f"doc {i}: {status} n_facts={r['n_facts']}"
              + (f" missing={r['required_missing']}" if r["required_missing"] else "")
              + (f" forbidden={r['forbidden_present']}" if r["forbidden_present"] else ""))
    det_line = f"docs={len(CORPUS)} deterministic_failures={failed}"

    if os.environ.get("AMES_LLM_PROBE_ONLY") == "1":
        print(f"{det_line} judged=skipped (probe-only mode)")
        return 0 if failed == 0 else 1

    from _llm_gate import llm_unreachable_reason
    reason = llm_unreachable_reason()
    if reason:
        print(f"{det_line} judged=skipped (llm gateway unreachable: {reason})")
        return 0 if failed == 0 else 1

    judged, n = judged_report(all_facts)
    print(f"{det_line} judged_facts={n} " + " ".join(f"{k}={v}" for k, v in judged.items()))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
