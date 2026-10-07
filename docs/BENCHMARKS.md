# Benchmarks — agent-memory-es vs Hindsight

AMES is built as a self-hosted replacement for [Hindsight](https://github.com/vectorize-io/hindsight)
(the memory provider it replaces). Every comparison below runs **both systems through the same
harness, the same answer model, and the same judge**; only the memory backend differs.

**Bottom line (holdout, release `abe292d`): parity with Hindsight on LongMemEval-S, with less
context per question.** AMES does not beat Hindsight. Where one side is numerically ahead without
statistical significance, it is reported as parity.

| Benchmark | AMES | Hindsight | Paired test | Verdict |
|---|---|---|---|---|
| **LongMemEval-S holdout 100** (release `abe292d`) | **84/100** | **84/100** | McNemar b=9 c=9, p=1.00, 95% CI [-8, +8] | **Parity (exact tie)** |
| LongMemEval-S anchor 100 (release `abe292d`) | 83/100 | 76/100¹ | McNemar b=15 c=8, p=0.21 | Parity, AMES numerically ahead |
| PersonaMem 32k, all 589 (`2c346cd`, pre-fact-extraction) | 352/589 (59.8%) | 372/589 (63.2%) | McNemar b=58 c=78, p=0.10 | Parity, Hindsight numerically ahead |
| PrecisionMemBench single-turn, 77 cases (`2c346cd`) | 58/77 | 66/77 | — | Hindsight ahead (precision 0.75 vs 0.86) |

¹ Includes 2 Hindsight-side answer-call failures (connection refused on the Hindsight host), scored as
incorrect per policy. Excluding both, AMES is still numerically ahead and the verdict is unchanged.

Context sent to the answer model: AMES ≤ Hindsight on **100/100** holdout questions
(median 17.3k vs 20.0k characters; AMES keeps max 8 items, Hindsight 59–115). On PersonaMem,
median context 3.2k vs 17.5k tokens. AMES recall p95 on the holdout: 53 ms.

---

## Method

### LongMemEval-S

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) `S` split: per question, ~50 chat sessions
are ingested into a fresh memory scope, then the question is answered from recalled context only.

- **Samples are frozen and hashed before any run.** `sample_100.json` (seed 42) is the
  development *anchor*. `sample_holdout100.json` (seed 1337) is the *holdout*. The holdout is
  run **once per frozen release**, and only after a pre-registered pass on the anchor.
  The holdout sample sha256 starts `0c2de15b88b9`.
- **Symmetric arms.** Same answer model and judge for both (OpenAI-compatible gateway, temperature 0),
  same judge source hash, same `as_of` date per question. Neither side has question-specific logic,
  and the product code contains no benchmark vocabulary.
- **Scoring.** One rejudge pass fills empty judge responses using the same policy for both arms.
  After that, `count(label is None) == 0` is asserted. Unanswered questions count as incorrect.
  Comparisons are paired: exact McNemar test plus a bootstrap 95% CI of the paired difference.
- **Claim policy (pre-registered, binding).** "AMES beats Hindsight" only if AMES > HS **and**
  p < 0.05. Otherwise the result is reported as parity, naming the side that is numerically ahead.
  A criterion is never relaxed after the data is seen.

### AMB (Agent Memory Benchmark)

[AMB](https://github.com/vectorize-io/agent-memory-benchmark) is Hindsight's own open harness.
AMES runs through an AMES provider adapter. The dataset, modes, prompts, judge and scoring files
are byte-identical to upstream `f618ed7`. For PersonaMem, both arms ran the same patched LLM
adapter: OpenRouter `gemini-2.5-flash-lite`, temperature 0, the upstream prompt/schema and the
upstream 6 retries. Answer failures are scored as incorrect, not excluded.

---

## Results in detail

### LongMemEval-S holdout (release `abe292d`, the shipped product)

Pre-registered before any holdout row existed. Same frozen config as the anchor pass below.

| Question type | n | AMES `abe292d` | Hindsight | AMES `3771a71` (first holdout spend) |
|---|---|---|---|---|
| knowledge-update | 22 | 14 (0.64) | 17 (0.77) | 19 (0.86) |
| multi-session | 20 | 17 (0.85) | 19 (0.95) | 12 (0.60) |
| single-session-assistant | 10 | 10 (1.00) | 4 (0.40) | 10 (1.00) |
| single-session-preference | 2 | 2 (1.00) | 1 (0.50) | 1 (0.50) |
| single-session-user | 13 | 13 (1.00) | 13 (1.00) | 13 (1.00) |
| temporal-reasoning | 33 | 28 (0.85) | 30 (0.91) | 24 (0.73) |
| **overall** | **100** | **84** | **84** | **79** |

- **vs Hindsight:** 84 vs 84, b=9 c=9, p=1.00 → parity.
- **vs the first holdout spend (`3771a71`):** 84 vs 79, b=14 c=9, p=0.40 → parity, `abe292d` numerically ahead.
- **Validity:** 0 of 4,780 fact-extraction jobs failed; 0 unlabelled rows after one rejudge pass.
- **Retrieval is not the bottleneck.** In 15 of the 16 questions AMES got wrong, every gold
  session reached the answer model. The misses are answer-side: in knowledge-update it picked an
  outdated value (8 misses), in temporal reasoning it got date arithmetic or ordering wrong (5),
  and in multi-session questions it miscounted (3).
- **Where AMES wins:** 6 of AMES's 9 questions over Hindsight are *single-session-assistant*
  questions about what the assistant said earlier. AMES stores and extracts facts from both
  sides of a conversation.

The earlier holdout spend on release `3771a71` (round 3) scored **79 vs 84** (p=0.47, parity, Hindsight
numerically ahead). Write-time fact extraction (below) closed that gap.

### LongMemEval-S anchor (pre-registered ship gate, release `abe292d`)

All five pre-registered criteria passed (scored file `r7a-anchor100_rejudged.jsonl`):

| Criterion | Bar | Result |
|---|---|---|
| S1 multi-session | ≥ 14/23 | 14/23 |
| S2 overall vs previous release (76/100) | ≥ 76 and not significantly worse | 83/100, b=11 c=4, p=0.12 |
| S3 no question type drops > 2 | — | none dropped (KU 17/18, MS 14/23, SSA 13/13, SSP 8/9, SSU 12/12, TR 19/25) |
| S4 single rejudge pass, no other re-scoring | — | 2 empty judgements filled, 0 unlabelled |
| S5 extraction failures at drain ≤ 2% | ≤ 2% | 0/4,774 |

### Development history on the anchor (`sample_100`)

What was tried, what moved, and what was rejected. Rows are AMES vs Hindsight on the same 100
questions unless noted. None of these deltas is statistically significant on its own.

| Round | Change | Anchor result | Decision |
|---|---|---|---|
| 1 | Hybrid BM25 + kNN + RRF baseline (`127cdbe`) | 77 vs HS 76, p=1.00 | Baseline |
| 2 | Per-document passage collapse (`per_doc=2`) | 73/100 (−4 vs baseline, p=0.34); sessions in top-8 4.25 → 5.28 | Reverted to `per_doc=3` |
| 3 | `per_doc=3` + over-fetch (`3771a71`) | 75/100 | Shipped; holdout ingest started |
| 4 | Session-first seeding (breadth) | Session coverage up, evidence completeness flat | Not shipped |
| 5 | ES-native rerank (`_inference/rerank`) | Session-complete 87 → 96/100; answers 33 → 35/50 (p=0.73); **cold latency median 27.7 s** | Merged **opt-in, off by default** |
| 6 | Group-diversity fusion | 76–77/100, no multi-session gain | Not shipped |
| Filtering | Caller tags + label filters (`2c346cd`) | Default path top-8 identical 100/100; PrecisionMemBench 9/77 → 58/77 | Shipped |
| 7 | **Write-time fact extraction** (`abe292d`) | **83/100**, multi-session 11 → 14/23, temporal 17 → 19/25 | **Shipped** |

### PersonaMem 32k (AMB, all 589 questions, release `2c346cd`)

| Category | n | AMES | Hindsight |
|---|---|---|---|
| recalling the reasons behind previous updates | 99 | **88 (0.89)** | 81 (0.82) |
| recall user-shared facts | 129 | 89 (0.69) | 91 (0.71) |
| track full preference evolution | 139 | 74 (0.53) | 76 (0.55) |
| provide preference-aligned recommendations | 55 | 33 (0.60) | 38 (0.69) |
| generalizing to new scenarios | 57 | 41 (0.72) | **46 (0.81)** |
| recalling facts mentioned by the user | 17 | 10 (0.59) | **14 (0.82)** |
| suggest new ideas | 93 | 17 (0.18) | **26 (0.28)** |
| **overall** | **589** | **352 (59.8%)** | **372 (63.2%)** |

Paired McNemar b=58 c=78, p=0.10 → parity, Hindsight numerically ahead. AMES had 2 answer
failures (output length), scored as incorrect. Run on release `2c346cd`, *before* write-time fact
extraction. It has not been re-run on `abe292d` (see follow-ups).

### PrecisionMemBench (AMB, single-turn, retrieval mode)

PrecisionMemBench scores the *exact set* of returned memories: noise counts as a failure. There
is no answer model and no judge.

| Release | Passes | Precision | Recall |
|---|---|---|---|
| `2c346cd` (label filters) | 58/77 (active 26/43, structural 24/25, trivially-empty 8/9) | 0.75 | 0.86 |
| before filters | 9/77 | 0.06 | 1.00 |
| Hindsight | 66/77 | 0.86 | 0.84 |

The remaining gap is precision: AMES returns relevant but over-inclusive sets.

### Published Hindsight numbers (for context, not comparable)

Hindsight publishes LongMemEval-S 94.6%, PersonaMem 86.6% and PrecisionMemBench 85.7%
([arXiv 2512.12818](https://arxiv.org/abs/2512.12818)). Those use their own answer model and judge.
The Hindsight numbers in this document come from running Hindsight ourselves on the same samples,
answer model and judge as AMES. Only those are compared.

---

## Write-time fact extraction (what release `abe292d` added)

Every `retain` of episodic text queues an asynchronous job. A worker
(`python -m app.worker --facts`) asks an LLM for up to **12** dated, source-linked one-line facts.
They are written to `semantic` with `source_id` pointing at the original passage.

- **Supersession:** a newer contradicting fact sets `valid_to` / `superseded_by` on the old fact.
  The old fact is kept for audit, never deleted. Recall at `as_of` excludes facts that had expired
  by that time.
- **Durable queue:** jobs live in `am_fact_jobs`, with a lease token, up to 4 attempts, then `failed`.
  `GET /memory/facts/drain` reports pending, running, completed and failed counts.
  Failed jobs and backfills can be retried: `POST /memory/facts/jobs/{id}/retry` and
  `POST /memory/facts/backfill/retry`.
- **Backfill:** `POST /memory/facts/backfill` queues extraction over an owner's existing episodic
  memory. The backfill worker is `python -m app.worker --backfill`.
- **Opt out:** `"extract_facts": false` on retain, for bulk imports.
- **Model:** `AMES_LLM_MODEL` on an OpenAI-compatible endpoint (`AMES_LLM_BASE`).
  The benchmarked configuration used a small reasoning model at low effort.

---

## Follow-ups

Ordered by expected value. Benchmarking is **closed** for this release: round 7 was the last
pre-registered round and the holdout is spent. The items below are product work. The anchor is
used only as a regression check for them, and no new benchmark claim will be made without a fresh,
untouched sample.

| # | Follow-up | Targets | Status |
|---|---|---|---|
| 1 | **Latest-value-only recall:** extraction assigns a stable slot key (`subject|attribute`); recall collapses on it (ES `collapse`, newest first), so outdated values never reach the answer model | Knowledge-update (8 of 16 holdout misses) | Proposed — needs slot-key design |
| 2 | **Date arithmetic in code:** store ISO dates on facts and precompute the gaps between the dates a question mentions | Temporal reasoning (4–5 misses) | Proposed |
| 3 | **Counting via aggregation:** ES\|QL aggregation over facts tagged with a stable item and category | Multi-session counting (3 misses) | Deferred — depends on 1 |
| 4 | **Queue robustness ("fix-2"):** the heartbeat renews the job lease with the token its own claim minted, and can no longer renew another worker's claim | Crash/reboot recovery; no effect on answers | In progress (4th review requested changes) |
| 5 | **Harness hardening:** stale outputs set aside before any abort; malformed lines refused; `--expect-n` required | Benchmark integrity | Review approved, not yet merged |
| 6 | Re-run PersonaMem and PrecisionMemBench on `abe292d` | The two benchmarks still on the pre-extraction release | Not scheduled |
| 7 | ES-native rerank on by default | Retrieval ordering | **Rejected for now** — cold latency (median 27.7 s); recall is already saturated |
| 8 | `semantic_text` migration | Maintenance | Paused — accuracy-neutral; align with the Elastic Context Engine schema instead |
| 9 | PrecisionMemBench precision gap (0.75 vs 0.86) | Over-inclusive recall sets | Open |

### Rejected / closed

- **Diversity collapse (`per_doc=2`), session-first seeding, group-diversity fusion:** each changed
  retrieval as designed, but none improved answers on the anchor (rounds 2, 4 and 6).
- **More benchmark rounds on the same samples:** at n=100 the 95% CI is roughly ±8 points. Further
  tuning against the anchor would overfit it, and the holdout can be spent only once per release.
