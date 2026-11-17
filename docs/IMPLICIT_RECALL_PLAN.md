# Plan: implicit (associative) recall in AMES

> **Sequencing (2026-11-17):** this is Phase 3 of the EngProd Brain + team-memory plan
> (`ao-workspace/projects/engprod-brain/plans/implicit-recall-brain-team-memory-plan.md`). The Brain and
> team memory (Context Engine) go first. AMES feature work here starts only after a Brain-side mechanism
> passes its gate there. The PersonaMem rerun (Phase 0 here, follow-up #6) may still run as a plain
> measurement.
>
> Status: **draft plan, nothing built.** Every claim about current behaviour names its source file
> (as of `c89baaa`). Every claim about future effect is an **unverified hypothesis** until the
> pre-registered eval in §6 says otherwise.

## 1. Problem

Some questions need a memory that shares **no words and little meaning** with the question:

> Months ago: "I was diagnosed with type 2 diabetes."
> Today: "Suggest a breakfast for tomorrow."

A good answer avoids sugary cereal. Nothing in "breakfast" points to "diabetes". Hybrid retrieval
(BM25 + kNN + RRF, `memory.py` `_recall_inner`) ranks by surface or embedding similarity, so the
clue is unlikely to reach the answer model unless the bank is tiny.

This is where AMES is weakest today. PersonaMem 32k (`2c346cd`, `docs/BENCHMARKS.md`):

| PersonaMem category | AMES | Shape of the question |
|---|---|---|
| suggest new ideas | **0.18** | Needs preferences the user never tied to this topic |
| generalizing to new scenarios | **0.72** | Apply a known trait to a new situation |
| recalling facts mentioned by user | **0.59** | Facts said in passing, inside other topics |

The overall gap with Hindsight on that benchmark (352 vs 372/589, p=0.10) is not significant, and
that run predates write-time fact extraction (`abe292d`). So the first job is to **measure again**;
the plan is built to stop early if extraction already closed the gap.

## 2. What the Memorilla paper and video teach

Memorilla (arXiv 2610.21375, snap-stanford/memorilla) compresses each stored document into 16 soft
tokens that are fed straight into the model's input-embedding layer.

**Not adoptable, for hard reasons:**

- Soft tokens need the model's embedding layer. Our answer models are hosted (Claude, GPT, GLM
  behind OmniRoute) and expose no such layer.
- It is trained for one pair (Qwen3-Embedding-4B + Qwen3-8B) and needs an 80 GB CUDA GPU.
  oMLX on m1max cannot run its training or inference path.
- Answers carry **no provenance**: a soft token cannot be cited, audited, superseded or deleted.
  AMES governance (`source_id`, `valid_to`, tombstones, visibility) depends on text.

**Results weaker than they look:**

- Paper wins come after stage-3 fine-tuning on the same 8 benchmarks: in-distribution.
- Mean-pooling a whole document into one vector drops names and dates. The video caught this:
  it invented a shellfish allergy and "Emily, April 12" (the real name was Priya). The project
  site admits it "can lose fine details such as exact names".
- Its wins are on **gist** tasks (PersonaMem-v2 44.6 vs 27.9 full context). It loses on detail
  tasks (PubMedQA 59 vs 69; video plot questions: full context 5/8, RAG 4/8, Memorilla 1/8).
- The 27× speed-up excludes embedding. The video measured ~6× (3.2 s with embedding vs 19 s full
  context). RAG latency was never measured.
- The video's own buried-clue test failed for **every** method, and its full-context arm had a
  prompt-framing bug, so it is not evidence that full context solves the problem either.

**What is worth borrowing: the target, not the mechanism.** Memorilla bets that a compact,
always-present summary of the user beats retrieving by similarity for gist questions. We can test
that bet with **readable text plus provenance**, which keeps names exact and every claim citable.

## 3. Ideas, shaped

| # | Idea | Verdict | Why |
|---|---|---|---|
| A | Latent / soft-token memory (Memorilla as is) | **Reject** | Needs embedding-layer access and one fixed open model; no provenance; loses exact details (§2) |
| B | Buried-clue eval + fresh PersonaMem sample | **Do first** (Phase 0) | Without it nothing below can be judged; the holdout for `abe292d` is spent |
| C | Write-time **trait** extraction: lasting user traits (health, diet, values, constraints, routines, strong preferences) as their own semantic facts | **Borrow** (Phase 1) | Turns "said in passing" into a first-class, short, retrievable fact; reuses the queue, `source_id`, supersession in `facts.py` |
| D | Fixed-budget **profile block**: a short readable summary of active trait facts injected every turn | **Borrow** (Phase 2) | This is Memorilla's bet in text form: the clue is present without needing to be retrieved. Cost: tokens every turn |
| E | Trait-aware **second query**: "which known traits of this user matter for this question?", merged into fusion | **Borrow** (Phase 3) | Covers traits too many to fit in the block; reuses the `rewrite_queries` path reflect already has |
| F | Graph edges / `graph_expand` for association | **Defer** | `graph_expand` (`graph.py`) hops on entities **in the query**; diabetes→breakfast shares none. Follow-up #10 measured graph as neutral. Revisit only if Phase 3 misses are entity-linked |
| G | Promote worker draft models (`worker.py` `_candidate_clusters`) into a profile | **Reject as the mechanism** | Drafts cluster by most frequent entity ("X facts and conventions"), not by user trait, and are never auto-promoted by design. `match_model` is BM25 top-1 on the pattern, so it also needs word overlap. Keep am_models for curated answers |
| H | HyDE (embed a hypothetical answer) | **Fold into E** | Same cost class; E's prompt asks for traits explicitly, which is the gap |
| I | Bigger recall size / full context | **Reject** | Costs context on every question; the video and our per_doc rounds show more context does not fix ranking |

## 4. Design sketch

### Phase 1 — trait facts (`facts.py`)

- A **separate** extraction pass with its own prompt, not more instructions in `_extract`.
  `_extract` (`PROMPT_VERSION = "document-v5-short-chat"`) is the change that won LongMemEval;
  editing it risks that result. A separate pass can be turned off on its own.
- Output: 0–3 trait sentences per document, each starting with the subject
  ("The user has type 2 diabetes."), stored in `semantic` with:
  - `tags: ["fact:trait", "trait:<category>"]` (no mapping change: `tags` is already a keyword
    field, `store.py`),
  - `source_id` of the passage (same as facts), so every trait is citable and deletable,
  - visibility of the source (a private passage never yields a team trait).
- Categories (closed list, so the block can be grouped and filtered): `health`, `diet`,
  `constraint`, `preference`, `value`, `routine`, `relationship`, `goal`.
- Supersession: reuse `_replacement_ids` restricted to `fact:trait` so "I'm vegan now" supersedes
  "I eat meat".
- Prompt rule: **only what the user states about themselves.** No inference from one remark
  ("asked about gluten" ≠ "is coeliac"). Questions are never traits.
- Flag: `AMES_TRAIT_EXTRACTION` (default off), `TRAIT_PROMPT_VERSION` stamped on each fact.

### Phase 2 — profile block

- New endpoint `GET /memory/profile` → active `fact:trait` facts of the caller's owner, newest
  first, grouped by category, capped at a fixed budget (start: 400 tokens / 25 traits).
- Each line keeps its fact id: `[id] The user has type 2 diabetes.` — readable, citable.
- Hermes plugin (`hermes_plugin/agent_memory_es/__init__.py`): `prefetch` prepends the block to the
  recall hits. Cached per session (refreshed on `sync_turn` writes) so it costs one ES call, not one
  per turn. MCP: optional `include_profile` on `memory_recall`.
- Health and relationship traits are sensitive: they go in the block only from `private` facts of
  the same owner, never from team scope.

### Phase 3 — trait-aware second query

- On recall (opt-in `trait_query: true`, and by default inside reflect): one small LLM call
  "List up to 3 kinds of personal facts that would change a good answer to this question"
  → query restricted to `tags: fact:trait` → merged into RRF at its own weight.
- Skip on trivial prompts (the Hermes plugin already filters them with Hermes' `is_trivial_prompt`
  before calling recall; the server has no such check, so MCP/HTTP callers opt in explicitly) and when the profile block already holds every
  trait (owner has ≤ budget traits): then Phase 3 adds latency for nothing.

## 5. Phases and gates

| Phase | Work | Gate to continue |
|---|---|---|
| **0** | Buried-clue fixture + baseline on `fork/main`; PersonaMem rerun on current release (closes BENCHMARKS follow-up #6) | Baseline shows a real miss: buried-clue pass rate < 60% **or** PersonaMem implicit categories still < HS. If extraction already closed it → **stop, record, no feature work** |
| **1** | Trait extraction pass behind flag; unit tests on prompt output shape; trait precision audit on 50 docs | Trait precision ≥ 90% (human-labelled: stated, not inferred); LongMemEval anchor not worse than −3 (regression check only, not a claim) |
| **2** | `/memory/profile` + prefetch block | Buried-clue pass rate up vs Phase 0 on the dev split; control questions unchanged |
| **3** | Trait-aware second query | Adds pass rate on top of Phase 2 on the dev split; p95 recall latency increase ≤ 1.5 s |
| **4** | Pre-registered fresh evaluation (§6) | Final verdict; ship flags on only if §7 success criteria hold |

Phases 2 and 3 are independent experiments on top of Phase 1: if Phase 2 alone reaches the target,
Phase 3 is dropped.

## 6. Evaluation design

### 6a. Buried-clue fixture (deterministic, fast, in CI)

Mirrors the video's test, built like `backend/tests/eval_recall.py` (owner prefix `ame_eval_`,
wipes its own slice, cross-owner negative assertion).

- 40 personas. Each: 30–60 filler messages over several sessions, **one** trait buried in an
  unrelated message ("…anyway, since the diabetes diagnosis I've been tired, but back to the
  migration…"), and a later question that needs it with **no lexical or entity overlap**.
- Every probe has a **control question** that does not need the trait, to catch the profile block
  bending unrelated answers (the "irrelevant memory bends the answer" failure reflect already
  guards against).
- Every persona also has a **distractor trait** that must not be used (e.g. a sibling's allergy),
  to catch over-application.
- Metrics:
  - retrieval: is the trait fact in the context given to the answer model (by id) — deterministic;
  - answer: LLM judge, three labels `uses_trait` / `ignores_trait` / `misapplies_trait`;
  - control: answer unchanged vs the no-profile arm (judge, `same` / `different`).
- Split 20 dev / 20 test personas, written before any Phase 1 code. Dev may be iterated on;
  test is run **once**, in Phase 4.

### 6b. PersonaMem-v2 fresh sample (external, the claim-bearing run)

- Source: HF `bowen-upenn/PersonaMem-v2`. Sample from the personas **not** used in any earlier run;
  the train/val split has no persona overlap with test, so a fresh draw is possible.
- n chosen for power, not convenience: at n=100 the 95% CI is about ±8 points (BENCHMARKS policy).
  Target the implicit categories with **n ≥ 300** questions so a 10-point gain is detectable.
- Arms: AMES current release, AMES + traits (+ block, + query, as enabled), Hindsight. Same harness
  (AMB), answer model and judge as every earlier comparison.
- Paired McNemar per arm vs baseline. Pre-register in this file, **before the run**: sample seed,
  n, arms, primary metric (accuracy on the three implicit categories pooled), secondary metrics
  (all categories, context tokens per question, p95 latency).

### 6c. Regression guard

LongMemEval-S anchor 100: run as a **regression check only** (not a claim; it has been tuned
against). Fail if AMES + traits drops by more than 3 on it.

## 7. Success and kill criteria

**Ship (flags on by default) only if all hold on the fresh, pre-registered runs:**

1. PersonaMem implicit categories: AMES + traits beats AMES baseline with p < 0.05.
2. All PersonaMem categories pooled: not worse than baseline (no significant loss).
3. Buried-clue test split: `uses_trait` rate up ≥ 20 points; `misapplies_trait` ≤ 5%; control
   answers changed ≤ 5%.
4. LongMemEval anchor within −3.
5. Cost: extra context ≤ 400 tokens per turn; extraction cost per retain ≤ +40% of today's.

**Kill (stop the line, record why in BENCHMARKS.md) if any hold:**

- Phase 0 shows the gap is already gone.
- Trait precision < 90% after two prompt revisions (hallucinated traits are worse than none).
- Phase 2 and Phase 3 both fail to move the dev split.
- Control-question drift > 10%: the block is bending unrelated answers.

## 8. Risks

| Risk | Mitigation |
|---|---|
| **Invented traits** (the Memorilla failure, in text form) | Stated-only prompt rule; `source_id` on every trait; precision audit gate; supersession; owner can delete/retract |
| **Over-application**: diet trait bends a coding answer | Control questions; block framed as "background, use only when relevant"; kill criterion on drift |
| **Privacy**: health and relationship traits are sensitive | Traits inherit source visibility; sensitive categories private-only and never promoted; existing secrets guard still runs on the source text |
| **Extraction cost** doubles per retain | Separate pass with a tiny output (0–3 items); skip passages under N chars or with no first-person statements; flag-gated |
| **Benchmark contamination** | Fresh persona draw; test split run once; pre-registration recorded before the run |
| **Stale profile** (old traits never expire) | Supersession + `valid_to`; block sorts newest first; trait age shown in the block |
| **Interaction with follow-up #1** (slot keys) | Traits are natural slot-key candidates (`user|diet`); design them so #1 can reuse the category, not fight it |

## 9. Relation to other work

- **BENCHMARKS follow-up #6** (PersonaMem rerun on `abe292d`) becomes Phase 0 of this plan.
- **Follow-up #10** (graph edges): stays neutral; this plan does not depend on it (idea F).
- **`CE_BENCHMARK_PLAN.md`** (AMES vs Context Engine B5): independent. If both run, share the
  PersonaMem harness setup and judge, but keep separate pre-registrations and samples so neither
  contaminates the other.
- **am_models**: unchanged. The profile block is not a mental model: it is generated from facts,
  not user-curated, and is always injected rather than BM25-matched.

## 10. Open decisions

1. Default for the profile block in Hermes once shipped: on for every turn, or only when the prompt
   is non-trivial and recall returns few hits?
2. Is a 400-token per-turn budget acceptable on the paid models, or should the block be capped
   lower (200) and Phase 3 carry the rest?
3. Run Phase 4's Hindsight arm on m1max (as before) or skip Hindsight and compare AMES to itself
   only? The latter is cheaper but drops the external reference point.

## Pre-registration (fill before Phase 4; do not edit after the run starts)

- Release / commit: _tbd_
- PersonaMem sample seed and n: _tbd_
- Arms and flags: _tbd_
- Answer model, judge model: _tbd_
- Primary metric and test: implicit categories pooled, paired McNemar vs baseline, α = 0.05
