"""Issue #1 acceptance: a long procedural doc must not take a fused top-3 slot for
unrelated queries.

Every kind's rank-1 hit gets the same cross-kind RRF weight, and `procedural` is a small
pool, so one long procedural doc becomes that kind's #1 for almost any query. Two arms put
it there:
  * kNN: without a floor, the nearest neighbour is returned however far away it is.
    Fixed by `min_similarity`; pinned below with nonsense queries.
  * lexical: the `text` field has no stopword filtering, so "what is the capital of
    france" matches a long English doc on "the"/"of"/"is". The kNN floor cannot remove
    that; AMES_LEXICAL_STOPWORDS (memory.LEXICAL_STOPWORDS) strips stopwords from the
    BM25 text clause.

FLOOR is calibrated for multilingual-e5-small (the suite's embedding endpoint). On 30
labelled query->doc pairs, the lowest relevant cosine is 0.840 and the highest nonsense
cosine is 0.833. The positive control fails loudly if a different model shifts that
range.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qproctop3"
GROUP = "proctop3-long-profile"
FLOOR = 0.84

LONG_PROCEDURAL = "Project profile and operating procedure. " + " ".join([
    "The team maintains a set of services deployed to staging and production clusters.",
    "Every change goes through a pull request with at least one reviewer and green CI.",
    "Release cadence is weekly; hotfixes are cherry-picked onto the release branch.",
    "On-call rotates weekly and owns incident triage, paging, and postmortems.",
    "Dashboards track latency, error rate, saturation and cost per tenant.",
    "Secrets live in the vault and are never committed; rotate on any suspected leak.",
    "Benchmarks are pre-registered and the holdout is spent once per frozen release.",
    "Local development uses docker compose with a throwaway search cluster.",
] * 10)

FILLER = [
    ("semantic", "OmniRoute heap watchdog warns at 9450MB and restarts at 10200MB"),
    ("semantic", "Kibana PR author email must be dev@example.com for the CLA"),
    ("semantic", "kbn-evals conversations die at 300s headersTimeout on plain HTTP"),
    ("episodic", "Never force-push PR branches; rebase locally then regular push"),
    ("episodic", "ssh heredocs eat single quotes; ship scripts as files instead"),
    ("episodic", "The deslop gate runs on the whole branch diff vs upstream main"),
]

NONSENSE = ["zzqx qqvv nonsense", "asdf ghjk lqwe", "xylophone quartz vortex 7781",
            "blorp snarf wibble", "qwertyuiop"]
UNRELATED = ["what is the capital of france", "best pizza dough hydration",
             "how tall is mount everest", "recipe for banana bread",
             "who won the 1998 world cup"]
# No word in common with the doc, so only the kNN arm can surface it (cosine 0.858).
ON_TOPIC = "deployment frequency pager duty roster"


def _wipe():
    for kind in ("episodic", "semantic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": OWNER}}})


@pytest.fixture(scope="module", autouse=True)
def corpus():
    _wipe()
    memory.retain(OWNER, "procedural", LONG_PROCEDURAL, doc_group=GROUP, extract_facts=False)
    for kind, text in FILLER:
        memory.retain(OWNER, kind, text, extract_facts=False)
    yield
    _wipe()


@pytest.fixture(autouse=True)
def _no_env_floor(monkeypatch):
    monkeypatch.setattr(memory, "KNN_MIN_SIMILARITY", None)


def _in_top3(query, floor):
    res = memory.recall(OWNER, query, size=8, rerank=False, min_similarity=floor)["results"]
    return any(h.get("doc_group") == GROUP for h in res[:3])


def test_without_floor_long_procedural_doc_is_top3_for_nonsense():
    # The bug as reported. If this stops holding, the floor test below proves nothing.
    assert all(_in_top3(q, None) for q in NONSENSE)


def test_floor_keeps_long_procedural_doc_out_of_top3_for_nonsense():
    leaked = [q for q in NONSENSE if _in_top3(q, FLOOR)]
    assert leaked == []


def test_floor_keeps_long_procedural_doc_for_on_topic_query():
    assert _in_top3(ON_TOPIC, FLOOR)


def test_floor_alone_leaves_long_procedural_doc_top3_for_unrelated_questions():
    # Control for the next test: without stopword stripping the lexical arm matches
    # the/of/is, so the kNN floor cannot keep the doc out.
    assert any(_in_top3(q, FLOOR) for q in UNRELATED)


def test_stopword_lexical_keeps_long_procedural_doc_out_of_top3_for_unrelated_questions(monkeypatch):
    monkeypatch.setattr(memory, "LEXICAL_STOPWORDS", True)
    leaked = [q for q in UNRELATED if _in_top3(q, FLOOR)]
    assert leaked == []
    assert _in_top3(ON_TOPIC, FLOOR)


# ------------------------------------------------------------------ query shape (no ES)
def test_default_lexical_clause_is_unchanged(monkeypatch):
    monkeypatch.setattr(memory, "LEXICAL_STOPWORDS", False)
    assert memory._lexical_clause("what is the plan") == {"multi_match": {
        "query": "what is the plan", "fields": ["text^3", "entities"], "type": "most_fields"}}


def test_stopword_clause_strips_text_query_and_keeps_entities_raw(monkeypatch):
    monkeypatch.setattr(memory, "LEXICAL_STOPWORDS", True)
    q = "What is THE capital of France? isn't it Bank of America"
    text_q, ent_q = memory._lexical_clause(q)["bool"]["should"]
    assert text_q == {"match": {"text": {"query": "What capital France? isn't Bank America", "boost": 3}}}
    assert ent_q == {"match": {"entities": q}}


def test_all_stopword_query_matches_no_text(monkeypatch):
    monkeypatch.setattr(memory, "LEXICAL_STOPWORDS", True)
    assert memory._lexical_clause("the of is")["bool"]["should"][0]["match"]["text"]["query"] == ""
