"""P2 recall contract: per-hit `confidence` + `abstain_hint`, no-LLM `exists`, and 409 + live_id
when update/delete address a superseded memory.

Runs against the throwaway test ES only (tests/_safety.py refuses anything else). The labelled
probe corpus is seeded under its own owner and wiped first, never against prod.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401

import pytest
from fastapi.testclient import TestClient

from app import auth, memory  # noqa: E402
from app.main import app  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "p2contract"
OTHER = "p2contractother"
client = TestClient(app)
H = {"X-API-Key": auth.create_key(OWNER)}
HO = {"X-API-Key": auth.create_key(OTHER)}


def _clean():
    from app.store import PREFIX
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"terms": {"owner_id": [OWNER, OTHER]}}})
    for name in ("am_rejected", "am_mutations"):     # delete tombstones would reject a re-run's retain
        try:
            es("POST", f"/{PREFIX}{name}/_delete_by_query?refresh=true",
               {"query": {"terms": {"owner_id": [OWNER, OTHER]}}})
        except RuntimeError:
            pass


_clean()


def _retain(text, kind="semantic", headers=H, **kw):
    r = client.post("/memory/retain", headers=headers,
                    json={"kind": kind, "text": text, "visibility": "private", **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ── labelled probe corpus ────────────────────────────────────────────────────────────────────
CORPUS = {
    "harbour": "The harbour crane inspection is scheduled every quarter and the maintenance log is kept by the port engineer.",
    "bakery": "The bakery on the high street opens at six in the morning and closes at two in the afternoon on weekdays.",
    "kafka": "The payments service consumes events from the kafka topic orders-v2 using a consumer group named payments-cg.",
    "oncall": "The on-call rotation for the platform team changes every Monday at ten and is managed in PagerDuty.",
    "postgres": "Production Postgres backups run nightly at 02:00 UTC and are retained for thirty days in the archive bucket.",
    "vpn": "To reach the staging cluster, connect to the corporate VPN and use the bastion host jump-stg.internal.",
    "deploy": "Deployments to production are frozen on Fridays after noon unless the release manager approves an exception.",
    "cat": "Patryk's grey cat is called Biscuit and she is allergic to chicken based food.",
    "invoice": "Invoices over five thousand euros need a second approver from the finance department before payment.",
    "tls": "The wildcard TLS certificate for the internal domain expires in March and is renewed by the cert-manager operator.",
    "marathon": "Anna ran the Warsaw marathon in three hours and fifty minutes after training for sixteen weeks.",
    "schema": "Database schema migrations are written with Flyway and must be backwards compatible for one release.",
}

# (query, expected corpus key | None for 'should abstain')
PROBES = [
    ("when is the harbour crane inspected", "harbour"),
    ("what time does the bakery open", "bakery"),
    ("which kafka topic does the payments service read", "kafka"),
    ("when does the on-call rotation change", "oncall"),
    ("how long do we keep postgres backups", "postgres"),
    ("how do I reach the staging cluster", "vpn"),
    ("can we deploy to production on friday afternoon", "deploy"),
    ("what is my cat's name and what can't she eat", "cat"),
    ("who must approve large invoices", "invoice"),
    ("when does the internal tls certificate expire", "tls"),
    ("how fast did Anna run the marathon", "marathon"),
    ("how are database migrations written", "schema"),
    ("what is the capital of france", None),
    ("who won the football world cup in 1998", None),
    ("explain how photosynthesis works in plants", None),
    ("what is the airspeed velocity of an unladen swallow", None),
    ("recipe for chocolate lava cake with raspberries", None),
    ("how many moons does jupiter have", None),
    ("translate good morning into japanese", None),
    ("what is the best programming language for game development", None),
]


# Hard negatives: on-topic queries (they share vocabulary with a stored doc) whose answer is NOT
# stored. Written against CORPUS only; not tuned to any score.
HARD_NEGATIVES = [
    "what is the weight capacity of the harbour crane",
    "what is the price of a sourdough loaf at the bakery",
    "how many partitions does the orders-v2 kafka topic have",
    "who is on call for the platform team this week",
    "which region holds the postgres backup archive bucket",
    "what is the ssh key passphrase for the bastion host",
    "who approved the last production deploy freeze exception",
    "how old is Patryk's cat Biscuit",
    "which bank pays out the approved invoices",
    "which certificate authority issued the internal wildcard certificate",
    "what was Anna's finishing position at the Warsaw marathon",
    "which flyway version do we run for schema migrations",
]
PROBES_HARD = PROBES + [(q, None) for q in HARD_NEGATIVES]


def auc(scores, labels):
    """Mann-Whitney AUC with average ranks for ties; P(random positive outranks random negative)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = [r for r, l in zip(ranks, labels) if l]
    n1, n0 = len(pos), len(labels) - len(pos)
    return (sum(pos) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


@pytest.fixture(scope="module", params=["app", "es"])
def fusion(request):
    """Run every recall-quality test under both fusion modes: es_rrf hides the per-arm scores, so
    its confidence comes from the stored vector, which ES only returns when asked for it."""
    mp = pytest.MonkeyPatch()
    mp.setenv("AMES_FUSION", request.param)
    assert memory.fusion_mode() == ("es_rrf" if request.param == "es" else "app_rrf")
    yield request.param
    mp.undo()


@pytest.fixture(scope="module")
def corpus_ids():
    _clean()
    ids = {k: _retain(t)["_id"] for k, t in CORPUS.items()}
    yield ids
    _clean()


def _recall(query, **kw):
    r = client.post("/memory/recall", headers=H, json={"query": query, "kinds": ["semantic"], **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _top_conf(out):
    return max((h["confidence"] for h in out["results"]), default=0.0)


@pytest.fixture(scope="module")
def probe_run(corpus_ids, fusion):
    rows = []
    for q, key in PROBES_HARD:
        out = _recall(q)
        rows.append({"q": q, "key": key, "hard": (q, key) not in PROBES, "out": out, "conf": _top_conf(out),
                     "top_is_expected": bool(out["results"]) and out["results"][0]["id"] == corpus_ids.get(key)})
    return rows


def _subset(rows, hard):
    """Answerable probes plus either the easy negatives or the hard negatives."""
    return [r for r in rows if r["key"] or r["hard"] == hard]


# ── confidence ───────────────────────────────────────────────────────────────────────────────
def test_every_hit_has_confidence_in_unit_interval(probe_run):
    hits = [h for r in probe_run for h in r["out"]["results"]]
    assert hits
    for h in hits:
        assert isinstance(h["confidence"], float) and 0.0 <= h["confidence"] <= 1.0


def test_answerable_probe_ranks_expected_doc_first(probe_run):
    answerable = [r for r in probe_run if r["key"]]
    assert sum(r["top_is_expected"] for r in answerable) >= len(answerable) - 1   # label sanity


def _auc_of(rows):
    return auc([r["conf"] for r in rows], [r["key"] is not None for r in rows])


def _abstain_acc(rows, thr):
    """Share of probes where `max confidence < thr` agrees with the label."""
    return sum((r["conf"] < thr) == (r["key"] is None) for r in rows) / len(rows)


def test_confidence_auc_separates_answerable_from_abstain(probe_run, fusion, capsys):
    easy, hard, allp = _subset(probe_run, False), _subset(probe_run, True), probe_run
    real = _auc_of(allp)
    labels = [r["key"] is not None for r in allp]
    scores = [r["conf"] for r in allp]
    rng = random.Random(20261105)
    shuffled = []
    for _ in range(500):
        perm = labels[:]
        rng.shuffle(perm)
        shuffled.append(auc(scores, perm))
    mean_shuf = sum(shuffled) / len(shuffled)
    with capsys.disabled():
        print(f"\nP2 PROBE [{fusion}]: AUC easy-only={_auc_of(easy):.3f} (n={len(easy)})  "
              f"hard-only={_auc_of(hard):.3f} (n={len(hard)})  all={real:.3f} (n={len(allp)})  "
              f"shuffled-mean={mean_shuf:.3f}")
        print(f"  abstain accuracy @{memory.ABSTAIN_CONFIDENCE}: easy={_abstain_acc(easy, memory.ABSTAIN_CONFIDENCE):.2f} "
              f"hard={_abstain_acc(hard, memory.ABSTAIN_CONFIDENCE):.2f} all={_abstain_acc(allp, memory.ABSTAIN_CONFIDENCE):.2f}")
        for r in sorted(allp, key=lambda r: -r["conf"]):
            print(f"  conf={r['conf']:.3f} hint={r['out']['abstain_hint']!s:5} "
                  f"{'ANSWER' if r['key'] else 'HARD-N' if r['hard'] else 'EASY-N'} {r['q']}")
    assert _auc_of(easy) >= 0.8 and real >= 0.8, f"confidence AUC easy={_auc_of(easy):.3f} all={real:.3f} < 0.8"
    assert 0.45 <= mean_shuf <= 0.55, f"negative control: shuffled-label AUC {mean_shuf:.3f} should be ~0.5"


def test_confidence_is_nonzero_for_answerable_probes_in_every_fusion_mode(probe_run, fusion):
    """Regression for es_rrf: ES 9.x omits dense_vector from _source, which zeroed every confidence."""
    answerable = [r for r in probe_run if r["key"]]
    assert sum(r["conf"] > 0.3 for r in answerable) >= len(answerable) - 1, [r["conf"] for r in answerable]


def test_abstain_hint_fires_on_off_topic_not_on_answerable(probe_run, fusion):
    answerable = [r for r in probe_run if r["key"]]
    easy_neg = [r for r in probe_run if not r["key"] and not r["hard"]]
    assert sum(not r["out"]["abstain_hint"] for r in answerable) >= len(answerable) - 2
    assert sum(r["out"]["abstain_hint"] for r in easy_neg) >= len(easy_neg) - 2


def test_abstain_hint_is_only_a_hint_results_are_not_filtered(corpus_ids, fusion):
    weak = [q for q, key in PROBES if key is None]
    seen = 0
    for q in weak:
        out = _recall(q)
        if out["abstain_hint"] is True and out["results"]:
            seen += 1
            assert all(h["confidence"] < memory.ABSTAIN_CONFIDENCE for h in out["results"])   # hint is honest
    assert seen >= 3, "need weak queries that return hits while abstain_hint is true"


def test_abstain_hint_is_true_for_empty_result(corpus_ids):
    out = _recall("zzzz qqqq", min_score=5.0)         # min_score no hit can clear
    assert out["results"] == [] and out["abstain_hint"] is True


def test_confidence_falls_back_to_lexical_when_vector_arm_missing(corpus_ids, monkeypatch):
    from app import embeddings
    monkeypatch.setattr(embeddings, "embed", lambda texts: (_ for _ in ()).throw(RuntimeError("down")))
    out = _recall("harbour crane inspection")
    assert out["vector_arm"] is False
    top = out["results"][0]
    assert top["id"] == corpus_ids["harbour"] and 0.0 < top["confidence"] <= 1.0


def test_confidence_math_units():
    assert memory._confidence(None, None, True) == 0.0
    assert memory._confidence(1000.0, 1.0, True) > 0.95
    assert memory._confidence(0.0, 0.0, True) == 0.0
    assert memory._confidence(8.0, None, False) == round(8.0 / (8.0 + memory.CONF_BM25_K), 4)
    assert memory._confidence(None, 0.95, True) == 1.0     # es_rrf: vector term only
    # blend: both arms contribute, in the configured proportion (neither may be dropped)
    assert memory._confidence(0.0, 0.95, True) == 0.6               # pinned defaults, not read back
    assert memory._confidence(1e9, 0.0, True) == 0.4
    assert (memory.CONF_COS_LO, memory.CONF_COS_HI, memory.CONF_BM25_K) == (0.80, 0.90, 10.0)


# ── exists ───────────────────────────────────────────────────────────────────────────────────
def test_exists_by_text_hash_is_normalised_and_owner_scoped(corpus_ids):
    r = client.post("/memory/exists", headers=H, json={"text": "  " + CORPUS["bakery"].upper() + " "})
    assert r.status_code == 200 and r.json()["exists"] is True
    assert r.json()["matches"][0]["id"] == corpus_ids["bakery"]
    miss = client.post("/memory/exists", headers=H, json={"text": "a sentence nobody ever stored"})
    assert miss.json() == {"exists": False, "count": 0, "matches": []}
    other = client.post("/memory/exists", headers=HO, json={"text": CORPUS["bakery"]})
    assert other.json()["exists"] is False          # a private doc is invisible to another owner


def test_exists_by_tags_requires_all_tags():
    _retain("exists tag probe alpha", tags=["proj:p2x", "area:recall"])
    _retain("exists tag probe beta", tags=["proj:p2x"])
    both = client.post("/memory/exists", headers=H, json={"tags": ["proj:p2x", "area:recall"]}).json()
    one = client.post("/memory/exists", headers=H, json={"tags": ["proj:p2x"], "kind": "semantic"}).json()
    assert both["count"] == 1 and one["count"] == 2
    assert client.post("/memory/exists", headers=H, json={"tags": ["proj:nope"]}).json()["exists"] is False


def test_exists_makes_no_embedding_or_llm_call(monkeypatch):
    from app import embeddings

    def boom(*a, **k):
        raise AssertionError("exists must not embed")
    monkeypatch.setattr(embeddings, "embed", boom)
    monkeypatch.setattr(embeddings, "embed_safe", boom)
    assert client.post("/memory/exists", headers=H, json={"text": CORPUS["cat"]}).status_code == 200


def test_exists_needs_text_or_tags_and_validates_kind():
    assert client.post("/memory/exists", headers=H, json={}).status_code == 422
    assert client.post("/memory/exists", headers=H, json={"text": "x", "kind": "nope"}).status_code == 422
    assert client.post("/memory/exists", headers=H, json={"tags": ["no-colon"]}).status_code == 422


def test_exists_via_mcp_tools_surface():
    r = client.post("/mcp/tools/exists", headers=H, json={"text": CORPUS["cat"]})
    assert r.json()["result"]["exists"] is True


@pytest.mark.parametrize("bad", [[5], {"n": 5}, "5", True, 0, 21, 2.5])
def test_mcp_exists_rejects_a_malformed_size_with_422_not_500(bad):
    r = client.post("/mcp/tools/exists", headers=H, json={"text": CORPUS["cat"], "size": bad})
    assert r.status_code == 422, (bad, r.status_code, r.text)


def test_exists_excludes_superseded_and_inactive_docs():
    old = _retain("exists-excl probe - the lighthouse keeper starts at dusk", tags=["probe:exclusion"])["_id"]
    new = _retain("exists-excl probe - the lighthouse keeper starts at dawn", tags=["probe:exclusion"])["_id"]
    by_text = {"text": "exists-excl probe - the lighthouse keeper starts at dusk"}
    assert client.post("/memory/exists", headers=H, json=by_text).json()["count"] == 1
    memory._supersede(OWNER, old, new)                                # old: active=false + superseded_by
    assert client.post("/memory/exists", headers=H, json=by_text).json()["exists"] is False
    by_tag = client.post("/memory/exists", headers=H, json={"tags": ["probe:exclusion"]}).json()
    assert [m["id"] for m in by_tag["matches"]] == [new] and by_tag["count"] == 1
    es("POST", f"/{idx('semantic')}/_update/{new}?refresh=true", {"doc": {"active": False}})   # inactive, not superseded
    assert client.post("/memory/exists", headers=H, json={"tags": ["probe:exclusion"]}).json()["exists"] is False


# ── superseded ids: 409 + live_id ────────────────────────────────────────────────────────────
def _supersede_chain():
    a = _retain("supersede chain v1 - release train departs from platform nine")["_id"]
    b = _retain("supersede chain v2 - release train departs from platform four")["_id"]
    c = _retain("supersede chain v3 - release train departs from platform two")["_id"]
    memory._supersede(OWNER, a, b)
    memory._supersede(OWNER, b, c)
    return a, b, c


def test_update_on_superseded_id_is_409_with_live_id_following_the_chain():
    a, b, c = _supersede_chain()
    r = client.post("/memory/update", headers=H, json={"kind": "semantic", "doc_id": a, "text": "edited"})
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["live_id"] == c and r.json()["detail"]["superseded"] == a
    # the live doc is untouched, and is itself updatable (200)
    ok = client.post("/memory/update", headers=H, json={"kind": "semantic", "doc_id": c, "tags": ["live:yes"]})
    assert ok.status_code == 200, ok.text


def test_delete_on_superseded_id_is_409_and_deletes_nothing():
    a, b, c = _supersede_chain()
    for r in (client.post("/memory/delete", headers=H, json={"kind": "semantic", "doc_id": b}),
              client.delete(f"/memory/semantic/{b}", headers=H)):
        assert r.status_code == 409 and r.json()["detail"]["live_id"] == c
    assert es("GET", f"/{idx('semantic')}/_doc/{b}")["found"] is True
    assert es("GET", f"/{idx('semantic')}/_doc/{c}")["found"] is True


def _link(ids):
    """ids[i] is superseded by ids[i+1]; the last one is left alone (caller closes a cycle)."""
    for x, y in zip(ids, ids[1:]):
        es("POST", f"/{idx('semantic')}/_update/{x}?refresh=true", {"doc": {"active": False, "superseded_by": y}})


@pytest.mark.parametrize("n", [2, 3, 7])
def test_superseded_cycle_does_not_hang_and_reports_no_live_id(n):
    """Cycles of any length end at the hop bound (no separate visited-set), with no live_id."""
    ids = [_retain(f"cycle probe {n}-{i} - the ferry leaves at {i} o'clock")["_id"] for i in range(n)]
    _link(ids + [ids[0]])
    es("POST", f"/{idx('semantic')}/_update/{ids[-1]}?refresh=true", {"doc": {"active": False, "superseded_by": ids[0]}})
    r = client.post("/memory/delete", headers=H, json={"kind": "semantic", "doc_id": ids[0]})
    assert r.status_code == 409 and r.json()["detail"]["live_id"] is None


def test_chain_longer_than_the_hop_bound_reports_no_live_id_and_exactly_at_the_bound_resolves():
    assert memory.MAX_SUPERSEDE_HOPS == 10      # literal on purpose: the test must not move with the constant
    n = 10
    ids = [_retain(f"long chain probe - the tram stop number {i} is closed")["_id"] for i in range(n + 2)]
    _link(ids)                                                    # ids[-1] is the live doc
    over = client.post("/memory/delete", headers=H, json={"kind": "semantic", "doc_id": ids[0]})
    assert over.status_code == 409 and over.json()["detail"]["live_id"] is None        # n+1 hops
    at_bound = client.post("/memory/delete", headers=H, json={"kind": "semantic", "doc_id": ids[1]})
    assert at_bound.status_code == 409 and at_bound.json()["detail"]["live_id"] == ids[-1]   # n hops


def test_near_duplicate_link_alone_does_not_block_update():
    """retain links near-duplicates with superseded_by but BOTH stay active (coexistence)."""
    a = _retain("coexist probe - the library opens at nine")["_id"]
    b = _retain("coexist probe - the library opens at ten")["_id"]
    es("POST", f"/{idx('semantic')}/_update/{a}?refresh=true", {"doc": {"superseded_by": b}})
    r = client.post("/memory/update", headers=H, json={"kind": "semantic", "doc_id": a, "tags": ["still:active"]})
    assert r.status_code == 200, r.text


def test_foreign_owner_still_gets_403_not_409():
    a, b, c = _supersede_chain()
    r = client.post("/memory/delete", headers=HO, json={"kind": "semantic", "doc_id": a})
    assert r.status_code == 403         # no live_id leak across owners
    assert "live_id" not in r.text and c not in r.text


def test_foreign_owner_update_on_a_superseded_id_gets_403_and_no_live_id():
    a, b, c = _supersede_chain()
    for body in ({"text": "hijack"}, {"tags": ["x:y"]}):
        r = client.post("/memory/update", headers=HO, json={"kind": "semantic", "doc_id": a, **body})
        assert r.status_code == 403, r.text
        assert "live_id" not in r.text and c not in r.text


def test_mcp_tools_update_and_delete_on_a_superseded_id_are_409_with_live_id():
    a, b, c = _supersede_chain()
    up = client.post("/mcp/tools/update", headers=H, json={"kind": "semantic", "doc_id": a, "text": "edited"})
    de = client.post("/mcp/tools/delete", headers=H, json={"kind": "semantic", "doc_id": b})
    for r, sid in ((up, a), (de, b)):
        assert r.status_code == 409, r.text
        assert r.json()["detail"]["live_id"] == c and r.json()["detail"]["superseded"] == sid
    assert es("GET", f"/{idx('semantic')}/_doc/{b}")["found"] is True
