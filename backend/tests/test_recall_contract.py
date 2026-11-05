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
def probe_run(corpus_ids):
    rows = []
    for q, key in PROBES:
        out = _recall(q)
        rows.append({"q": q, "key": key, "out": out, "conf": _top_conf(out),
                     "top_is_expected": bool(out["results"]) and out["results"][0]["id"] == corpus_ids.get(key)})
    return rows


# ── confidence ───────────────────────────────────────────────────────────────────────────────
def test_every_hit_has_confidence_in_unit_interval(probe_run):
    hits = [h for r in probe_run for h in r["out"]["results"]]
    assert hits
    for h in hits:
        assert isinstance(h["confidence"], float) and 0.0 <= h["confidence"] <= 1.0


def test_answerable_probe_ranks_expected_doc_first(probe_run):
    answerable = [r for r in probe_run if r["key"]]
    assert sum(r["top_is_expected"] for r in answerable) >= len(answerable) - 1   # label sanity


def test_confidence_auc_separates_answerable_from_abstain(probe_run, capsys):
    scores = [r["conf"] for r in probe_run]
    labels = [r["key"] is not None for r in probe_run]
    real = auc(scores, labels)
    rng = random.Random(20261105)
    shuffled = []
    for _ in range(500):
        perm = labels[:]
        rng.shuffle(perm)
        shuffled.append(auc(scores, perm))
    mean_shuf = sum(shuffled) / len(shuffled)
    with capsys.disabled():
        print(f"\nP2 PROBE: n={len(PROBES)} AUC real={real:.3f} shuffled-mean={mean_shuf:.3f} "
              f"(500 seeded shuffles)")
        for r in sorted(probe_run, key=lambda r: -r["conf"]):
            print(f"  conf={r['conf']:.3f} hint={r['out']['abstain_hint']!s:5} "
                  f"{'ANSWER ' if r['key'] else 'ABSTAIN'} {r['q']}")
    assert real >= 0.8, f"confidence AUC {real:.3f} < 0.8"
    assert 0.45 <= mean_shuf <= 0.55, f"negative control: shuffled-label AUC {mean_shuf:.3f} should be ~0.5"


def test_abstain_hint_fires_on_unanswerable_not_on_answerable(probe_run):
    answerable = [r for r in probe_run if r["key"]]
    abstain = [r for r in probe_run if not r["key"]]
    assert sum(not r["out"]["abstain_hint"] for r in answerable) >= len(answerable) - 2
    assert sum(r["out"]["abstain_hint"] for r in abstain) >= len(abstain) - 2


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


def test_superseded_cycle_does_not_hang_and_reports_no_live_id():
    a = _retain("cycle probe one - the ferry leaves at dusk")["_id"]
    b = _retain("cycle probe two - the ferry leaves at dawn")["_id"]
    for x, y in ((a, b), (b, a)):
        es("POST", f"/{idx('semantic')}/_update/{x}?refresh=true", {"doc": {"active": False, "superseded_by": y}})
    r = client.post("/memory/delete", headers=H, json={"kind": "semantic", "doc_id": a})
    assert r.status_code == 409 and r.json()["detail"]["live_id"] is None


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
