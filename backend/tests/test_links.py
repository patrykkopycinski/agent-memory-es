"""Stored edges + hidden-connections pass (links.py): edge lifecycle, provenance,
visibility, idempotency.

Two-tier: link/find_similar/pass/approve/reject via the app layer (live ES), REST
surface via TestClient. The hidden-connections pass is PROPOSAL-ONLY (status=
'proposed'), never auto-linked — asserting that invariant is the point of this
suite: it is the Distillery #653 adoption with the 0.7.0 governance stance.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

import pytest
from fastapi.testclient import TestClient

from app import auth, embeddings, links, memory  # noqa: E402
from app.links import EDGES, edge_id  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.store import es, idx  # noqa: E402

client = TestClient(fastapi_app)
ALICE = auth.create_key("alice", person="alice")
BOB = auth.create_key("bob", person="bob")
H = lambda k: {"X-API-Key": k}  # noqa: E731


def _seed(owner, text, kind="semantic", visibility="private",
          occurred_at="2026-01-01T00:00:00Z"):
    """Seed a doc directly (bypasses retain's dedup-on-write) with a real vector."""
    doc = {"kind": kind, "owner_id": owner, "visibility": visibility, "text": text,
           "entities": memory.extract_entities(text),
           "occurred_at": occurred_at, "active": True,
           "embedding": embeddings.embed([text])[0]}
    return es("POST", f"/{idx(kind)}/_doc?refresh=true", doc)["_id"]


def _wipe_edges():
    links.ensure_edges_index()   # _delete_by_query 404s on a not-yet-created index
    es("POST", f"/{EDGES}/_delete_by_query?refresh=true", {"query": {"match_all": {}}})


def _wipe_docs(owner):
    """Re-run idempotency: drop this owner's seeded semantic docs (the suite's
    own slice; ES prefixes make prod unreachable)."""
    es("POST", f"/{idx('semantic')}/_delete_by_query?refresh=true",
       {"query": {"term": {"owner_id": owner}}})


# ---- link() core: identity, validation, idempotency -----------------------------------

def test_link_validation_and_idempotency():
    _wipe_edges()
    with pytest.raises(ValueError):
        links.link("o1", "semantic", "a", "b", "bogus_source")
    with pytest.raises(ValueError):
        links.link("o1", "semantic", "a", "b", "human", status="bogus")
    with pytest.raises(ValueError):
        links.link("o1", "semantic", "a", "a", "human")       # self-edge
    e1 = links.link("o1", "semantic", "b", "a", "human")      # reversed pair
    e2 = links.link("o1", "semantic", "a", "b", "human")      # canonical pair
    assert e1["id"] == e2["id"]                               # undirected: same edge
    assert e1["src"] == "a" and e1["dst"] == "b"              # canonicalized
    got = es("GET", f"/{EDGES}/_doc/{e1['id']}")["_source"]
    assert got["status"] == "active" and got["proposer"] == "human"
    assert got["valid_from"] and got["created_at"]            # temporal validity fields


def test_linked_ids_both_directions_and_status():
    _wipe_edges()
    links.link("o1", "semantic", "a", "b", "human")
    links.link("o1", "semantic", "c", "a", "human")
    links.link("o1", "semantic", "a", "z", "co_occurrence", status="proposed")
    ids = set(links.linked_ids("o1", "a"))
    assert ids == {"b", "c"}                                  # proposed excluded (active_only)
    assert "z" in set(links.linked_ids("o1", "a", active_only=False))
    assert links.linked_ids("o2", "a") == []                  # owner scope


# ---- find_similar: hypotheses, visibility, exclusions ---------------------------------

def test_find_similar_excludes_linked_and_inactive():
    _wipe_edges()
    owner = "linksim"
    _wipe_docs(owner)
    auth.create_key(owner, person=owner)
    a = _seed(owner, "OmniRoute gateway restarts at heap 10200MB")
    b = _seed(owner, "OmniRoute heap watchdog restarts the gateway at 10200MB")
    c = _seed(owner, "Kibana jest workers are self-computed per worktree")
    dead = _seed(owner, "OmniRoute heap limit config is 9000MB")
    es("POST", f"/{idx('semantic')}/_update/{dead}?refresh=true",
       {"doc": {"active": False}})
    # link A-B (human): find_similar(A) must drop B, keep nothing-noise out via threshold
    links.link(owner, "semantic", a, b, "human")
    res = links.find_similar(owner, "semantic", a, threshold=0.3, k=5)
    got_ids = {r["id"] for r in res}
    assert b not in got_ids, "linked doc leaked into find_similar"
    assert dead not in got_ids, "inactive doc leaked into find_similar"
    assert c in got_ids or len(res) == 0                      # c is dissimilar: allowed absent
    # cosine sanity: b-vs-a near-dup vectors score higher than c
    if c in got_ids:
        sims = {r["id"]: r["cosine"] for r in res}
        assert sims.get(c, 0) < 0.99                          # c must not look like a twin


def test_find_similar_cross_owner_probe_blocked():
    _wipe_edges()
    owner = "probe_owner"
    _wipe_docs(owner)
    auth.create_key(owner, person=owner)
    secret = _seed(owner, "Alice private token vault detail")  # private, owner=probe_owner
    # BOB cannot probe another owner's private doc even by id
    with pytest.raises(PermissionError):
        links.find_similar("bob", "semantic", secret, threshold=0.0, k=5)
    # same owner CAN see their own private docs in similarity (recall semantics),
    # while a third owner's private doc about the same topic never appears
    auth.create_key("carol_probe", person="carol_probe")
    _wipe_docs("carol_probe")
    carol = _seed("carol_probe", "Carol private token vault detail")
    own = _seed(owner, "probe_owner private second vault note")
    res = links.find_similar(owner, "semantic", own, threshold=0.0, k=5)
    ids = {r["id"] for r in res}
    assert secret in ids, res                     # same-owner private is in scope
    assert carol not in ids, res                  # another owner's private never is


def test_find_similar_doc_not_found():
    assert links.find_similar("nosuch", "semantic", "missing_id", threshold=0.0) == []


# ---- hidden_connections_pass: PROPOSAL-ONLY invariant ---------------------------------

def _twin_cosine(owner, a, b):
    """Known-positive control: observed cosine between two seeded twins, via
    find_similar itself. Fails the suite (not silently returns 0) when the
    similarity signal is absent — a pass proposing 0 must be distinguishable
    from a broken probe."""
    res = links.find_similar(owner, "semantic", a, threshold=0.0, k=5,
                             exclude_linked=False)
    hit = next((r for r in res if r["id"] == b), None)
    assert hit is not None, f"twin control failed: {b} not among top-5 for {a}: {res}"
    return hit["cosine"]


def test_hidden_connections_proposes_never_links():
    _wipe_edges()
    owner = "hcowner"
    _wipe_docs(owner)
    auth.create_key(owner, person=owner)
    a = _seed(owner, "Weekly matrix exports go to GCS bucket eval-exports")
    b = _seed(owner, "Weekly matrix exports upload to GCS bucket eval-exports")
    _seed(owner, "Unrelated note about zsh quoting pitfalls")
    twin = _twin_cosine(owner, a, b)
    th = max(0.05, twin - 0.05)
    stats = links.hidden_connections_pass(owner, threshold=th, max_proposals_per_pass=10)
    assert stats["scanned"] >= 3, stats
    assert stats["proposed"] >= 1, stats                      # near-twin pair proposed
    for e in links.list_edges(owner, edge_source="co_occurrence"):
        assert e["status"] == "proposed", e                   # nothing auto-linked
        assert e["proposer"] == "system"
        assert e["score"] is not None and e["score"] >= th
    # approval flips to active (owner action)
    edge = links.list_edges(owner, status="proposed", edge_source="co_occurrence")[0]
    links.approve_edge(edge["id"], owner)
    assert links.list_edges(owner, status="active", edge_source="co_occurrence")
    # rejection: seed a fresh twin pair so a NEW proposal exists, reject it, and
    # assert the pass never re-proposes a decided (approved or rejected) pair
    c = _seed(owner, "Golden trace waterfall UI needs the vault config export profile")
    d = _seed(owner, "Golden trace waterfall requires config export profile from vault")
    twin2 = _twin_cosine(owner, c, d)
    s2 = links.hidden_connections_pass(owner, threshold=max(0.05, twin2 - 0.05),
                                       max_proposals_per_pass=10)
    if s2["proposed"]:
        rej = links.list_edges(owner, status="proposed", edge_source="co_occurrence")[0]
        links.reject_edge(rej["id"], owner)
    s3 = links.hidden_connections_pass(owner, threshold=th, max_proposals_per_pass=10)
    assert s3["proposed"] == 0, s3                            # decided pairs stay decided
    assert s3["skipped_existing"] >= stats["proposed"], s3


def test_pass_idempotent_never_duplicates():
    _wipe_edges()
    owner = "idemowner"
    _wipe_docs(owner)
    auth.create_key(owner, person=owner)
    a = _seed(owner, "Benchmarker daemon runs only on kibana-i9 host")
    b = _seed(owner, "Benchmarker daemon must run on the kibana-i9 host only")
    twin = _twin_cosine(owner, a, b)
    th = max(0.05, twin - 0.05)
    s1 = links.hidden_connections_pass(owner, threshold=th, max_proposals_per_pass=10)
    n1 = s1["proposed"]
    assert n1 >= 1, s1
    s2 = links.hidden_connections_pass(owner, threshold=th, max_proposals_per_pass=10)
    assert s2["proposed"] == 0 and s2["skipped_existing"] >= n1, s2


def test_orphan_rate_counts():
    _wipe_edges()
    owner = "orphowner"
    _wipe_docs(owner)
    auth.create_key(owner, person=owner)
    a = _seed(owner, "Orphan test alpha fact about gateway restarts")
    b = _seed(owner, "Orphan test beta fact about gateway restart thresholds")
    c = _seed(owner, "Orphan test gamma fact totally different topic quilting")
    links.link(owner, "semantic", a, b, "human")
    s = links.orphan_rate(owner)
    assert s["active_docs"] == 3, s
    assert s["linked_docs"] >= 2, s                           # a + b linked
    assert s["orphans"] == 1 and s["orphan_rate"] == 0.3333, s


def test_structure_components_hubs_bridges():
    # a - b - c chain, plus isolated d:  b is a bridge (removal splits a|c),
    # d its own component. Structure must report both facts.
    _wipe_edges()
    owner = "structowner"
    _wipe_docs(owner)
    auth.create_key(owner, person=owner)
    a = _seed(owner, "Structure test fact alpha about gateway restarts")
    b = _seed(owner, "Structure test fact beta about gateway restarts")
    c = _seed(owner, "Structure test fact gamma about gateway restarts")
    d = _seed(owner, "Structure test fact delta about quilting patterns")
    links.link(owner, "semantic", a, b, "human")
    links.link(owner, "semantic", b, c, "human")
    s = links.structure(owner)
    assert s["active_docs"] == 4, s
    assert s["components"] == 2, s                            # chain + island
    assert s["largest_component"] == 3, s                     # a-b-c
    assert s["linked_docs"] == 3, s                           # d is the orphan
    assert s["orphans"] == 1 and s["orphan_rate"] == 0.25, s
    hub_ids = [h["id"] for h in s["hubs"]]
    assert b in hub_ids and d not in hub_ids, s
    assert b in set(s["bridges"]), s                          # b splits a | c
    assert a not in set(s["bridges"]) and c not in set(s["bridges"]), s


def test_retain_near_duplicate_writes_near_dup_edge():
    # Cause-1 invariant site: the retain path's near-dup LINK (not drop) now also
    # mirrors into the edge store with near_dup provenance, so hidden connections
    # and structure see dedup links. The memory write itself must stay intact.
    _wipe_edges()
    owner = "near_dup_owner"
    _wipe_docs(owner)
    auth.create_key(owner, person=owner)
    r1 = memory.retain(owner, "semantic",
                       "Near dup test: OmniRoute gateway restarts at heap 10200MB")
    assert r1["deduped"] is False, r1
    r2 = memory.retain(owner, "semantic",
                       "Near dup test: OmniRoute gateway restarts at heap limit 10200MB")
    assert r2["dedup_reason"] == "near_duplicate_linked", r2
    assert r2["supersedes"], r2
    edges = [e for e in links.list_edges(owner, edge_source="near_dup")]
    assert len(edges) == 1, edges
    e = edges[0]
    assert e["status"] == "active" and e["proposer"] == "system"
    assert e["kind"] == "semantic"
    assert e["src"] < e["dst"]                                # canonicalized pair
    # the edge spans exactly the retained pair (canonical pair of both doc ids)
    assert {e["src"], e["dst"]} == {r1["_id"], r2["_id"]}
    assert e["score"] is not None and e["score"] >= memory.DEDUP_SIM_THRESHOLD
    # structure() sees the dedup-linked pair as connected
    s = links.structure(owner)
    assert s["linked_docs"] >= 2, s


# ---- REST surface ----------------------------------------------------------------------

def test_rest_links_surface():
    _wipe_edges()
    owner = "restowner"
    _wipe_docs(owner)
    key = auth.create_key(owner, person=owner)
    d1 = _seed(owner, "REST edge test fact one about eval exports to GCS")
    d2 = _seed(owner, "REST edge test fact two uploading eval exports to GCS")
    # find_related: excludes nothing yet, returns candidate with cosine
    r = client.post("/memory/links/find_related", headers=H(key),
                    json={"doc_id": d1, "threshold": 0.3, "k": 5})
    assert r.status_code == 200, r.text
    assert any(h["id"] == d2 for h in r.json()["results"])
    # human link + list + 404/403/409 paths
    r = client.post("/memory/links", headers=H(key),
                    json={"kind": "semantic", "src": d1, "dst": d2, "reason": "same source"})
    assert r.status_code == 200, r.text
    eid = r.json()["id"]
    r = client.get("/memory/links", headers=H(key))
    assert r.status_code == 200 and any(e["id"] == eid for e in r.json()["edges"])
    r = client.post("/memory/links", headers=H(key),
                    json={"kind": "semantic", "src": d1, "dst": d1})
    assert r.status_code == 422, r.text                       # self-edge via API
    r = client.post("/memory/links/find_related", headers=H(key),
                    json={"doc_id": "does-not-exist", "threshold": 0.0})
    assert r.status_code == 404, r.text
    r = client.post("/memory/links/find_related", headers=H(key),
                    json={"doc_id": d1, "kind": "bogus"})
    assert r.status_code == 422, r.text
    r = client.get("/memory/links/structure", headers=H(key))
    assert r.status_code == 200 and r.json()["active_docs"] >= 2, r.text
    # cross-owner: BOB must not see ALICE's owner-scoped edges or act on them
    r = client.get("/memory/links", headers=H(BOB))
    assert not any(e["id"] == eid for e in r.json()["edges"])
    # approve/reject lifecycle via API needs a proposed edge first
    _wipe_edges()
    stats = links.hidden_connections_pass(owner, threshold=0.3, max_proposals_per_pass=10)
    if stats["proposed"]:
        pe = links.list_edges(owner, status="proposed")[0]
        r = client.post("/memory/links/approve", headers=H(BOB), json={"edge_id": pe["id"]})
        assert r.status_code == 403, r.text                   # wrong owner
        r = client.post("/memory/links/approve", headers=H(key), json={"edge_id": "nope"})
        assert r.status_code == 404, r.text
        r = client.post("/memory/links/approve", headers=H(key), json={"edge_id": pe["id"]})
        assert r.status_code == 200 and r.json()["status"] == "active"
        r = client.post("/memory/links/approve", headers=H(key), json={"edge_id": pe["id"]})
        assert r.status_code == 409, r.text                   # not proposed anymore
        r = client.post("/memory/links/reject", headers=H(key), json={"edge_id": pe["id"]})
        assert r.status_code == 409, r.text


def test_worker_links_pass_and_entry_flag():
    _wipe_edges()
    owner = "wlinks"
    _wipe_docs(owner)
    auth.create_key(owner, person=owner)
    a = _seed(owner, "Worker links pass fact: treadmill exports land in GCS daily")
    b = _seed(owner, "Worker links pass fact: treadmill exports landed in GCS daily")
    twin = _twin_cosine(owner, a, b)
    from app.worker import links_pass
    stats = links_pass()
    assert owner in stats, stats
    # the worker default threshold must be below the twin cosine, or the pass is
    # dead in prod: this is the calibration contract, not a test detail
    from app.links import propose_threshold
    assert twin > propose_threshold(), \
        f"twin cosine {twin:.3f} fell below default threshold {propose_threshold()}"
    assert stats[owner]["proposed"] >= 1, stats
    for e in links.list_edges(owner, edge_source="co_occurrence"):
        assert e["status"] == "proposed", e                   # worker proposals only
    # `--links` one-shot entry: module-main dispatch only; assert the dispatch
    # branch exists and the pass function is what it calls
    import inspect
    src = inspect.getsource(links_pass)
    assert "hidden_connections_pass" in src
