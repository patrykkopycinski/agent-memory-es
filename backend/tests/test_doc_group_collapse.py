"""Round 2: recall collapses to at most `per_doc` passages per doc_group BEFORE the
size cut, so a fixed-size window covers more distinct sessions.

Measured motivation (v3 LongMemEval scopes, 100 questions): the old cap of 3 left the
top-8 spanning a mean of 4.25 distinct sessions (23% of rows only 3).

Pure tests drive _apply_budgets directly; integration tests run against live ES like the
rest of the suite:  docker compose ... run --rm --no-deps -v <repo>:/srv -w /srv/backend \
         ames-backend python -m pytest tests/test_doc_group_collapse.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import memory  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qcollapse"


def _wipe():
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": OWNER}}})


def _items(spec):
    """spec: list of (group, score) in any order -> score-ordered fused items."""
    items = [{"id": "%s-%d" % (g, i), "doc_group": g, "kind": "episodic", "score": sc}
             for i, (g, sc) in enumerate(spec)]
    return sorted(items, key=lambda x: -x["score"])


def _groups(window):
    return [w["doc_group"] for w in window]


# ------------------------------------------------------------------ pure units
def test_default_budget_is_two_per_group():
    assert memory.PER_DOC_BUDGET == 2


def test_eight_slots_cover_at_least_four_groups_when_one_group_dominates():
    # group A owns the 6 best passages; B, C, D, E each have one lower-ranked passage
    spec = [("A", 1.0 - 0.01 * i) for i in range(6)]
    spec += [("B", 0.5), ("C", 0.4), ("D", 0.3), ("E", 0.2)]
    window = memory._apply_budgets(_items(spec), size=8, per_kind_budget=8,
                                   per_doc_budget=memory.PER_DOC_BUDGET)
    groups = _groups(window)
    assert groups.count("A") == 2, groups
    assert len(set(groups)) >= 4, groups
    assert len(window) == 8 or len(window) == 6, len(window)


def test_old_budget_of_three_would_crowd_more():
    spec = [("A", 1.0 - 0.01 * i) for i in range(6)] + [("B", 0.5), ("C", 0.4)]
    old = memory._apply_budgets(_items(spec), size=8, per_kind_budget=8, per_doc_budget=3)
    new = memory._apply_budgets(_items(spec), size=8, per_kind_budget=8, per_doc_budget=2)
    assert _groups(old).count("A") == 3 and _groups(new).count("A") == 2


def test_rank_is_preserved_inside_the_collapse():
    spec = [("A", 0.9), ("A", 0.8), ("A", 0.7), ("B", 0.6), ("B", 0.5)]
    window = memory._apply_budgets(_items(spec), size=8, per_kind_budget=8, per_doc_budget=2)
    assert [round(w["score"], 2) for w in window] == [0.9, 0.8, 0.6, 0.5]


def test_collapse_never_starves_a_single_group_query():
    """If every hit lives in ONE group the window is capped at per_doc, not padded with
    over-cap passages: crowding protection must not invent evidence."""
    spec = [("A", 1.0 - 0.01 * i) for i in range(6)]
    window = memory._apply_budgets(_items(spec), size=8, per_kind_budget=8, per_doc_budget=2)
    assert _groups(window) == ["A", "A"]


def test_items_without_doc_group_are_their_own_group():
    items = [{"id": "x%d" % i, "kind": "episodic", "score": 1.0 - i * 0.1} for i in range(5)]
    window = memory._apply_budgets(items, size=8, per_kind_budget=8, per_doc_budget=2)
    assert len(window) == 5, "ungrouped docs must not be collapsed together"


# ------------------------------------------------------------------ integration
def _crowder_session(n=40):
    """Every passage is strongly on-topic: group A would fill the whole window alone."""
    return "\n".join("user: quarterly warehouse relocation budget plan item %03d "
                      "quarterly warehouse relocation budget %s" % (i, "q" * 60)
                      for i in range(n))


def _lone_hit_session(tag):
    """One weakly on-topic line buried in unrelated filler."""
    filler = "\n".join("user: %s unrelated chatter about cooking %03d %s" % (tag, i, "w" * 70)
                       for i in range(40))
    return filler + "\nuser: the warehouse relocation budget was discussed once"


def _seed():
    _wipe()
    memory.retain(OWNER, "episodic", _crowder_session(), doc_group="sess-A",
                  occurred_at="2023-05-10T09:00:00")
    for g in ("sess-B", "sess-C", "sess-D"):
        memory.retain(OWNER, "episodic", _lone_hit_session(g), doc_group=g,
                      occurred_at="2023-05-11T09:00:00")


Q = "quarterly warehouse relocation budget plan"


def test_without_a_cap_one_group_crowds_the_window_and_the_collapse_fixes_it():
    """The fixture must actually crowd (so the test needs the cap): with a huge per-doc
    budget the on-topic group takes (almost) every slot; the default collapses it."""
    _seed()
    open_ = memory.recall(OWNER, Q, kinds=["episodic"], size=8, as_of="2023-05-30",
                          per_doc=1000)
    capped = memory.recall(OWNER, Q, kinds=["episodic"], size=8, as_of="2023-05-30")
    go, gc = _groups(open_["results"]), _groups(capped["results"])
    assert go.count("sess-A") >= 5, go          # the crowding really happens
    assert gc.count("sess-A") <= memory.PER_DOC_BUDGET, gc
    # the slots the crowder gave up go to OTHER sessions (more evidence diversity)
    assert len([g for g in gc if g != "sess-A"]) > len([g for g in go if g != "sess-A"]), (go, gc)
    assert capped["doc_groups"] == len(set(gc))


def test_window_still_fills_to_size_when_other_groups_have_passages():
    _seed()
    r = memory.recall(OWNER, Q, kinds=["episodic"], size=8, as_of="2023-05-30")
    gs = _groups(r["results"])
    assert len(gs) >= 5, gs      # over-fetch: the cap must not leave the window starved
    assert {"sess-B", "sess-C", "sess-D"} & set(gs), gs


def test_per_doc_one_is_honoured_end_to_end_through_the_http_api():
    from fastapi.testclient import TestClient
    from app.main import app
    from app import auth
    _seed()
    key = auth.create_key(OWNER)
    client = TestClient(app)
    r = client.post("/memory/recall", headers={"X-API-Key": key},
                    json={"query": Q, "kinds": ["episodic"], "size": 8,
                          "as_of": "2023-05-30", "per_doc": 1})
    assert r.status_code == 200, r.text
    gs = _groups(r.json()["results"])
    assert gs and max(gs.count(g) for g in set(gs)) == 1, gs
    r2 = client.post("/memory/recall", headers={"X-API-Key": key},
                     json={"query": Q, "kinds": ["episodic"], "size": 8,
                           "as_of": "2023-05-30", "per_doc": 1000})
    g2 = _groups(r2.json()["results"])
    assert g2.count("sess-A") >= 5, g2           # same data, no cap -> crowded


def test_per_doc_request_param_is_part_of_the_api_model():
    from app.main import RecallIn
    assert RecallIn(query="x", per_doc=1).per_doc == 1
    assert RecallIn(query="x").per_doc is None
