"""Cause 2 fix contract: long memories are stored as VERBATIM passages sharing a
`doc_group`, so every turn — not just the head — is embedded and retrievable.

The regression this pins: a whole-session doc (median ~10.5k chars) is embedded only
at its head by e5-small, so a fact at char 3k-6k never entered the vector arm
(context blew up to ~105k chars while Hindsight used ~20k, and multi-session
accuracy was 0.00 on iter50).

Integration tests run against live ES in the backend image, like the rest of the
suite:  docker compose ... run --rm --no-deps -v <repo>:/srv -w /srv/backend \
         ames-backend python -m pytest tests/test_chunking.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _safety  # noqa: F401  (script-path guard: conftest bypass)

from app import embeddings, memory  # noqa: E402
from app.store import es, idx  # noqa: E402

OWNER = "qchunk"


def _wipe(owner=OWNER):
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": owner}}})


def _count(owner=OWNER, kind="episodic"):
    return es("POST", f"/{idx(kind)}/_count",
              {"query": {"term": {"owner_id": owner}}})["count"]


def _session(n_lines, marker_line=None, marker_text="", line_len=90):
    """A synthetic session of ~n_lines lines; optional marker inserted in the middle."""
    lines = []
    for i in range(n_lines):
        if marker_line is not None and i == marker_line:
            lines.append(marker_text)
        else:
            lines.append("user: note %03d %s" % (i, "x" * line_len))
    return "\n".join(lines)


# ------------------------------------------------------------------ pure unit

def test_short_text_is_a_single_passage():
    assert memory.chunk_text("tiny note") == ["tiny note"]
    assert memory.chunk_text("") == [""]


def test_long_text_splits_into_verbatim_contiguous_passages():
    text = _session(200)
    parts = memory.chunk_text(text)
    assert len(parts) > 2, len(parts)
    # every passage is verbatim, and every passage except the last is <= target
    for p in parts:
        assert p in text, "passages must be verbatim slices, never rewritten"
    assert all(len(p) <= memory.CHUNK_TARGET_CHARS for p in parts[:-1])
    # contiguity: the pieces tile the input (with overlap), nothing is skipped
    pos = 0
    for p in parts:
        i = text.index(p, pos if pos == 0 else max(0, pos - memory.CHUNK_OVERLAP_CHARS))
        assert i <= pos, (i, pos)
        pos = i + len(p)
    assert pos == len(text), (pos, len(text))


def test_consecutive_passages_overlap():
    text = _session(200)
    parts = memory.chunk_text(text)
    for a, b in zip(parts, parts[1:]):
        tail = a[-memory.CHUNK_OVERLAP_CHARS:]
        assert tail[:40] in b, "consecutive passages must share an overlapping tail"


def test_chunk_cuts_on_a_line_boundary_when_one_is_available():
    text = "\n".join("line %03d %s" % (i, "y" * 60) for i in range(80))
    parts = memory.chunk_text(text)
    assert all(p.endswith("\n") or p == parts[-1] for p in parts[:-1]), parts


# ------------------------------------------------------------- integration

def test_every_passage_gets_its_own_embedding():
    """The mechanism behind cause 2: with whole-session writes only the head vector
    exists, so mid-document content is invisible to the kNN arm. After chunking every
    passage must carry its own embedding.

    ES does not return dense_vector fields in _source, so the vector is proved by
    querying with the passage's OWN text: the top kNN hit inside the group must be
    that same passage at cosine ~1.0. A head-only vector cannot do this."""
    _wipe()
    text = _session(80, 40, "user: my passport number is XK2049 and it expires on 2031-07-14")
    r = memory.retain(OWNER, "episodic", text, doc_group="sess-emb")
    assert r["passages"] >= 4, r
    docs = [es("GET", f"/{idx('episodic')}/_doc/{i}")["_source"] for i in r["ids"]]
    marker = [d for d in docs if "XK2049" in d["text"]]
    assert len(marker) == 1, "the marker must land in exactly one passage"
    assert marker[0]["passage_index"] > 0, "the marker must NOT be in the head passage"
    qvec = embeddings.embed([marker[0]["text"]])[0]
    res = es("POST", f"/{idx('episodic')}/_search", {
        "size": 3, "_source": ["passage_index"],
        "knn": {"field": "embedding", "query_vector": qvec, "k": 3, "num_candidates": 50,
                "filter": {"term": {"doc_group": "sess-emb"}}},
    })
    hits = res["hits"]["hits"]
    assert hits, "the passage must be kNN-searchable (its own vector must exist)"
    assert hits[0]["_source"]["passage_index"] == marker[0]["passage_index"], hits
    assert hits[0]["_score"] > 0.99, hits[0]["_score"]


def test_mid_session_fact_is_retrievable_after_chunking():
    """The bug: the answer turn sits at char 3k-6k, outside the head-only embedding.
    The query deliberately shares NO token with the marker, so BM25 cannot rescue a
    whole-document write and only the passage embedding can find it."""
    _wipe()
    marker = "user: my passport number is XK2049 and it expires on 2031-07-14"
    text = _session(80, marker_line=40, marker_text=marker)
    assert len(text) > 4 * memory.CHUNK_TARGET_CHARS, len(text)
    assert text.index(marker) > memory.CHUNK_TARGET_CHARS, "marker must sit past the head"
    r = memory.retain(OWNER, "episodic", text, doc_group="sess-mid")
    assert r["passages"] >= 4, r
    assert r["doc_group"] == "sess-mid" and r["_id"] == r["ids"][0]
    assert len(r["ids"]) == r["passages"]
    assert _count() == r["passages"]
    got = memory.recall(OWNER, "when does the travel document I carry expire?",
                        kinds=["episodic"], size=8)
    hits = [h for h in got["results"] if "XK2049" in h["text"]]
    assert hits, [h["text"][:80] for h in got["results"]]
    assert hits[0]["doc_group"] == "sess-mid", hits[0]
    assert hits[0]["occurred_at"], "recall items must carry occurred_at (cause 3 symmetry)"


def test_passages_carry_group_index_and_date():
    _wipe()
    r = memory.retain(OWNER, "episodic", _session(80, 40, "user: marker fact ZQ77"),
                      occurred_at="2024-05-05T09:00:00", doc_group="sess-meta")
    docs = [es("GET", f"/{idx('episodic')}/_doc/{i}")["_source"] for i in r["ids"]]
    for n, d in enumerate(docs):
        assert d["doc_group"] == "sess-meta"
        assert d["passage_index"] == n
        assert d["passages_total"] == r["passages"]
        assert d["occurred_at"] == "2024-05-05T09:00:00"
        assert d["active"] is True


def test_group_without_caller_id_gets_generated_group():
    _wipe()
    a = memory.retain(OWNER, "episodic", "short note A")
    b = memory.retain(OWNER, "episodic", "short note B")
    assert a["doc_group"].startswith("dg_") and a["doc_group"] != b["doc_group"]
    assert a["_id"] == a["ids"][0] and a["passages"] == 1


def test_rewriting_the_same_long_session_dedupes_the_whole_group():
    _wipe()
    text = _session(80, 40, "user: marker fact QQ11")
    first = memory.retain(OWNER, "episodic", text, doc_group="sess-dup")
    again = memory.retain(OWNER, "episodic", text, doc_group="sess-dup")
    assert again["deduped"] is True and again["dedup_reason"] == "exact_text_hash"
    assert again["doc_group"] == "sess-dup"
    assert _count() == first["passages"], "a deduped group must not add passages"


def test_per_doc_budget_still_caps_one_group_in_the_top_window():
    """Anti-cheat: chunking must not let one source session fill the top-8 window.
    PER_DOC_BUDGET (3) already caps a doc_group; this pins it stays capped."""
    _wipe()
    text = _session(120, 60, "user: the shared codename is FLAMINGO-9")
    r = memory.retain(OWNER, "episodic", text, doc_group="sess-budget")
    assert r["passages"] > memory.PER_DOC_BUDGET
    for i in range(4):
        memory.retain(OWNER, "episodic", "unrelated note %d about FLAMINGO-9 planning" % i,
                      doc_group="other-%d" % i)
    got = memory.recall(OWNER, "FLAMINGO-9 codename", kinds=["episodic"], size=8)
    from_group = [h for h in got["results"] if h.get("doc_group") == "sess-budget"]
    assert len(from_group) <= memory.PER_DOC_BUDGET, len(from_group)
