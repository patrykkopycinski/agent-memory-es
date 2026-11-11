"""Stored relations between memories: edges with provenance, validity and a
scheduled hidden-connections pass.

Design (adopted from Distillery 0.5-0.7, issue #653):
- Edges live in their own index `am_edges`, never in the doc body: multiple
  mechanisms (near-dup link, entity co-occurrence, human, approved proposal)
  write edges concurrently without compare-and-set churn on memories.
- Every edge records HOW it came to exist (`edge_source`) and temporal
  validity (`valid_from`/`valid_to`), mirroring the facts governance fields.
- Semantic auto-link is NEVER silent: the hidden-connections pass only files
  proposals into the governance queue; an edge becomes `active` when the
  proposal activates. Distillery converged on the same stance (0.7.0 made
  auto-link config-gated, off by default).
- `find_related(doc_id, exclude_linked=True)` surfaces unlinked-but-similar
  memories: hypotheses the embedding space generates, never auto-linked.

Edge identity is deterministic: `_id = sha1(f"{src}\0{dst}\0{edge_source}")`
so re-running a pass is idempotent (PUT overwrites, 409-free).
"""
import datetime as _dt
import hashlib
import logging
import os
from typing import Optional

from .store import es, is_doc_miss, knn_num_candidates, PREFIX

log = logging.getLogger(__name__)

EDGES = f"{PREFIX}am_edges"

# Mechanisms that may create an edge. `edge_source` is stored per edge so a
# human-authored edge is never silently replaced by a machine one.
EDGE_SOURCES = ("human", "near_dup", "co_occurrence", "proposal")

# structure() bridge/articulation probe is O(nodes * edges): past this many
# linked docs the probe is skipped and `bridges` comes back empty (the snapshot
# degrades to components/hubs/orphans — bounded, never wrong).
REMOVAL_PROBE_LIMIT = 200

# structure() fetches the owner's active doc ids to identify orphans exactly
# (an edge may point at a now-inactive doc, so active-minus-linked would
# miscount). Beyond this many docs the snapshot falls back to the estimate.
STRUCT_DOC_LIMIT = 2000

EDGE_MAPPINGS = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "owner_id": {"type": "keyword"},      # scope: edges never cross owners
            "kind": {"type": "keyword"},          # memory kind of both endpoints
            "src": {"type": "keyword"},           # doc id
            "dst": {"type": "keyword"},           # doc id
            "edge_source": {"type": "keyword"},   # how this edge came to exist
            "status": {"type": "keyword"},        # active | proposed | rejected
            "score": {"type": "float"},           # cosine at proposal time (proposal/co_occurrence)
            "proposer": {"type": "keyword"},      # 'system' for worker passes
            "created_at": {"type": "date"},
            "valid_from": {"type": "date"},
            "valid_to": {"type": "date"},         # null = still valid
            "reason": {"type": "keyword", "ignore_above": 256},
        },
    },
}


def ensure_edges_index() -> None:
    """Create the edge index if missing. Idempotent; race-safe (409 tolerated)."""
    try:
        es("GET", f"/{EDGES}")
    except RuntimeError as exc:
        if "-> 404:" not in str(exc):
            raise
        try:
            es("PUT", f"/{EDGES}", EDGE_MAPPINGS)
        except RuntimeError as race:
            if "-> 400:" not in str(race):   # resource_already_exists_exception
                raise


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def edge_id(src: str, dst: str, edge_source: str) -> str:
    """Deterministic id per (src, dst, mechanism): re-runs of a pass overwrite
    in place instead of duplicating."""
    return hashlib.sha1(f"{src}\0{dst}\0{edge_source}".encode()).hexdigest()


def _canonical_pair(a: str, b: str) -> tuple:
    """Order a doc-id pair so an edge A-B is the same doc as B-A (undirected).
    Symmetric relations (near-dup, similarity) store one edge per pair."""
    return (a, b) if a <= b else (b, a)


def link(owner_id: str, kind: str, src: str, dst: str, edge_source: str,
         status: str = "active", score: Optional[float] = None,
         reason: Optional[str] = None, proposer: str = "human") -> dict:
    """Write one edge. Undirected semantics (pair canonicalized). Idempotent per
    (pair, mechanism): re-running a pass refreshes in place. `status='proposed'`
    files a candidate awaiting governance; `active` writes a real edge."""
    if edge_source not in EDGE_SOURCES:
        raise ValueError(f"edge_source must be one of {EDGE_SOURCES}")
    if status not in ("active", "proposed", "rejected"):
        raise ValueError("status must be active | proposed | rejected")
    if src == dst:
        raise ValueError("self-edges are not stored")
    a, b = _canonical_pair(src, dst)
    ensure_edges_index()
    doc = {
        "owner_id": owner_id,
        "kind": kind,
        "src": a,
        "dst": b,
        "edge_source": edge_source,
        "status": status,
        "proposer": proposer,
        "created_at": _now(),
        "valid_from": _now(),
    }
    if score is not None:
        doc["score"] = round(float(score), 4)
    if reason:
        doc["reason"] = reason[:256]
    es("PUT", f"/{EDGES}/_doc/{edge_id(a, b, edge_source)}?refresh=true", doc)
    return {"id": edge_id(a, b, edge_source), **doc}


def unlink(edge_id_value: str) -> dict:
    """Remove one edge by id (human take-back)."""
    ensure_edges_index()
    try:
        es("DELETE", f"/{EDGES}/_doc/{edge_id_value}?refresh=true")
    except RuntimeError as exc:
        if not is_doc_miss(exc):
            raise
    return {"deleted": edge_id_value}


def linked_ids(owner_id: str, doc_id: str, active_only: bool = True) -> list:
    """Doc ids already linked to `doc_id` (either direction). Cheap exact query."""
    ensure_edges_index()
    filt = [{"term": {"owner_id": owner_id}}]
    if active_only:
        filt.append({"term": {"status": "active"}})
    r = es("POST", f"/{EDGES}/_search", {
        "size": 200,
        "query": {"bool": {"filter": filt,
                           "should": [{"term": {"src": doc_id}}, {"term": {"dst": doc_id}}],
                           "minimum_should_match": 1}}})
    out = []
    for h in r["hits"]["hits"]:
        s = h["_source"]
        out.append(s["dst"] if s["src"] == doc_id else s["src"])
    return out


def list_edges(owner_id: str, status: Optional[str] = None,
               edge_source: Optional[str] = None, size: int = 100) -> list:
    ensure_edges_index()
    filt = [{"term": {"owner_id": owner_id}}]
    if status:
        filt.append({"term": {"status": status}})
    if edge_source:
        filt.append({"term": {"edge_source": edge_source}})
    size = max(1, min(int(size), 500))
    r = es("POST", f"/{EDGES}/_search", {
        "size": size, "query": {"bool": {"filter": filt}},
        "sort": [{"created_at": {"order": "desc"}}]})
    return [{"id": h["_id"], **h["_source"]} for h in r["hits"]["hits"]]


def find_similar(owner_id: str, kind: str, doc_id: str, threshold: float = 0.7,
                 k: int = 5, exclude_linked: bool = True) -> list:
    """Unlinked-but-similar memories: hypotheses, never auto-linked.

    kNN by the doc's stored vector against active docs in the same scope,
    minus already-linked ids. `threshold` is real cosine (ES scores are
    (1 + cosine) / 2; inverted here). Superseded/inactive docs are excluded
    by the active filter (Distillery #603 fix analog)."""
    from .store import idx as _idx
    try:
        # ES 9.x omits dense_vector from _source by default (see memory.py
        # _source includes comment); _source_includes forces it back.
        r = es("GET", f"/{_idx(kind)}/_doc/{doc_id}?_source_includes=embedding,owner_id,visibility,supersedes,superseded_by")
        src = r["_source"]
    except RuntimeError as exc:
        if is_doc_miss(exc):
            return []
        raise
    # Visibility guard: a doc this owner cannot see (another owner's private
    # doc) must not be usable as a similarity probe. Treat as not-found (404
    # upstream) so existence is not revealed.
    if src.get("owner_id") != owner_id and src.get("visibility") not in ("team", "common"):
        raise PermissionError("doc is not visible to this owner")
    vec = src.get("embedding")
    if vec is None:
        return []
    skip = set(linked_ids(owner_id, doc_id)) if exclude_linked else set()
    skip.add(doc_id)
    skip.update(src.get("supersedes") or [])
    if src.get("superseded_by"):
        skip.add(src["superseded_by"])
    # Same visibility semantics as recall: the owner's private docs plus shared
    # team/common (an owner's team doc may match their own private docs too).
    flt = [{"term": {"active": True}},
           {"bool": {"should": [{"term": {"owner_id": owner_id}},
                                 {"terms": {"visibility": ["team", "common"]}}],
                     "minimum_should_match": 1}},
           {"term": {"kind": kind}}]
    fetch = k + len(skip)
    r = es("POST", f"/{_idx(kind)}/_search", {
        "size": fetch,
        "knn": {"field": "embedding", "query_vector": vec, "k": fetch,
                "num_candidates": knn_num_candidates(fetch),
                "filter": {"bool": {"filter": flt,
                                    "must_not": [{"terms": {"_id": sorted(skip)}}]}}},
        "_source": ["text", "occurred_at"]})
    out = []
    for h in r["hits"]["hits"]:
        cosine = 2.0 * h["_score"] - 1.0
        if cosine < threshold:
            continue
        out.append({"id": h["_id"], "cosine": round(cosine, 4),
                    "text": h["_source"].get("text", "")[:160],
                    "occurred_at": h["_source"].get("occurred_at")})
    return out[:k]


def propose_threshold() -> float:
    """Cosine floor for pass proposals, read per call so operators can tune a
    deployment without a restart (see hidden_connections_pass for calibration)."""
    return float(os.environ.get("AMES_LINKS_PROPOSE_THRESHOLD", "0.5"))


def hidden_connections_pass(owner_id: str, kind: str = "semantic",
                            threshold: Optional[float] = None,
                            max_proposals_per_pass: int = 25) -> dict:
    """Scheduled pass (worker): for the owner's active semantic docs, find
    similar-but-unlinked docs and file each pair as a `proposed` co_occurrence
    edge — a link PROPOSAL, never an active edge. Approval (governance or the
    owner) flips status to active.

    The pass only writes edges with `status='proposed'` and `proposer='system'`;
    nothing is auto-linked (Distillery 0.7.0 settled on the same default-off
    stance for semantic auto-link).

    `threshold` calibration is model-specific (Distillery's 0.7 was OpenAI
    embeddings; AMES's local model sits lower): tune AMES_LINKS_PROPOSE_THRESHOLD
    per deployment rather than chasing a universal constant. Proposals are cheap
    — they queue for human decision — so erring low costs review time, not truth."""
    from .store import idx as _idx
    if threshold is None:
        threshold = propose_threshold()
    ensure_edges_index()
    scan = es("POST", f"/{_idx(kind)}/_search", {
        "size": 200,
        "_source": ["text", "occurred_at"],
        "query": {"bool": {"filter": [
            {"term": {"owner_id": owner_id}},
            {"term": {"active": True}},
            {"exists": {"field": "embedding"}}]}},
        "sort": [{"occurred_at": {"order": "desc"}}]})
    hits = scan["hits"]["hits"]
    stats = {"scanned": len(hits), "proposed": 0, "skipped_existing": 0, "candidates": []}
    # Everything the pass already decided about: proposed (awaiting decision),
    # active (linked), rejected (known-bad — re-proposing would overwrite the
    # rejection, so rejected ids are treated as decided too).
    existing = set()
    for st in ("proposed", "active", "rejected"):
        existing |= {e["id"] for e in list_edges(owner_id, status=st,
                                                 edge_source="co_occurrence", size=500)}
    for h in hits:
        if stats["proposed"] >= max_proposals_per_pass:
            break
        doc_id = h["_id"]
        sims = find_similar(owner_id, kind, doc_id, threshold=threshold, k=3,
                            exclude_linked=True)
        for s in sims:
            pair = _canonical_pair(doc_id, s["id"])
            eid = edge_id(pair[0], pair[1], "co_occurrence")
            if eid in existing:
                stats["skipped_existing"] += 1
            else:
                link(owner_id, kind, pair[0], pair[1], "co_occurrence",
                     status="proposed", score=s["cosine"],
                     reason=f"cosine {s['cosine']:.3f} unlinked similarity",
                     proposer="system")
                existing.add(eid)
                stats["proposed"] += 1
                stats["candidates"].append({"edge": eid, "src": pair[0], "dst": pair[1],
                                            "cosine": s["cosine"]})
            if stats["proposed"] >= max_proposals_per_pass:
                break
    return stats


def _load_edge(edge_id_value: str, owner_id: str) -> dict:
    """Edge source if it exists and belongs to this owner; raises otherwise."""
    ensure_edges_index()
    try:
        r = es("GET", f"/{EDGES}/_doc/{edge_id_value}")
    except RuntimeError as exc:
        if is_doc_miss(exc):
            raise KeyError(f"edge {edge_id_value} not found") from None
        raise
    src = r["_source"]
    if src.get("owner_id") != owner_id:
        raise PermissionError("edge belongs to another owner")
    return src


def approve_edge(edge_id_value: str, owner_id: str) -> dict:
    """Flip a proposed edge to active (owner action, not silent)."""
    src = _load_edge(edge_id_value, owner_id)
    if src.get("status") != "proposed":
        raise ValueError(f"edge status is {src.get('status')}, not proposed")
    es("POST", f"/{EDGES}/_update/{edge_id_value}?refresh=true",
       {"doc": {"status": "active"}})
    return {"id": edge_id_value, "status": "active"}


def reject_edge(edge_id_value: str, owner_id: str) -> dict:
    """Reject a proposed link (owner action). The pair stays unlinked; the
    rejected edge is kept so the pass does not re-propose it."""
    src = _load_edge(edge_id_value, owner_id)
    if src.get("status") != "proposed":
        raise ValueError(f"edge status is {src.get('status')}, not proposed")
    es("POST", f"/{EDGES}/_update/{edge_id_value}?refresh=true",
       {"doc": {"status": "rejected", "valid_to": _now()}})
    return {"id": edge_id_value, "status": "rejected"}


def orphan_rate(owner_id: str, kind: str = "semantic") -> dict:
    """Orphaned docs (active, embedded, no active edge) over active docs."""
    return structure(owner_id, kind)


def _active_edges(owner_id: str, kind: Optional[str] = None, limit: int = 2000) -> list:
    """Active edges as (src, dst) tuples. `kind` narrows to edges stored for one
    memory kind: the structure snapshot divides active docs (per kind) by linked
    docs, so an edge from another kind must not count toward this kind's total."""
    ensure_edges_index()
    out = []
    filt = [{"term": {"owner_id": owner_id}},
            {"term": {"status": "active"}}]
    if kind is not None:
        filt.append({"term": {"kind": kind}})
    r = es("POST", f"/{EDGES}/_search", {
        "size": limit, "_source": ["src", "dst"],
        "query": {"bool": {"filter": filt}}})
    for h in r["hits"]["hits"]:
        s = h["_source"]
        out.append((s["src"], s["dst"]))
    return out


def structure(owner_id: str, kind: str = "semantic") -> dict:
    """Structural snapshot of the owner's edge graph (Distillery `--structure`
    analog): orphan rate over the kind's active docs, plus connected components
    (communities approximation), degree hubs, and articulation-point bridges
    (nodes whose removal splits a component). Bounded: beyond
    REMOVAL_PROBE_LIMIT linked docs the bridge list degrades to empty."""
    from .store import idx as _idx
    ensure_edges_index()
    edges = _active_edges(owner_id, kind)
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    # connected components (union-find)
    parent = {n: n for n in adj}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps = {}
    for n in adj:
        comps.setdefault(find(n), []).append(n)
    sizes = sorted((len(v) for v in comps.values()), reverse=True)

    # hubs by degree
    hubs = sorted(((n, len(nb)) for n, nb in adj.items()),
                  key=lambda t: (-t[1], t[0]))[:5]

    # articulation points ("bridges"): nodes whose removal splits a component.
    # Correct-and-simple removal probe, bounded: beyond REMOVAL_PROBE_LIMIT nodes
    # the list stays empty (the snapshot degrades to components/hubs, never lies).
    bridges = []
    if len(adj) <= REMOVAL_PROBE_LIMIT:

        def _components_without(skip):
            p = {n: n for n in adj if n != skip}

            def f2(x):
                while p[x] != x:
                    p[x] = p[p[x]]
                    x = p[x]
                return x
            for a, b in edges:
                if a == skip or b == skip:
                    continue
                ra, rb = f2(a), f2(b)
                if ra != rb:
                    p[ra] = rb
            return len({f2(n) for n in p})

        base = len(comps)
        for n in adj:
            if _components_without(n) > base:
                bridges.append(n)

    total = es("POST", f"/{_idx(kind)}/_search", {
        "size": 0, "track_total_hits": True,
        "query": {"bool": {"filter": [{"term": {"owner_id": owner_id}},
                                      {"term": {"active": True}}]}}})
    active = total["hits"]["total"]["value"]
    # Orphans (active docs with no active edge) are singleton components of the
    # memory graph. Exact ids are fetched: an edge may point at a now-inactive
    # doc (supersede keeps edges), so active-minus-len(adj) would miscount.
    # Beyond STRUCT_DOC_LIMIT docs the snapshot falls back to the estimate and
    # orphans stop counting toward components (degraded, never wrong).
    orphan_ids: list = []
    exact = False
    if 0 < active <= STRUCT_DOC_LIMIT:
        r = es("POST", f"/{_idx(kind)}/_search", {
            "size": STRUCT_DOC_LIMIT, "_source": False,
            "query": {"bool": {"filter": [{"term": {"owner_id": owner_id}},
                                          {"term": {"active": True}}]}}})
        ids = [h["_id"] for h in r["hits"]["hits"]]
        if len(ids) >= active:
            orphan_ids = [i for i in ids if i not in adj]
            exact = True
    orphans = len(orphan_ids) if exact else max(0, active - len(adj))
    linked = active - orphans
    components = len(sizes) + (len(orphan_ids) if exact else 0)
    largest = max([sizes[0] if sizes else 0, 1 if orphan_ids else 0])
    return {
        "active_docs": active, "linked_docs": linked, "orphans": orphans,
        "orphan_rate": round(orphans / active, 4) if active else 0.0,
        "components": components, "largest_component": largest,
        "hubs": [{"id": n, "degree": d} for n, d in hubs],
        "bridges": bridges,
    }
