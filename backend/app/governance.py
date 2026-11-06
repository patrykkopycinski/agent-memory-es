"""Decision governance for the shared (team) bank.

A governed entry is a `semantic` memory with visibility `team` and the tag `kind:decision`.
It is written as a PROPOSAL and only becomes `active` (in force) once enough people other than
its author approved it. See docs/GOVERNANCE.md for the rules; the short version:

    retain -> proposed -> active | rejected | expired        (active -> superseded by an override)

Every governed doc carries the same status fields on every passage of its doc_group. The head
passage (passage_index 0) is the state authority: transitions are compare-and-set on it
(`if_seq_no` / `if_primary_term`, retried on conflict) and are then mirrored to the other
passages. Every transition is written to `am_mutations` BEFORE it takes effect, and a failed
audit write aborts the transition (fail closed).

Inactive decisions (rejected, expired, superseded) are stored with `active: false`, so every
existing "active only" path (recall, dedup, consolidation, near-duplicate linking) already
ignores them; recall opts back in with `include_inactive_decisions`.
"""
import calendar
import os
import time
import urllib.parse
from typing import Optional

from . import tagfilter as _tf
from .store import es, idx

DECISION_TAG = _tf.make_tag("kind", "decision")
GOVERNED_MESSAGE = "governed decision: propose an override"
MAX_NOTE_CHARS = 2000
MAX_LIST = 100
MAX_CAS_ATTEMPTS = 8
PAGE = 200
MAX_EXPIRY_PAGES = 1000
PROPOSED, ACTIVE, REJECTED, EXPIRED, SUPERSEDED = (
    "proposed", "active", "rejected", "expired", "superseded")
TERMINAL = (REJECTED, EXPIRED, SUPERSEDED)


class GovernedError(Exception):
    """The request conflicts with the decision's state. HTTP 409."""

    def __init__(self, message: str = GOVERNED_MESSAGE):
        super().__init__(message)


# ---- config (read at call time so tests and operators can change it without a restart) ----

def _env_num(name: str, default: float, cast=float):
    try:
        return cast(os.environ.get(name, "").strip() or default)
    except ValueError:
        return cast(default)


def new_approvals() -> int:
    return max(1, _env_num("AMES_GOV_NEW_APPROVALS", 1, int))


def override_approvals() -> int:
    # never below the new-decision threshold: replacing a decision is the harder act
    return max(new_approvals(), _env_num("AMES_GOV_OVERRIDE_APPROVALS", 2, int))


def proposal_ttl_days() -> float:
    return max(0.0, _env_num("AMES_GOV_PROPOSAL_TTL_DAYS", 14.0, float))


def veto_owners() -> set:
    return {p.strip() for p in os.environ.get("AMES_GOV_VETO_OWNERS", "").split(",") if p.strip()}


# ---- classification ----

def is_governed_write(kind: str, visibility: str, tags) -> bool:
    return kind == "semantic" and visibility == "team" and DECISION_TAG in (tags or [])


def is_governed(src: dict) -> bool:
    return bool(src.get("decision_status"))


def check_mutable(src: dict) -> None:
    """update / delete / promote on a decision that is in force or closed -> 409."""
    status = src.get("decision_status")
    if status in (ACTIVE, SUPERSEDED):
        raise GovernedError()
    if status in (REJECTED, EXPIRED):
        raise GovernedError(f"governed decision is closed ({status}): propose a new decision")


def exclusion_clauses() -> list:
    """must_not clauses that keep governed docs out of consolidation and near-duplicate linking.
    Also covers team decisions written before governance existed and not yet backfilled."""
    return [{"exists": {"field": "decision_status"}},
            {"bool": {"filter": [{"term": {"visibility": "team"}},
                                 {"term": {"tags": DECISION_TAG}}]}}]


# ---- helpers ----

def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _q(doc_id: str) -> str:
    return urllib.parse.quote(str(doc_id), safe="")


def _audit(intent: str, actor: str, doc_id: str, detail: dict) -> Optional[str]:
    """Write the audit row; returns its id so a lost compare-and-set can mark it (see _lost)."""
    from . import memory
    return memory._log_mutation(intent, actor, "semantic", doc_id, detail, strict=True)


def _lost(*row_ids) -> None:
    """Mark the audit rows of an attempt whose compare-and-set lost: the row exists (it was
    written before the attempt, as required) but the transition it describes did not happen."""
    from .store import PREFIX
    for row_id in row_ids:
        if not row_id:
            continue
        try:
            es("POST", f"/{PREFIX}am_mutations/_update/{_q(row_id)}?refresh=true",
               {"doc": {"detail": {"outcome": "conflict"}}})
        except RuntimeError:
            pass                          # best effort: the original row stays as written


def _is_conflict(exc: RuntimeError) -> bool:
    return "-> 409:" in str(exc)


def _scope(src: dict) -> list:
    """Filters selecting the passages of THIS decision: its group, its owner and its text. A doc
    group is only a label, so group alone could also match unrelated docs sharing the name."""
    out = [{"term": {"doc_group": src["doc_group"]}}, {"term": {"owner_id": src["owner_id"]}}]
    if src.get("text_hash"):
        out.append({"term": {"text_hash": src["text_hash"]}})
    return out


def _load(doc_id: str) -> dict:
    """Head passage of the doc's group as {id, src, seq, pt}. LookupError when absent."""
    from . import memory
    if not isinstance(doc_id, str) or not doc_id.strip() or len(doc_id) > memory.MAX_ID_CHARS:
        raise ValueError("doc_id must be a non-empty string")
    try:
        r = es("GET", f"/{idx('semantic')}/_doc/{_q(doc_id)}")
    except RuntimeError as exc:
        if memory.is_doc_miss(exc):
            raise LookupError(f"no such decision: {doc_id}") from exc
        raise
    src = r["_source"]
    if src.get("passage_index", 0) == 0 or not src.get("doc_group"):
        return {"id": r["_id"], "src": src, "seq": r["_seq_no"], "pt": r["_primary_term"]}
    hit = es("POST", f"/{idx('semantic')}/_search", {
        "size": 1, "seq_no_primary_term": True,
        "query": {"bool": {"filter": _scope(src) + [{"term": {"passage_index": 0}}]}}})["hits"]["hits"]
    if not hit:
        raise LookupError(f"no such decision: {doc_id}")
    h = hit[0]
    return {"id": h["_id"], "src": h["_source"], "seq": h["_seq_no"], "pt": h["_primary_term"]}


def _cas(doc: dict, patch: dict) -> bool:
    """Update the head passage iff it has not changed since it was loaded."""
    try:
        es("POST", f"/{idx('semantic')}/_update/{_q(doc['id'])}?refresh=true"
                   f"&if_seq_no={doc['seq']}&if_primary_term={doc['pt']}", {"doc": patch})
        return True
    except RuntimeError as exc:
        if _is_conflict(exc):
            return False
        raise


def _mirror(doc: dict, patch: dict) -> None:
    """Copy a head-passage patch to the other passages of the group."""
    from . import memory
    src = doc["src"]
    if not src.get("doc_group"):
        return
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": memory.MAX_PASSAGES, "_source": False,
        "query": {"bool": {"filter": _scope(src)}}})
    for h in r["hits"]["hits"]:
        if h["_id"] != doc["id"]:
            es("POST", f"/{idx('semantic')}/_update/{_q(h['_id'])}?refresh=true", {"doc": patch})


def _valid_approvals(src: dict) -> list:
    """Approvals that count: bound to the current text, by someone other than the author, one
    per person."""
    seen, out = set(), []
    for a in src.get("approvals") or []:
        who = a.get("by")
        if (a.get("text_hash") == src.get("text_hash") and who and who != src.get("proposed_by")
                and who not in seen):
            seen.add(who)
            out.append(a)
    return out


def approvals_required(src: dict) -> int:
    return override_approvals() if src.get("proposal_for") else new_approvals()


def approvals_needed(src: dict) -> int:
    return max(0, approvals_required(src) - len(_valid_approvals(src)))


# ---- propose (called from memory._retain) ----

def validate_override_target(target_id) -> str:
    """Head id of an ACTIVE governed decision, or ValueError (HTTP 422). Runs before any write."""
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("supersedes_decision must be a non-empty doc id")
    try:
        t = _load(target_id)
    except LookupError:
        raise ValueError(f"supersedes_decision: no such decision: {target_id}") from None
    if t["src"].get("decision_status") != ACTIVE:
        raise ValueError("supersedes_decision must point at an active governed decision "
                         f"(found: {t['src'].get('decision_status') or 'not governed'})")
    return t["id"]


def proposal_fields(owner_id: str, override_of: Optional[str] = None) -> dict:
    fields = {"decision_status": PROPOSED, "proposed_by": owner_id, "approvals": []}
    ttl = proposal_ttl_days()
    if ttl:
        fields["proposal_expires_at"] = _iso(_now() + ttl * 86400)
    if override_of:
        fields["proposal_for"] = override_of
    return fields


def audit_propose(owner_id: str, group: str, text_hash: str, override_of: Optional[str]) -> None:
    _audit("decision_propose", owner_id, group,
           {"text_hash": text_hash, "by": owner_id, "supersedes_decision": override_of,
            "doc_id_is": "doc_group"})


def propose_result(src: dict) -> dict:
    return {"decision_status": src.get("decision_status"), "approvals_needed": approvals_needed(src)}


# ---- approve / reject ----

def _require_open(doc: dict) -> dict:
    src = doc["src"]
    if src.get("decision_status") != PROPOSED:
        raise GovernedError("not an open proposal "
                            f"(status: {src.get('decision_status') or 'not governed'})")
    exp = src.get("proposal_expires_at")
    if exp and exp <= _iso(_now()):
        raise GovernedError("proposal expired")
    return src


def approve(person: str, doc_id: str, note: Optional[str] = None) -> dict:
    """Record `person`'s approval. PermissionError for the author, GovernedError (409) when the
    proposal is not open or `person` already approved. Activates at the threshold.

    An override first CLAIMS its target (compare-and-set active -> superseded) and only then
    activates itself, so two overrides of one decision cannot both activate: the loser's claim
    fails, its proposal stays `proposed` and the approval returns 409."""
    if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE_CHARS):
        raise ValueError(f"note must be a string of at most {MAX_NOTE_CHARS} chars")
    claimed = None                        # (new_id, target_id) once this call superseded the target
    try:
        for attempt in range(MAX_CAS_ATTEMPTS):
            doc = _load(doc_id)
            src = _require_open(doc)
            if person == src.get("proposed_by"):
                raise PermissionError("the author's approval does not count")
            valid = _valid_approvals(src)
            if any(a["by"] == person for a in valid):
                raise GovernedError("already approved by this person")
            entry = {"by": person, "at": _iso(_now()), "text_hash": src["text_hash"]}
            if note:
                entry["note"] = note
            approvals = valid + [entry]
            activating = len(approvals) >= approvals_required(src)
            target = src.get("proposal_for")
            patch = {"approvals": approvals}
            detail = {"text_hash": src["text_hash"], "by": person, "attempt": attempt + 1,
                      "approvals": len(approvals), "required": approvals_required(src)}
            rows = [_audit("decision_approve", person, doc["id"], detail)]
            if activating and target:
                try:
                    _claim_target(person, doc["id"], target)
                except GovernedError:
                    _lost(*rows)
                    raise
                claimed = (doc["id"], target)
            if activating:
                patch.update({"decision_status": ACTIVE, "decided_at": _iso(_now()),
                              "approved_via": "approvals"})
                rows.append(_audit("decision_activate", person, doc["id"], detail))
            if not _cas(doc, patch):
                _lost(*rows)
                if claimed:               # the proposal changed under us: do not hold the target
                    _release_target("system", *claimed)
                    claimed = None
                continue                  # someone else changed it: re-read and recount
            claimed = None                # activated: the claim is now permanent
            _mirror(doc, patch)
            if activating and target:
                _link_target(doc["id"], target)
            return {"_id": doc["id"], "decision_status": patch.get("decision_status", PROPOSED),
                    "approvals": len(approvals),
                    "approvals_needed": max(0, approvals_required(src) - len(approvals)),
                    "activated": activating}
        raise GovernedError("could not record the approval after repeated conflicts; retry")
    except Exception:
        if claimed:
            _release_target("system", *claimed)   # the override did not activate: give the target back
        raise


def _claim_target(person: str, new_id: str, target_id: str) -> None:
    """Compare-and-set the overridden decision from active to superseded (by `new_id`). Already
    superseded by `new_id` is success (a retry after the proposal's own CAS lost). Superseded by
    anyone else, or no longer active: GovernedError, nothing changed."""
    for _ in range(MAX_CAS_ATTEMPTS):
        t = _load(target_id)
        status = t["src"].get("decision_status")
        if status == SUPERSEDED and t["src"].get("superseded_by") == new_id:
            return
        if status != ACTIVE:
            raise GovernedError("override target is no longer an active decision")
        patch = {"decision_status": SUPERSEDED, "active": False, "superseded_by": new_id,
                 "decided_at": _iso(_now())}
        row = _audit("decision_supersede", person, t["id"],
                     {"text_hash": t["src"].get("text_hash"), "by": person, "superseded_by": new_id})
        if _cas(t, patch):
            _mirror(t, patch)
            return
        _lost(row)
    raise GovernedError("could not supersede the override target after repeated conflicts; retry")


def _release_target(actor: str, new_id: str, target_id: str) -> None:
    """Undo a claim whose override did not become active (rejected, withdrawn, expired, or the
    activation failed): the target is an active decision again. No-op unless the target is
    superseded by exactly this proposal and the proposal is not active."""
    try:
        for _ in range(MAX_CAS_ATTEMPTS):
            t = _load(target_id)
            if t["src"].get("decision_status") != SUPERSEDED or t["src"].get("superseded_by") != new_id:
                return
            if _load(new_id)["src"].get("decision_status") == ACTIVE:
                return
            patch = {"decision_status": ACTIVE, "active": True, "superseded_by": None,
                     "decided_at": _iso(_now())}
            row = _audit("decision_restore", actor, t["id"],
                         {"text_hash": t["src"].get("text_hash"), "by": actor, "released_from": new_id})
            if _cas(t, patch):
                _mirror(t, patch)
                return
            _lost(row)
    except LookupError:
        return


def _link_target(new_id: str, target_id: str) -> None:
    es("POST", f"/{idx('semantic')}/_update/{_q(new_id)}?refresh=true",
       {"doc": {"supersedes": [target_id]}})


def _supersede_target(person: str, new_id: str, target_id: str, new_src: dict) -> None:
    """Worker completion of an interrupted override: the proposal is active but never got its
    `supersedes` link. Idempotent."""
    _claim_target(person, new_id, target_id)
    _link_target(new_id, target_id)


def reject(person: str, doc_id: str, reason: str) -> dict:
    """A veto owner's rejection rejects the proposal; anyone else's is an objection."""
    if not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_NOTE_CHARS:
        raise ValueError(f"reason must be a non-empty string of at most {MAX_NOTE_CHARS} chars")
    for attempt in range(MAX_CAS_ATTEMPTS):
        doc = _load(doc_id)
        src = _require_open(doc)
        if person == src.get("proposed_by"):
            raise PermissionError("the author withdraws a proposal by deleting it")
        veto = person in veto_owners()
        objections = list(src.get("objections") or [])
        if not veto and any(o.get("by") == person for o in objections):
            raise GovernedError("already objected by this person")
        objections.append({"by": person, "at": _iso(_now()), "text_hash": src["text_hash"],
                           "reason": reason})
        patch = {"objections": objections}
        if veto:
            patch.update({"decision_status": REJECTED, "active": False, "decided_at": _iso(_now())})
        row = _audit("decision_reject", person, doc["id"],
                     {"text_hash": src["text_hash"], "by": person, "veto": veto,
                      "attempt": attempt + 1})
        if _cas(doc, patch):
            _mirror(doc, patch)
            return {"_id": doc["id"], "decision_status": patch.get("decision_status", PROPOSED),
                    "veto": veto, "objections": len(objections)}
        _lost(row)
    raise GovernedError("could not record the rejection after repeated conflicts; retry")


def withdraw(author: str, doc_id: str) -> dict:
    """The author (a person, or the owner id for a key without one) deleting a proposal: it
    becomes rejected (withdrawn), never hard-deleted."""
    for _ in range(MAX_CAS_ATTEMPTS):
        doc = _load(doc_id)
        src = doc["src"]
        if src.get("proposed_by") != author:
            raise PermissionError("only the creator can delete their memory")
        if src.get("decision_status") != PROPOSED:
            raise GovernedError()
        patch = {"decision_status": REJECTED, "active": False, "decided_at": _iso(_now()),
                 "objections": list(src.get("objections") or []) + [
                     {"by": author, "at": _iso(_now()), "text_hash": src["text_hash"],
                      "reason": "withdrawn"}]}
        row = _audit("decision_withdraw", author, doc["id"],
                     {"text_hash": src["text_hash"], "by": author, "reason": "withdrawn"})
        if _cas(doc, patch):
            _mirror(doc, patch)
            return {"deleted": False, "withdrawn": True, "decision_status": REJECTED,
                    "doc_group": src.get("doc_group") or doc["id"], "kind": "semantic",
                    "doc_id": doc_id}
        _lost(row)
    raise GovernedError("could not withdraw after repeated conflicts; retry")


# ---- listing ----

def list_proposals(limit: int = MAX_LIST) -> dict:
    limit = max(1, min(int(limit), MAX_LIST))
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": limit, "track_total_hits": True,
        "query": {"bool": {"filter": [{"term": {"decision_status": PROPOSED}},
                                      {"term": {"passage_index": 0}},
                                      {"term": {"visibility": "team"}}]}},
        "sort": [{"occurred_at": {"order": "asc"}}]})
    out = []
    for h in r["hits"]["hits"]:
        s = h["_source"]
        out.append({
            "id": h["_id"], "doc_group": s.get("doc_group"), "text": s.get("text"),
            "text_hash": s.get("text_hash"), "proposed_by": s.get("proposed_by"),
            "proposal_for": s.get("proposal_for"), "occurred_at": s.get("occurred_at"),
            "expires_at": s.get("proposal_expires_at"), "tags": s.get("tags", []),
            "approved_by": [a["by"] for a in _valid_approvals(s)],
            "approvals_required": approvals_required(s), "approvals_needed": approvals_needed(s),
            "objections": [{"by": o.get("by"), "reason": o.get("reason"), "at": o.get("at")}
                           for o in s.get("objections") or []]})
    return {"proposals": out, "total": r["hits"]["total"]["value"]}


# ---- maintenance ----

def expire_stale_proposals(now: Optional[float] = None) -> dict:
    """Worker pass: proposals past their TTL without enough approvals become `expired`; interrupted
    overrides (activated, target not yet superseded) are completed. Reads every page of stale
    proposals: each round takes the next PAGE that is still `proposed`."""
    now_iso = _iso(_now() if now is None else now)
    expired = 0
    skipped = []                          # could not be moved (repeated conflicts): not retried here
    for _page in range(MAX_EXPIRY_PAGES):
        must_not = [{"ids": {"values": skipped}}] if skipped else []
        r = es("POST", f"/{idx('semantic')}/_search", {
            "size": PAGE, "_source": False,
            "query": {"bool": {"filter": [{"term": {"decision_status": PROPOSED}},
                                          {"term": {"passage_index": 0}},
                                          {"range": {"proposal_expires_at": {"lte": now_iso}}}],
                               "must_not": must_not}}})
        hits = r["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            for _ in range(MAX_CAS_ATTEMPTS):
                doc = _load(h["_id"])
                src = doc["src"]
                if src.get("decision_status") != PROPOSED:
                    break
                patch = {"decision_status": EXPIRED, "active": False, "decided_at": now_iso}
                row = _audit("decision_expire", "system", doc["id"],
                             {"text_hash": src.get("text_hash"), "by": "system",
                              "expired_at": src.get("proposal_expires_at")})
                if _cas(doc, patch):
                    _mirror(doc, patch)
                    expired += 1
                    break
                _lost(row)
            else:
                skipped.append(h["_id"])
    released = _release_orphaned_claims(now_iso)
    completed = 0
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": 100, "_source": ["proposal_for"],
        "query": {"bool": {"filter": [{"term": {"decision_status": ACTIVE}},
                                      {"term": {"approved_via": "approvals"}},
                                      {"term": {"passage_index": 0}},
                                      {"exists": {"field": "proposal_for"}}],
                           "must_not": [{"exists": {"field": "supersedes"}}]}}})
    for h in r["hits"]["hits"]:
        try:
            _supersede_target("system", h["_id"], h["_source"]["proposal_for"], {})
        except GovernedError:
            continue
        completed += 1
    return {"expired": expired, "completed_overrides": completed, "released_claims": released}


CLAIM_GRACE_SECONDS = 300
CLAIM_WINDOW_SECONDS = 86400


def _release_orphaned_claims(now_iso: str) -> int:
    """An override claims its target before it activates. If the process died in between, the
    target stays superseded by a proposal that never became active: after a grace period, give the
    target back. Looks only at targets superseded in the last day."""
    now = calendar.timegm(time.strptime(now_iso, "%Y-%m-%dT%H:%M:%SZ"))
    lo, hi = _iso(now - CLAIM_WINDOW_SECONDS), _iso(now - CLAIM_GRACE_SECONDS)
    r = es("POST", f"/{idx('semantic')}/_search", {
        "size": PAGE, "_source": ["superseded_by"],
        "query": {"bool": {"filter": [{"term": {"decision_status": SUPERSEDED}},
                                      {"term": {"passage_index": 0}},
                                      {"exists": {"field": "superseded_by"}},
                                      {"range": {"decided_at": {"gte": lo, "lte": hi}}}]}}})
    released = 0
    for h in r["hits"]["hits"]:
        new_id = h["_source"].get("superseded_by")
        try:
            if _load(new_id)["src"].get("decision_status") == ACTIVE:
                continue
        except (LookupError, ValueError):
            continue
        _release_target("system", new_id, h["_id"])
        released += 1
    return released


def backfill(max_pages: int = 100) -> dict:
    """Admin: mark team `kind:decision` entries that predate governance as active
    (`approved_via: backfill`, `proposed_by` = the doc's owner). Compare-and-set per doc, so a
    concurrent transition always wins and a doc that lost is skipped. Idempotent: already-governed
    docs are skipped."""
    done = 0
    skipped = []
    for _ in range(max_pages):
        must_not = [{"exists": {"field": "decision_status"}}]
        if skipped:
            must_not.append({"ids": {"values": skipped}})
        r = es("POST", f"/{idx('semantic')}/_search", {
            "size": PAGE, "_source": ["text_hash", "owner_id"], "seq_no_primary_term": True,
            "query": {"bool": {"filter": [{"term": {"visibility": "team"}},
                                          {"term": {"tags": DECISION_TAG}},
                                          {"term": {"active": True}}],
                               "must_not": must_not}}})
        hits = r["hits"]["hits"]
        if not hits:
            break
        for h in hits:
            owner = h["_source"].get("owner_id")
            row = _audit("decision_backfill", "admin", h["_id"],
                         {"text_hash": h["_source"].get("text_hash"), "by": "admin"})
            patch = {"decision_status": ACTIVE, "approved_via": "backfill",
                     "decided_at": _iso(_now())}
            if owner:
                patch["proposed_by"] = owner
            if _cas({"id": h["_id"], "seq": h["_seq_no"], "pt": h["_primary_term"]}, patch):
                done += 1
            else:
                _lost(row)
                skipped.append(h["_id"])
    return {"backfilled": done}
