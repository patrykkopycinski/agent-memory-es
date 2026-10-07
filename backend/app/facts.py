"""Durable, source-linked write-time fact extraction. No benchmark-specific vocabulary."""
import datetime as dt
import hashlib
import json
import logging
import os
import re
import time
import uuid
import threading

from . import embeddings, llm
from . import http_retry
from .store import es, idx, PREFIX

log = logging.getLogger(__name__)
QUEUE = f"{PREFIX}am_fact_jobs"
BACKFILLS = f"{PREFIX}am_fact_backfills"
BACKFILL_MAPPING = {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"dynamic": "strict", "properties": {
        "owner_id": {"type": "keyword"}, "visibility": {"type": "keyword"},
        "state": {"type": "keyword"}, "cursor": {"type": "keyword"},
        "queued": {"type": "integer"}, "error": {"type": "keyword", "ignore_above": 1024},
        "lease_token": {"type": "keyword"}, "lease_until": {"type": "date"},
        "created_at": {"type": "date"}, "finished_at": {"type": "date"}}}}
MAX_ATTEMPTS = 4
LEASE_SECONDS = 3600  # covers bounded extraction + 12 judgement calls at 120s each
MAX_FACTS = 12
MAX_FACT_CHARS = 320
PROMPT_VERSION = "document-v2"
WINDOW_CHARS = 12000
SUPERSESSION_MIN_SCORE = 0.85  # cosine 0.70: ES cosine score = (1 + cosine) / 2
CACHE = f"{PREFIX}am_fact_cache"
CACHE_MAPPING = {"mappings": {"dynamic": "strict", "properties": {"facts": {"type": "keyword", "ignore_above": 512}}}}

JOB_MAPPING = {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
               "mappings": {"dynamic": "strict", "properties": {
                   "owner_id": {"type": "keyword"}, "source_id": {"type": "keyword"},
                   "state": {"type": "keyword"}, "attempts": {"type": "integer"},
                   "next_at": {"type": "date"}, "lease_until": {"type": "date"},
                   "lease_token": {"type": "keyword"}, "error": {"type": "keyword", "ignore_above": 1024},
                   "created_at": {"type": "date"}, "finished_at": {"type": "date"},
                   "transient_retries": {"type": "integer"}}}}


def _now():
    return dt.datetime.now(dt.timezone.utc)


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


def ensure_queue():
    for index, mapping in ((QUEUE, JOB_MAPPING), (BACKFILLS, BACKFILL_MAPPING)):
        try:
            es("GET", f"/{index}")
        except RuntimeError as exc:
            if "-> 404:" not in str(exc):
                raise
            try:
                es("PUT", f"/{index}", mapping)
            except RuntimeError as race:
                if "-> 400:" not in str(race):
                    raise
    es("PUT", f"/{QUEUE}/_mapping",
       {"properties": {"transient_retries": {"type": "integer"}}})
    try:
        es("GET", f"/{CACHE}")
    except RuntimeError as exc:
        if "-> 404:" not in str(exc):
            raise
        es("PUT", f"/{CACHE}", CACHE_MAPPING)


def enqueue(owner_id, source_id):
    """Create only, deterministic source key. Never requeue completed sources."""
    job_id = hashlib.sha256(f"{owner_id}\0{source_id}".encode()).hexdigest()
    now = _iso(_now())
    try:
        es("PUT", f"/{QUEUE}/_create/{job_id}?refresh=true", {
            "owner_id": owner_id, "source_id": source_id, "state": "pending",
            "attempts": 0, "next_at": now, "created_at": now})
    except RuntimeError as exc:
        if "-> 409:" not in str(exc):
            raise
    return job_id


def status(owner_id):
    """Owner-scoped queue state including terminal failures; never expose other owners."""
    result = es("POST", f"/{QUEUE}/_search", {"size": 0,
        "query": {"term": {"owner_id": owner_id}},
        "aggs": {"states": {"terms": {"field": "state", "size": 10}}}})
    counts = {x["key"]: x["doc_count"] for x in result["aggregations"]["states"]["buckets"]}
    # Compare queued jobs to all retained episodic source groups: enqueue failures
    # must never masquerade as a drained queue.
    sources = es("POST", f"/{idx('episodic')}/_count", {"query": {"bool": {"filter": [
        {"term": {"owner_id": owner_id}}, {"term": {"passage_index": 0}},
        {"term": {"fact_requested": True}}]}}})
    total = sources["count"]
    queued = sum(counts.values())
    missing = max(0, total - queued)
    return {"owner_id": owner_id, "missing_jobs": missing, "pending": counts.get("pending", 0),
            "running": counts.get("running", 0), "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "drained": not (missing or counts.get("pending", 0) or counts.get("running", 0) or counts.get("failed", 0))}


def retry_failed_job(owner_id, job_id):
    """Explicit owner-scoped retry; CAS prevents rearming an active or completed job."""
    current = es("GET", f"/{QUEUE}/_doc/{job_id}")
    doc = current["_source"]
    if doc["owner_id"] != owner_id:
        raise ValueError("job not found for owner")
    if doc["state"] == "pending":
        return {"state": "pending", "job_id": job_id}
    if doc["state"] != "failed":
        raise ValueError("only failed jobs can be retried")
    es("POST", f"/{QUEUE}/_update/{job_id}?if_seq_no={current['_seq_no']}&if_primary_term={current['_primary_term']}&refresh=true",
       {"doc": {"state": "pending", "attempts": 0, "next_at": _iso(_now()),
                "error": "", "finished_at": None, "lease_token": None, "lease_until": None}})
    return {"state": "pending", "job_id": job_id}


def retry_failed_backfill(owner_id, visibility):
    """Resume at the saved cursor, not at the beginning; source jobs remain keyed."""
    if visibility not in ("private", "team", "common"):
        raise ValueError("invalid visibility")
    key = _backfill_id(owner_id, visibility)
    current = es("GET", f"/{BACKFILLS}/_doc/{key}")
    doc = current["_source"]
    if doc["owner_id"] != owner_id or doc["visibility"] != visibility:
        raise ValueError("backfill not found for owner/scope")
    if doc["state"] == "pending":
        return backfill_status(owner_id, visibility)
    if doc["state"] != "failed":
        raise ValueError("only failed backfills can be retried")
    es("POST", f"/{BACKFILLS}/_update/{key}?if_seq_no={current['_seq_no']}&if_primary_term={current['_primary_term']}&refresh=true",
       {"doc": {"state": "pending", "error": "", "finished_at": None,
                "lease_token": None, "lease_until": None}})
    return backfill_status(owner_id, visibility)


def claim(owner_id=None, job_id=None):
    now = _now()
    response = es("POST", f"/{QUEUE}/_search", {"size": 20, "seq_no_primary_term": True,
        "sort": [{"next_at": {"order": "asc", "missing": "_last"}}], "query": {"bool": {"should": [
            {"bool": {"filter": [{"term": {"state": "pending"}}, {"range": {"next_at": {"lte": _iso(now)}}}]}},
            {"bool": {"filter": [{"term": {"state": "running"}}, {"range": {"lease_until": {"lte": _iso(now)}}}]}}
        ], "minimum_should_match": 1,
        **({"filter": {"term": {"_id": job_id}}} if job_id else
           {"filter": {"term": {"owner_id": owner_id}}} if owner_id else {})}}})
    for hit in response["hits"]["hits"]:
        if not job_id:
            # Generic workers must not process jobs from a scope whose backfill
            # owns chronological extraction. The backfill runner claims its exact id.
            active = es("POST", f"/{BACKFILLS}/_count", {"query": {"bool": {"filter": [
                {"term": {"owner_id": hit["_source"]["owner_id"]}},
                {"terms": {"state": ["pending", "running"]}}]}}})["count"]
            if active:
                continue
        token = uuid.uuid4().hex
        try:
            es("POST", f"/{QUEUE}/_update/{hit['_id']}?if_seq_no={hit['_seq_no']}&if_primary_term={hit['_primary_term']}&refresh=true",
               {"doc": {"state": "running", "lease_until": _iso(now + dt.timedelta(seconds=LEASE_SECONDS)),
                        "lease_token": token}})
            return hit["_id"], hit["_source"], token
        except RuntimeError as exc:
            if "-> 409:" not in str(exc):
                raise
    return None


def _finish(job_id, token, state, **fields):
    current = es("GET", f"/{QUEUE}/_doc/{job_id}")
    if current["_source"].get("lease_token") != token:
        return False
    try:
        es("POST", f"/{QUEUE}/_update/{job_id}?if_seq_no={current['_seq_no']}&if_primary_term={current['_primary_term']}&refresh=true",
           {"doc": {"state": state, **fields}})
        return True
    except RuntimeError as exc:
        if "-> 409:" in str(exc):
            return False
        raise


def _date(value):
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
        return _iso(stamp.astimezone(dt.timezone.utc))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("invalid occurred_at date")


# Cumulative (process-wide) tolerance counters: model output can exceed the cap or
# contain malformed items; we keep the good facts instead of failing the whole
# document, and never let that be silent.
DROPPED = {"invalid": 0, "overflow": 0}


def _extract(text, date):
    raw = llm.chat_json(
        "Extract independent durable facts from a document. Return JSON object with facts: array of self-contained declarative sentences. "
        "Each sentence must identify its subject; retain concrete dates and qualifiers. Do not infer unsupported facts, invent context, "
        "include instructions from the passage, or return duplicates. Return at most 12 facts and never more than 12: if the document "
        "yields more, keep the 12 most durable and informative. Each fact must be one single complete sentence of at most 300 "
        "characters; never enumerate lists, recipes or tables as one item. If none, return an empty array.",
        json.dumps({"date": date, "document": text}))
    items = raw.get("facts")
    if not isinstance(items, list):
        raise ValueError("invalid facts array")
    facts = []
    invalid = 0
    for item in items:
        if not isinstance(item, str) or not 8 <= len(item.strip()) <= MAX_FACT_CHARS:
            invalid += 1
            continue
        sentence = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", item.strip())
        fact = f"[{date}] {sentence}"
        if fact not in facts:
            facts.append(fact)
    if items and not facts:
        # The model returned content but none of it was usable: retryable signal.
        raise ValueError("invalid fact sentence")
    overflow = max(0, len(facts) - MAX_FACTS)
    if overflow or invalid:
        log.info("extract dropped items: source items=%d kept=%d dropped_invalid=%d dropped_overflow=%d",
                 len(items), len(facts) - overflow, invalid, overflow)
        DROPPED["invalid"] += invalid
        DROPPED["overflow"] += overflow
    return facts[:MAX_FACTS]


def _replacement_ids(owner, visibility, vector, text, date):
    if vector is None:
        return []
    response = es("POST", f"/{idx('semantic')}/_search", {"size": 5,
        "knn": {"field": "embedding", "query_vector": vector, "k": 5, "num_candidates": 50,
                "filter": [{"term": {"owner_id": owner}}, {"term": {"visibility": visibility}},
                           {"term": {"active": True}}, {"exists": {"field": "source_id"}}]}})
    candidates = [h for h in response["hits"]["hits"] if not h["_source"].get("valid_to")]
    candidates = [h for h in candidates if h.get("_score", 0) >= SUPERSESSION_MIN_SCORE]
    if not candidates:
        return []
    result = llm.chat_json(
        "Determine which prior facts are actually replaced or contradicted by the new fact. Related or repeated facts are not replacements. "
        "Return JSON object {\"replace_ids\": [ids]}. Choose only given IDs; return [] if uncertain.",
        json.dumps({"new": text, "effective_date": date,
                    "candidates": [{"id": h["_id"], "text": h["_source"]["text"],
                                    "valid_from": h["_source"].get("valid_from")} for h in candidates]}))
    ids = result.get("replace_ids")
    if not isinstance(ids, list):
        raise ValueError("invalid replacement ids")
    allowed = {h["_id"] for h in candidates if h["_source"].get("valid_from", "") <= date}
    # The judge may echo ids it was never given (hallucination) or a future-dated
    # fact. Dropping the whole document hard-fails the job deterministically (4
    # retries -> scope aborted); instead keep the valid subset and log the rest.
    kept = []
    for i in ids:
        if isinstance(i, str) and i in allowed:
            kept.append(i)
        else:
            reason = "non-str" if not isinstance(i, str) else (
                "future" if isinstance(i, str) and any(
                    h["_id"] == i and h["_source"].get("valid_from", "") > date
                    for h in candidates) else "unknown")
            log.warning("supersession dropped id: fact=%r id=%r reason=%s",
                        text[:80], i, reason)
    return list(dict.fromkeys(kept))


def _document_facts(passages, date):
    """One extraction per document when it fits; only split oversized documents at passage boundaries."""
    windows = []
    current = ""
    for _, passage in passages:
        for part in (passage[i:i + WINDOW_CHARS] for i in range(0, len(passage), WINDOW_CHARS)):
            if current and len(current) + len(part) + 1 > WINDOW_CHARS:
                windows.append(current)
                current = ""
            current = f"{current}\n{part}" if current else part
    if current:
        windows.append(current)
    facts = []
    for window in windows:
        key = hashlib.sha256(json.dumps([window, date, PROMPT_VERSION, llm.MODEL]).encode()).hexdigest()
        try:
            cached = es("GET", f"/{CACHE}/_doc/{key}")["_source"]["facts"]
        except RuntimeError as exc:
            if "-> 404:" not in str(exc):
                raise
            cached = _extract(window, date)
            try:
                es("PUT", f"/{CACHE}/_create/{key}", {"facts": cached})
            except RuntimeError as conflict:
                if "-> 409:" not in str(conflict):
                    raise
        facts.extend(cached)
    return list(dict.fromkeys(facts))


def _passages(source_id, source):
    group = source.get("doc_group") or source_id
    response = es("POST", f"/{idx('episodic')}/_search", {"size": 1000,
        "query": {"bool": {"filter": [{"term": {"owner_id": source['owner_id']}},
            {"term": {"doc_group": group}}, {"term": {"visibility": source['visibility']}}]}},
        "sort": [{"passage_index": "asc"}]})
    hits = response["hits"]["hits"]
    if not hits or len(hits) != source.get("passages_total", 1):
        raise ValueError("incomplete episodic document")
    return [(h["_id"], h["_source"]["text"]) for h in hits]


def process(source_id, owner_id):
    source = es("GET", f"/{idx('episodic')}/_doc/{source_id}")["_source"]
    if source["owner_id"] != owner_id or not source.get("active", True):
        raise ValueError("source missing, inactive, or owner mismatch")
    date = _date(source["occurred_at"])
    passages = _passages(source_id, source)
    for fact in _document_facts(passages, date):
        fact_id = hashlib.sha256(f"{owner_id}\0{source_id}\0{fact}".encode()).hexdigest()
        # Retry after a partial failure must finish any missing history links.
        existing = None
        try:
            existing = es("GET", f"/{idx('semantic')}/_doc/{fact_id}")["_source"]
        except RuntimeError as exc:
            if "-> 404:" not in str(exc):
                raise
        if existing is not None:
            replacements = existing.get("supersedes", [])
            if existing.get("doc_group") != source.get("doc_group", source_id) or existing.get("owner_id") != owner_id:
                raise ValueError("fact id collision or source mismatch")
        else:
            vector = embeddings.embed([fact])[0]
            replacements = _replacement_ids(owner_id, source["visibility"], vector, fact, date)
            # Select the closest passage by lexical overlap; keep provenance even when
            # the sentence paraphrases the source and every overlap is zero.
            terms = set(re.findall(r"\w+", fact.lower()))
            passage_id = max(passages, key=lambda p: len(terms & set(re.findall(r"\w+", p[1].lower()))))[0]
            doc = {"kind": "semantic", "owner_id": owner_id, "visibility": source["visibility"],
                   "text": fact, "embedding": vector, "source_id": passage_id,
                   "entities": [], "occurred_at": source["occurred_at"], "active": True,
                   "valid_from": date, "doc_group": source.get("doc_group", source_id),
                   "supersedes": replacements}
            try:
                es("PUT", f"/{idx('semantic')}/_create/{fact_id}?refresh=true", doc)
            except RuntimeError as exc:
                if "-> 409:" not in str(exc):
                    raise
        for old_id in replacements:
            # Script is idempotent and guarded against racing updates to historical intervals.
            es("POST", f"/{idx('semantic')}/_update/{old_id}?refresh=true", {"script": {
                "source": "if (ctx._source.owner_id == params.owner && ctx._source.visibility == params.scope && (ctx._source.valid_to == null || ctx._source.superseded_by == params.newid)) { ctx._source.valid_to = params.date; ctx._source.superseded_by = params.newid; }",
                "params": {"owner": owner_id, "scope": source["visibility"], "newid": fact_id, "date": date}}})


def run_once(owner_id=None, job_id=None):
    claimed = claim(owner_id, job_id) if owner_id or job_id else claim()
    if claimed is None:
        return False
    job_id, job, token = claimed
    try:
        process(job["source_id"], job["owner_id"])
        _finish(job_id, token, "completed", finished_at=_iso(_now()), error="")
    except Exception as exc:
        if http_retry.is_transient(exc):
            # Transient upstream failure: requeue WITHOUT consuming an attempt.
            transient = job.get("transient_retries", 0) + 1
            if job.get("transient_retries", 0) >= http_retry.MAX_TRANSIENT_RETRIES:
                log.exception("fact extraction job %s transient retry %d failed",
                              job_id, transient)
                _finish(job_id, token, "failed", error=str(exc)[:900],
                        transient_retries=transient, finished_at=_iso(_now()))
            else:
                log.warning("fact extraction job %s transient failure (%s), "
                            "requeue %d/%d without consuming an attempt",
                            job_id, type(exc).__name__, transient, http_retry.MAX_TRANSIENT_RETRIES)
                _finish(job_id, token, "pending", error=str(exc)[:900],
                        transient_retries=transient,
                        next_at=_iso(_now() + dt.timedelta(
                            seconds=min(900, 60 * 2 ** transient))))
            return True
        attempts = job["attempts"] + 1
        state = "failed" if attempts >= MAX_ATTEMPTS else "pending"
        log.exception("fact extraction job %s attempt %d %s", job_id, attempts, state)
        _finish(job_id, token, state, attempts=attempts, error=str(exc)[:900],
                next_at=_iso(_now() + dt.timedelta(seconds=min(300, 2 ** attempts * 5))),
                **({"finished_at": _iso(_now())} if state == "failed" else {}))
    return True


def loop(owner_id=None):
    from .store import ensure_indices
    ensure_indices()
    ensure_queue()
    while True:
        try:
            if not run_once(owner_id):
                time.sleep(2)
        except Exception:
            log.exception("fact worker poll failed")
            time.sleep(5)


def _backfill_id(owner_id, visibility):
    return hashlib.sha256(f"{owner_id}\0{visibility}".encode()).hexdigest()


def backfill(owner_id, visibility):
    """Durably schedule a scope; repeated calls return the same job."""
    if visibility not in ("private", "team", "common"):
        raise ValueError("invalid visibility")
    key = _backfill_id(owner_id, visibility)
    try:
        es("PUT", f"/{BACKFILLS}/_create/{key}?refresh=true", {
            "owner_id": owner_id, "visibility": visibility, "state": "pending",
            "queued": 0, "created_at": _iso(_now()), "error": ""})
    except RuntimeError as exc:
        if "-> 409:" not in str(exc):
            raise
    return backfill_status(owner_id, visibility)


def backfill_status(owner_id, visibility):
    try:
        doc = es("GET", f"/{BACKFILLS}/_doc/{_backfill_id(owner_id, visibility)}")["_source"]
    except RuntimeError as exc:
        if "-> 404:" not in str(exc):
            raise
        return None
    return {field: doc.get(field) for field in
            ("owner_id", "visibility", "state", "queued", "error", "created_at", "finished_at")}


def _update_backfill(key, token, **fields):
    current = es("GET", f"/{BACKFILLS}/_doc/{key}")
    if current["_source"].get("lease_token") != token or current["_source"]["state"] != "running":
        raise RuntimeError("backfill lease lost")
    es("POST", f"/{BACKFILLS}/_update/{key}?if_seq_no={current['_seq_no']}&if_primary_term={current['_primary_term']}&refresh=true",
       {"doc": fields})


def _heartbeat(key, token, stop):
    while not stop.wait(30):
        try:
            _update_backfill(key, token, lease_until=_iso(_now() + dt.timedelta(seconds=LEASE_SECONDS)))
        except Exception:
            log.exception("backfill lease renewal failed: %s", key)
            return


def claim_backfill():
    now = _iso(_now())
    hits = es("POST", f"/{BACKFILLS}/_search", {"size": 30, "seq_no_primary_term": True,
        "sort": [{"created_at": "asc"}], "query": {"bool": {"should": [
            {"term": {"state": "pending"}},
            {"bool": {"filter": [{"term": {"state": "running"}},
                                  {"range": {"lease_until": {"lte": now}}}]}}],
            "minimum_should_match": 1}}})["hits"]["hits"]
    for hit in hits:
        token = uuid.uuid4().hex
        try:
            es("POST", f"/{BACKFILLS}/_update/{hit['_id']}?if_seq_no={hit['_seq_no']}&if_primary_term={hit['_primary_term']}&refresh=true",
               {"doc": {"state": "running", "lease_token": token,
                        "lease_until": _iso(_now() + dt.timedelta(seconds=LEASE_SECONDS))}})
            return hit["_id"], hit["_source"], token
        except RuntimeError as exc:
            if "-> 409:" not in str(exc):
                raise
    return None


def _run_backfill(key, doc, token):
    owner_id, visibility = doc["owner_id"], doc["visibility"]
    after = json.loads(doc["cursor"]) if doc.get("cursor") else None
    queued = doc.get("queued", 0)
    while True:
        body = {"size": 1, "query": {"bool": {"filter": [
            {"term": {"owner_id": owner_id}}, {"term": {"visibility": visibility}},
            {"term": {"passage_index": 0}}, {"term": {"active": True}}]}},
            "sort": [{"occurred_at": "asc"}, {"doc_group": "asc"}]}
        if after is not None:
            body["search_after"] = after
        hits = es("POST", f"/{idx('episodic')}/_search", body)["hits"]["hits"]
        if not hits:
            _update_backfill(key, token, state="completed", finished_at=_iso(_now()))
            return
        hit = hits[0]
        job_id = enqueue(owner_id, hit["_id"])
        es("POST", f"/{idx('episodic')}/_update/{hit['_id']}?refresh=true",
           {"doc": {"fact_requested": True}})
        while True:
            job = es("GET", f"/{QUEUE}/_doc/{job_id}")["_source"]
            if job["state"] == "completed":
                break
            if job["state"] == "failed":
                raise RuntimeError(f"backfill job failed: {job_id}: {job.get('error')}")
            now = _iso(_now())
            if job["state"] == "pending" and job["next_at"] <= now:
                run_once(owner_id, job_id)
            elif job["state"] == "running" and job.get("lease_until", "") <= now:
                # orphaned running job (worker died mid-lease, e.g. a rollout):
                # claim() re-runs expired leases by id; without this the scope
                # sleeps forever while generic workers skip the active scope.
                run_once(owner_id, job_id)
            else:
                time.sleep(2)
        after = hit["sort"]
        queued += 1
        _update_backfill(key, token, cursor=json.dumps(after), queued=queued)


def run_backfill_once():
    claimed = claim_backfill()
    if claimed is None:
        return False
    key, doc, token = claimed
    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat, args=(key, token, stop), daemon=True)
    heartbeat.start()
    try:
        _run_backfill(key, doc, token)
    except Exception as exc:
        log.exception("backfill %s failed", key)
        try:
            _update_backfill(key, token, state="failed", error=str(exc)[:900], finished_at=_iso(_now()))
        except Exception:
            log.exception("could not record backfill failure %s; lease will expire", key)
    finally:
        stop.set()
        heartbeat.join(timeout=5)
    return True


def backfill_loop():
    while True:
        try:
            if not run_backfill_once():
                time.sleep(2)
        except Exception:
            log.exception("backfill worker poll failed")
            time.sleep(5)

