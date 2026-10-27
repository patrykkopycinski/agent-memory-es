#!/usr/bin/env python3
"""Rerank latency x quality sweep against a DEDICATED Elasticsearch (never prod).

  AMES_ES_URL=http://127.0.0.1:19281 AMES_EMBED_BACKEND=es AMES_INDEX_PREFIX=amtest_ \
      python scripts/bench_rerank.py seed  [--docs 12000] [--profile short|long]
  ... python scripts/bench_rerank.py sweep [--profile short|long] [--depths 10,20,30,50]
                                           [--threads 1,2,4] [--allocs 1,2] [--queries 40] [--out f.json]

Profiles: `short` = fact-sized docs (~250 chars); `long` = passage-sized docs (~1800 chars, the
retain() chunk target), the worst case for a cross-encoder whose cost scales with tokens.

The corpus is synthetic (service/attribute facts + filler, built from word lists below) so the
sweep never touches a benchmark dataset; queries are generated from the same lists with a fixed
seed. Quality is reported, never optimised against: the sweep records the numbers, a human picks
the config from latency + the quality delta.

Caveats learned the hard way: ES caches rerank results per (query, passages), so each measured
cell uses its own query slice; and the built-in `.multilingual-e5-small-elasticsearch` endpoint
adaptively scales (it reached 7 allocations during bulk embedding) and holds the node's ML
processors, so a new rerank endpoint gets HTTP 429 until it scales back down (~3-4 min idle).
Seed first, wait for e5 to settle, then sweep.

Per deployment a throwaway inference endpoint `bench-t<threads>a<allocs>` is created, warmed up,
measured, and deleted, so only one trained-model deployment competes for CPU at a time.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

ES = os.environ.get("AMES_ES_URL", "")
OWNER = "benchrr"
PAD_CHARS = {"short": 0, "long": 1800}
SEED = 20261012


def _guard():
    if not ES or ":9268" in ES:
        sys.exit("refusing: AMES_ES_URL must point at a dedicated ES (not unset, not :9268 prod)")
    if os.environ.get("AMES_INDEX_PREFIX") != "amtest_":
        sys.exit("refusing: AMES_INDEX_PREFIX must be amtest_")


_guard()
from app import embeddings, memory, reranker  # noqa: E402
from app.store import es, ensure_indices, idx  # noqa: E402

SERVICES = ["billing", "catalog", "checkout", "search", "ingest", "auth", "gateway", "notify", "report",
            "sync", "export", "scheduler", "inventory", "pricing", "shipping", "returns", "ledger", "audit",
            "profile", "media"]
SUFFIX = ["alpha", "beta", "gamma", "delta", "sigma", "omega", "north", "south", "east", "west",
          "core", "edge", "prime", "lite", "batch"]
ATTRS = {
    "deploy window": ["Tuesday 09:00", "Wednesday 14:00", "Thursday 03:00", "Monday 22:00", "Friday 11:00"],
    "on-call rotation": ["weekly handoff on Monday", "follow-the-sun across three regions",
                         "two-person pairing", "daily handoff at 08:00"],
    "storage tier": ["hot ssd", "warm hdd", "cold object store", "tiered with weekly compaction"],
    "retention policy": ["30 days", "90 days", "one year", "seven years for compliance"],
    "rollback strategy": ["blue-green swap", "canary with automatic revert", "manual snapshot restore"],
    "alert threshold": ["p95 latency above 800 ms", "error rate above 2 percent", "queue depth above 5000"],
    "owning team": ["platform", "payments", "growth", "data", "infrastructure", "security"],
    "scaling limit": ["40 replicas", "12 replicas", "200 concurrent jobs", "64 partitions"],
}
FILLER_OPENERS = ["Meeting notes:", "Reminder:", "Decision log:", "Postmortem summary:", "Planning note:",
                  "Design review:", "Retro item:", "Support ticket:"]
FILLER_TOPICS = ["the migration of the reporting warehouse", "an upgrade of the build runners",
                 "onboarding for new contractors", "the quarterly capacity forecast",
                 "a certificate rotation", "the vendor contract renewal", "a flaky integration suite",
                 "the office network refresh", "cost allocation tags", "a documentation clean-up",
                 "the data retention review", "a dependency audit"]
FILLER_TAILS = ["was discussed and the group agreed to revisit after the next release.",
                "stalled because the owner was out and no backup had been named.",
                "finished ahead of schedule with no customer impact reported.",
                "needs a follow-up ticket before the end of the month.",
                "was deferred until budget approval arrives from finance.",
                "produced three action items that were assigned in the tracker."]


def _services():
    return ["%s-%s" % (s, x) for s in SERVICES for x in SUFFIX]            # 300 services


def _pad(text: str, target: int, rng: random.Random) -> str:
    while len(text) < target:
        text += " %s %s %s" % (rng.choice(FILLER_OPENERS), rng.choice(FILLER_TOPICS), rng.choice(FILLER_TAILS))
    return text


def build_corpus(n_docs: int, rng: random.Random, pad: int = 0):
    """-> (docs, queries). Each fact doc is the single correct answer for its query; near-duplicate
    facts for sibling services/attributes are the distractors the cross-encoder must separate."""
    docs = []
    for svc in _services():
        for attr, values in ATTRS.items():
            v = rng.choice(values)
            text = ("The %s of the %s service is %s. This was confirmed by the owning engineers during the "
                    "last operations review and is recorded here as the current agreed setting." % (attr, svc, v))
            docs.append({"text": _pad(text, pad, rng), "svc": svc, "attr": attr})
    facts = list(docs)
    while len(docs) < n_docs:
        text = "%s %s %s Reference %05d." % (
            rng.choice(FILLER_OPENERS), rng.choice(FILLER_TOPICS), rng.choice(FILLER_TAILS), len(docs))
        docs.append({"text": _pad(text, pad, rng)})
    queries = []
    for d in rng.sample(facts, min(len(facts), 2400)):
        q = rng.choice(["what is the %(a)s of the %(s)s service",
                        "%(s)s service %(a)s",
                        "tell me the current %(a)s for %(s)s",
                        "how is the %(a)s set for the %(s)s service"]) % {"a": d["attr"], "s": d["svc"]}
        queries.append({"q": q, "target_text": d["text"]})
    rng.shuffle(docs)
    return docs, queries


def cmd_seed(a):
    ensure_indices()
    rng = random.Random(SEED)
    docs, _ = build_corpus(a.docs, rng, PAD_CHARS[a.profile])
    es("POST", "/%s/_delete_by_query?refresh=true" % idx("semantic"), {"query": {"term": {"owner_id": OWNER}}})
    t0 = time.time()
    B = 64
    for i in range(0, len(docs), B):
        chunk = docs[i:i + B]
        vecs = embeddings.embed([d["text"] for d in chunk])
        lines = []
        for d, v in zip(chunk, vecs):
            h = memory.text_hash(d["text"])
            lines.append(json.dumps({"index": {"_index": idx("semantic")}}))
            lines.append(json.dumps({"kind": "semantic", "owner_id": OWNER, "visibility": "private",
                                     "text": d["text"], "text_hash": h, "doc_group": "dg_" + h[:16],
                                     "passage_index": 0, "passages_total": 1,
                                     "entities": memory.extract_entities(d["text"]),
                                     "occurred_at": "2026-09-01T00:00:00Z", "active": True, "embedding": v}))
        req = urllib.request.Request(ES + "/_bulk", data=("\n".join(lines) + "\n").encode(), method="POST",
                                     headers={"Content-Type": "application/x-ndjson"})
        r = json.load(urllib.request.urlopen(req, timeout=120))
        if r.get("errors"):
            sys.exit("bulk errors: %s" % json.dumps(r)[:400])
        if (i // B) % 20 == 0:
            print("seeded %d/%d (%.0fs)" % (i + len(chunk), len(docs), time.time() - t0), flush=True)
    es("POST", "/%s/_refresh" % idx("semantic"))
    n = es("POST", "/%s/_count" % idx("semantic"), {"query": {"term": {"owner_id": OWNER}}})["count"]
    print("seeded owner=%s docs=%d in %.0fs" % (OWNER, n, time.time() - t0))


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def _ids(res):
    return [h["id"] for h in res["results"]]


def _one(q, depth, rerank_on):
    t = time.perf_counter()
    r = memory.recall(OWNER, q, kinds=["semantic"], size=8, rerank=rerank_on,
                      rerank_depth=depth if rerank_on else None)
    return (time.perf_counter() - t) * 1000.0, r


def _stats(ms):
    return {"p50": round(_pct(ms, 50)), "p95": round(_pct(ms, 95)), "max": round(max(ms)),
            "mean": round(statistics.mean(ms))}


def _hit_rank(res, target_text):
    for i, h in enumerate(res["results"]):
        if h.get("text") == target_text:
            return i + 1
    return None


def measure(queries, depth, rerank_on, base, conc=1):
    lat, overlap, ranks, applied, reasons = [], [], [], 0, {}

    def work(i):
        ms, r = _one(queries[i]["q"], depth, rerank_on)
        return i, ms, r
    with ThreadPoolExecutor(max_workers=conc) as ex:
        out = list(ex.map(work, range(len(queries))))
    for i, ms, r in out:
        lat.append(ms)
        rr = r.get("rerank")
        if rerank_on:
            if rr and rr["applied"]:
                applied += 1
            else:
                k = (rr or {}).get("fallback_reason", "none")
                reasons[k] = reasons.get(k, 0) + 1
        if base is not None:
            overlap.append(len(set(_ids(r)) & set(_ids(base[i]))) / 8.0)
        ranks.append(_hit_rank(r, queries[i]["target_text"]))
    n = len(queries)
    mrr = sum((1.0 / x) if x else 0.0 for x in ranks) / n
    row = {"depth": depth, "rerank": rerank_on, "conc": conc, "n": n, "latency_ms": _stats(lat),
           "applied": applied, "fallbacks": reasons, "hit@1": round(sum(1 for x in ranks if x == 1) / n, 3),
           "hit@8": round(sum(1 for x in ranks if x) / n, 3), "mrr@8": round(mrr, 3)}
    if overlap:
        row["top8_overlap_vs_off"] = round(statistics.mean(overlap), 3)
    return row, [r for _, _, r in out]


def _put_endpoint(name, threads, allocs):
    body = {"service": "elasticsearch", "service_settings": {
        "model_id": ".rerank-v1", "num_threads": threads, "num_allocations": allocs}}
    req = urllib.request.Request(ES + "/_inference/rerank/" + name, data=json.dumps(body).encode(), method="PUT",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=300).read()


def _del_endpoint(name):
    req = urllib.request.Request(ES + "/_inference/rerank/%s?force=true" % name, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=120).read()
    except urllib.error.HTTPError:
        pass


def cmd_sweep(a):
    rng = random.Random(SEED)
    _, queries = build_corpus(a.docs, rng, PAD_CHARS[a.profile])
    reranker.RERANK_TIMEOUT = a.timeout          # measure the real latency, not the 2 s production cut
    cells = len(a.threads) * len(a.allocs) * (len(a.depths) + 1)
    need = a.queries * (cells + 1) + 10
    if len(queries) < need:
        sys.exit("need %d distinct queries (%d cells x %d + warmup), corpus has %d" % (need, cells, a.queries, len(queries)))
    # ES caches rerank results per (query, passages): every measured cell gets its OWN slice of
    # queries so no call is served from that cache and every latency is a real inference.
    cursor = [10]

    def take():
        sl = queries[cursor[0]:cursor[0] + a.queries]
        cursor[0] += a.queries
        return sl
    res = {"es": ES, "profile": a.profile, "docs": a.docs, "queries_per_cell": a.queries,
           "timeout_s": a.timeout, "rows": []}
    for q in queries[:10]:
        _one(q["q"], None, False)                # warm ES caches / e5
    off_row, _ = measure(take(), None, False, None)
    off_row["config"] = "rerank-off"
    res["rows"].append(off_row)
    print(json.dumps(off_row), flush=True)

    def cell(qs, depth, conc):
        _, base = measure(qs, None, False, None)
        row, _ = measure(qs, depth, True, base, conc=conc)
        return row
    for threads in a.threads:
        for allocs in a.allocs:
            name = "bench-t%da%d" % (threads, allocs)
            try:
                _put_endpoint(name, threads, allocs)
            except Exception as e:          # e.g. not enough ML processors on this node
                row = {"config": name, "error": str(e)[:200]}
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
                continue
            os.environ["AMES_RERANK_MODEL"] = name
            t = time.perf_counter()
            _one(queries[0]["q"], 50, True)  # cold call, reported separately
            cold = round((time.perf_counter() - t) * 1000)
            for q in queries[1:4]:
                _one(q["q"], 50, True)
            for depth in a.depths:
                row = cell(take(), depth, 1)
                row.update(config=name, threads=threads, allocs=allocs, first_call_ms=cold)
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
            row = cell(take(), a.conc_depth, 4)
            row.update(config=name, threads=threads, allocs=allocs, first_call_ms=cold)
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
            _del_endpoint(name)
    os.environ.pop("AMES_RERANK_MODEL", None)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--docs", type=int, default=12000)
    s.add_argument("--profile", choices=sorted(PAD_CHARS), default="short")
    w = sub.add_parser("sweep")
    w.add_argument("--docs", type=int, default=12000)
    w.add_argument("--profile", choices=sorted(PAD_CHARS), default="short")
    w.add_argument("--depths", type=lambda x: [int(i) for i in x.split(",")], default=[10, 20, 30, 50])
    w.add_argument("--threads", type=lambda x: [int(i) for i in x.split(",")], default=[1, 2, 4])
    w.add_argument("--allocs", type=lambda x: [int(i) for i in x.split(",")], default=[1, 2])
    w.add_argument("--queries", type=int, default=20)
    w.add_argument("--timeout", type=float, default=120.0)
    w.add_argument("--conc-depth", type=int, default=30)
    w.add_argument("--out", default="rerank_sweep.json")
    a = p.parse_args()
    {"seed": cmd_seed, "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    main()
