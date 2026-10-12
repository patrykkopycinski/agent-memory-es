#!/usr/bin/env python3
"""Replay recall queries against an AMES backend and report latency + context cost.

Runs a 2x2 matrix: rerank {off, on} x filter {none, given}. For every cell the first
--warmup queries are sent but excluded from the statistics. Per cell it reports p50/p95/p99
of the server-side `timings_ms.total` and `timings_ms.es_took`, the median `context_chars`,
the abstain rate, and (rerank-on cells) how many calls fell back to the fused order because
the rerank endpoint failed or timed out -- without that column a "rerank on" row is just the
fallback path.

Usage:
  recall_latency_bench.py --url http://127.0.0.1:8123 --key ame_... --queries q.jsonl \
      --filter '{"any": [["topic:alpha"]]}' [--warmup 10] [--repeat 1] [--size 8] [--out r.json]

q.jsonl: one JSON value per line, either a string or {"query": "...", "filter": {...}}.
A per-line "filter" overrides --filter in the filter=given cells. With neither, the filtered
cells are skipped and reported as such.

Percentiles use the nearest-rank method (value at rank ceil(p/100*n)); with n < 100 samples
p99 equals the max, and the output says so via "n".
Stdlib only. Never prints query text.
"""
import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(s)))
    return s[rank - 1]


def load_queries(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            v = json.loads(ln)
            out.append({"query": v} if isinstance(v, str) else v)
    if not out:
        raise SystemExit("no queries in %s" % path)
    return out


def post_recall(url, key, body, timeout):
    req = urllib.request.Request(
        url.rstrip("/") + "/memory/recall", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": key})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.load(r)
    return payload, (time.perf_counter() - t0) * 1000.0


def run_cell(args, queries, rerank, use_filter):
    samples, errors = [], 0
    seq = queries * max(1, args.repeat)
    for i, q in enumerate(seq):
        body = {"query": q["query"], "size": args.size, "rerank": rerank}
        if use_filter:
            flt = q.get("filter") or args.filter
            body["filter"] = flt
        try:
            payload, wall = post_recall(args.url, args.key, body, args.timeout)
        except (urllib.error.URLError, OSError, ValueError) as e:
            if i >= args.warmup:
                errors += 1
            continue
        if i < args.warmup:
            continue
        t = payload.get("timings_ms") or {}
        samples.append({
            "total": t.get("total"), "es_took": t.get("es_took"), "wall": wall,
            "context_chars": payload.get("context_chars"),
            "abstained": bool(payload.get("abstained")),
            "reranked": bool(payload.get("reranked")),
            "has_timings": "total" in t,
        })
    return summarize(samples, errors, rerank)


def stats(values):
    vals = [v for v in values if v is not None]
    return {"n": len(vals), "p50": percentile(vals, 50), "p95": percentile(vals, 95),
            "p99": percentile(vals, 99)}


def summarize(samples, errors, rerank):
    n = len(samples)
    cell = {
        "n": n, "errors": errors,
        "total_ms": stats([s["total"] for s in samples]),
        "es_took_ms": stats([s["es_took"] for s in samples]),
        "wall_ms": stats([s["wall"] for s in samples]),
        "median_context_chars": (statistics.median([s["context_chars"] for s in samples
                                                    if s["context_chars"] is not None])
                                 if any(s["context_chars"] is not None for s in samples) else None),
        "abstain_rate": (sum(s["abstained"] for s in samples) / n) if n else None,
    }
    if rerank:
        fb = sum(1 for s in samples if not s["reranked"])
        cell["rerank_fallbacks"] = fb
        cell["rerank_fallback_rate"] = (fb / n) if n else None
    if any(not s["has_timings"] for s in samples):
        cell["warning"] = "server returned no timings_ms for some responses (old backend?)"
    return cell


def fmt(x):
    return "-" if x is None else ("%.1f" % x if isinstance(x, float) else str(x))


def one_line(result):
    parts = []
    for name, c in result["cells"].items():
        if c.get("skipped"):
            parts.append("%s: skipped" % name)
            continue
        extra = " fb=%s" % c["rerank_fallbacks"] if "rerank_fallbacks" in c else ""
        parts.append("%s: p50/p95/p99 total=%s/%s/%s es_took=%s/%s/%s ctx=%s abst=%s%s" % (
            name, fmt(c["total_ms"]["p50"]), fmt(c["total_ms"]["p95"]), fmt(c["total_ms"]["p99"]),
            fmt(c["es_took_ms"]["p50"]), fmt(c["es_took_ms"]["p95"]), fmt(c["es_took_ms"]["p99"]),
            fmt(c["median_context_chars"]), fmt(c["abstain_rate"]), extra))
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="AMES backend base URL")
    ap.add_argument("--key", required=True, help="X-API-Key of the owner whose memory is queried")
    ap.add_argument("--queries", required=True, help="JSONL file of queries")
    ap.add_argument("--filter", type=json.loads, default=None,
                    help="JSON tag filter for the filter=given cells")
    ap.add_argument("--warmup", type=int, default=10, help="leading calls per cell excluded from stats")
    ap.add_argument("--repeat", type=int, default=1, help="replay the query list this many times per cell")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=90.0, help="client timeout per call, seconds")
    ap.add_argument("--out", help="also write the JSON result here")
    args = ap.parse_args(argv)

    queries = load_queries(args.queries)
    have_filter = args.filter is not None or any(q.get("filter") for q in queries)
    cells = {}
    for rerank in (False, True):
        for use_filter in (False, True):
            name = "rerank_%s/filter_%s" % ("on" if rerank else "off", "given" if use_filter else "none")
            if use_filter and not have_filter:
                cells[name] = {"skipped": "no --filter and no per-query filter"}
                continue
            cells[name] = run_cell(args, queries, rerank, use_filter)
    result = {"queries": len(queries), "repeat": args.repeat, "warmup": args.warmup,
              "size": args.size, "percentile_method": "nearest-rank", "cells": cells}
    result["summary"] = one_line(result)
    text = json.dumps(result, indent=2)
    print(text)
    print(result["summary"])
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
