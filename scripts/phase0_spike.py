#!/usr/bin/env python3
"""Phase 0 spike: index templates, retain/recall, leak test, supersession.
Run on a self-hosted host against am-es-spike (ES 9.6.0-SNAPSHOT, :9268)."""
import json
import urllib.error
import urllib.request

ES = "http://localhost:9268"

def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(ES + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(r))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:500]}

MAP = {
  "settings": {"number_of_shards": 1, "number_of_replicas": 0},
  "mappings": {
    "dynamic": "strict",
    "properties": {
      "kind":        {"type": "keyword"},           # episodic|semantic|procedural
      "owner_id":    {"type": "keyword"},
      "visibility":  {"type": "keyword"},           # private|team|common
      "text":        {"type": "text"},              # BM25 arm
      "occurred_at": {"type": "date"},
      "superseded_by":{"type": "keyword"},
      "supersedes":  {"type": "keyword"},
      "active":      {"type": "boolean"}
    }
  }
}

def setup():
    for k in ("episodic", "semantic", "procedural"):
        r = req("PUT", f"/am_{k}", MAP)
        print(k, r.get("acknowledged", r))

def retain(idx, doc):
    return req("POST", f"/am_{idx}/_doc", doc)

def recall(idx, q, owner):
    # visibility filter + BM25 (dense arm deferred: needs inference endpoint)
    body = {
      "size": 5,
      "query": {"bool": {"must": {"match": {"text": q}},
                         "filter": [
                            {"term": {"active": True}},
                            {"bool": {"should": [
                                {"term": {"owner_id": owner}},
                                {"bool": {"must_not": {"term": {"visibility": "private"}}}}
                            ]}}
                         ]}}
    }
    return req("POST", f"/am_{idx}/_search", body)

def main():
    setup()
    # private doc for alice
    retain("semantic", {"kind": "semantic", "owner_id": "alice", "visibility": "private",
        "text": "Alice's production OmniRoute admin token rotates every 14 days",
        "occurred_at": "2026-10-01T09:00:00Z", "active": True})
    # common doc
    retain("semantic", {"kind": "semantic", "owner_id": "bob", "visibility": "common",
        "text": "Team convention: evals always run on Azure VMs via suite_runner.py",
        "occurred_at": "2026-10-01T09:01:00Z", "active": True})
    import time; time.sleep(1)

    # LEAK TEST: bob recalls "OmniRoute admin token"
    bob = recall("semantic", "OmniRoute admin token", "bob")
    hits = bob["hits"]["hits"]
    private_leak = any(h["_source"]["visibility"] == "private" for h in hits)
    print("bob sees:", [(h["_source"]["visibility"], h["_source"]["text"][:40]) for h in hits])
    print("LEAK TEST:", "FAIL — private doc visible to bob" if private_leak else "PASS — bob sees no private docs")

    # alice recalls her own
    alice = recall("semantic", "OmniRoute admin token", "alice")
    print("alice own-recall hits:", alice["hits"]["total"]["value"])

    # SHARED TEST: bob recalls "evals Azure suite_runner"
    shared = recall("semantic", "evals Azure suite_runner", "bob")
    ok = any("suite_runner" in h["_source"]["text"] for h in shared["hits"]["hits"])
    print("SHARED TEST:", "PASS — bob recalls common doc" if ok else "FAIL")

    # SUPERSESSION: old fact superseded, history kept
    old = retain("semantic", {"kind": "semantic", "owner_id": "alice", "visibility": "private",
        "text": "Alice uses React for frontend work", "occurred_at": "2025-01-01T00:00:00Z", "active": True})
    time.sleep(0.5)
    old_id = old["_id"]
    new = retain("semantic", {"kind": "semantic", "owner_id": "alice", "visibility": "private",
        "text": "Alice now uses Vue for frontend work", "occurred_at": "2026-10-01T00:00:00Z",
        "active": True, "supersedes": old_id})
    req("POST", f"/am_semantic/_update/{old_id}", {"doc": {"active": False, "superseded_by": new["_id"]}})
    time.sleep(1)
    hist = req("GET", f"/am_semantic/_doc/{old_id}")
    print("SUPERSESSION:", "PASS — history kept" if (not hist["_source"]["active"] and hist["_source"]["superseded_by"] == new["_id"]) else "FAIL")
    cur = recall("semantic", "Alice frontend framework", "alice")
    print("current-answer hits (expect Vue, no React):",
          [h["_source"]["text"][:30] for h in cur["hits"]["hits"]])

if __name__ == "__main__":
    main()
