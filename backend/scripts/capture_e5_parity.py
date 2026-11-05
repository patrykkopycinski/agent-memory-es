#!/usr/bin/env python3
"""Capture the parity fixture for the local multilingual-e5-small backend from a live Elasticsearch.

Read-only: it POSTs texts to `_inference/text_embedding/<endpoint>` (an inference call, no index
or cluster state is written) and does GET / and GET _nodes/os for the provenance record. The
output is tests/fixtures/e5_parity.json: the ES-hosted vectors the in-process model must match
(cosine >= 0.9999). Run it where it can reach the ES you want to be the reference, ideally on the
same architecture the service runs on (the fixture records `arch` and the model id).

    python3 backend/scripts/capture_e5_parity.py --es http://127.0.0.1:9200 \\
        --out backend/tests/fixtures/e5_parity.json

Auth, when the cluster needs it: put the full Authorization header value (scheme included) in the
CAPTURE_ES_AUTH environment variable. It is sent as-is and never printed.
Python 3.9+, standard library only.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEXTS = os.path.join(HERE, "..", "tests", "fixtures", "e5_parity_texts.json")
DEFAULT_ENDPOINT = ".multilingual-e5-small-elasticsearch"
BATCH = 4


def call(base, method, path, body=None, timeout=120):
    headers = {"Content-Type": "application/json"}
    auth = os.environ.get("CAPTURE_ES_AUTH")
    if auth:
        headers["Authorization"] = auth
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def embed(base, endpoint, texts):
    out = call(base, "POST", "/_inference/text_embedding/" + endpoint, {"input": texts})
    rows = out["text_embedding"]
    if len(rows) != len(texts):
        raise SystemExit("ES returned %d vectors for %d texts" % (len(rows), len(texts)))
    return [r["embedding"] for r in rows]


def node_arch(base):
    try:
        nodes = call(base, "GET", "/_nodes/os?filter_path=nodes.*.os.arch", timeout=30).get("nodes", {})
        return sorted({n["os"]["arch"] for n in nodes.values()}) or ["unknown"]
    except Exception:
        return ["unknown"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--es", required=True, help="Elasticsearch base URL")
    ap.add_argument("--out", required=True, help="fixture file to write")
    ap.add_argument("--texts", default=DEFAULT_TEXTS, help='JSON file: {"texts": [...]}')
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = ap.parse_args(argv)

    with open(args.texts) as fh:
        texts = json.load(fh)["texts"]
    vectors = []
    for i in range(0, len(texts), BATCH):
        vectors.extend(embed(args.es, args.endpoint, texts[i:i + BATCH]))
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise SystemExit("inconsistent vector sizes: %s" % sorted(dims))
    version = call(args.es, "GET", "/").get("version", {}).get("number", "unknown")
    fixture = {
        "source": {
            "model_id": args.endpoint,
            "es_version": version,
            "es_node_arch": node_arch(args.es),
            "capture_host_arch": platform.machine(),
            "captured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "dims": dims.pop(),
        "items": [{"text": t, "vector": v} for t, v in zip(texts, vectors)],
    }
    with open(args.out, "w") as fh:
        json.dump(fixture, fh, ensure_ascii=False)
        fh.write("\n")
    print("wrote %d items (dims=%d, ES %s, node arch %s) to %s"
          % (len(texts), fixture["dims"], version, fixture["source"]["es_node_arch"], args.out))


if __name__ == "__main__":
    sys.exit(main())
