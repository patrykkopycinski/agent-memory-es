#!/usr/bin/env python3
"""Download multilingual-e5-small at a pinned HF revision into DEST, verify the sha256 of every
file, then write DEST/REVISION. Used by the Dockerfile so the image carries exactly the pinned
bytes; app/embeddings.py refuses to load a model dir whose REVISION marker differs from
LOCAL_MODEL_REVISION. Stdlib only. Exit 1 (and no marker) on any hash mismatch."""
import argparse
import hashlib
import os
import sys
import urllib.request

FILES = ("tokenizer.json", "onnx/model.onnx")   # small file first: a bad pin fails fast


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--repo", default="intfloat/multilingual-e5-small")
    ap.add_argument("--rev", required=True)
    ap.add_argument("--sha", action="append", required=True, metavar="FILE=SHA256")
    ap.add_argument("--base", default="https://huggingface.co")
    args = ap.parse_args(argv)
    pins = dict(item.split("=", 1) for item in args.sha)
    missing = [f for f in FILES if f not in pins]
    if missing:
        sys.exit(f"no sha256 pin for {missing}")
    marker = os.path.join(args.dest, "REVISION")
    if os.path.exists(marker):
        os.remove(marker)
    for rel in FILES:
        path = os.path.join(args.dest, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(f"{args.base}/{args.repo}/resolve/{args.rev}/{rel}", path)
        got = sha256_of(path)
        if got != pins[rel].lower():
            os.remove(path)
            sys.exit(f"sha256 mismatch for {rel}: got {got}, pinned {pins[rel].lower()}")
        print(f"ok {rel} sha256={got}")
    with open(marker, "w") as fh:
        fh.write(args.rev + "\n")


if __name__ == "__main__":
    main()
