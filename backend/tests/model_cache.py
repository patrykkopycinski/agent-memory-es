"""Test-side access to the local embedding models. The app never downloads at runtime; tests that
need a model call model_or_skip(), which (1) accepts a cache dir whose files match pinned sha256s
(writing the REVISION marker), (2) fetches only when AMES_LOCAL_MODEL_ALLOW_FETCH=1, and
(3) otherwise skips the test with the reason."""
import hashlib
import os

import pytest

from app import embeddings

CACHE = os.environ.get("AMES_TEST_MODEL_CACHE", os.path.expanduser("~/.cache/ames-test-models"))
E5 = dict(name="multilingual-e5-small", repo=embeddings.LOCAL_MODEL_REPO,
          revision=embeddings.LOCAL_MODEL_REVISION,
          sha256={"onnx/model.onnx": "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
                  "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"})
BGE = dict(name="bge-small-en-v1.5", repo="BAAI/bge-small-en-v1.5",
           revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
           sha256={"onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"})


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _files_match(dest, pins):
    return all(os.path.isfile(os.path.join(dest, rel)) and _sha256(os.path.join(dest, rel)) == sha
               for rel, sha in pins.items()) and all(
        os.path.isfile(os.path.join(dest, rel)) for rel in embeddings.MODEL_FILES)


def model_or_skip(spec):
    """Return (session, tokenizer) for `spec`, or skip the calling test cleanly."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    dest = os.path.join(CACHE, spec["name"])
    if embeddings.model_problem(dest, spec["revision"]):
        if _files_match(dest, spec["sha256"]):
            with open(os.path.join(dest, embeddings.REVISION_FILE), "w") as fh:
                fh.write(spec["revision"] + "\n")
        elif not embeddings.fetch_allowed():
            pytest.skip(f"{spec['name']} not in {CACHE}; set AMES_LOCAL_MODEL_ALLOW_FETCH=1 to download")
    return embeddings.load_local(dest, repo=spec["repo"], revision=spec["revision"])
