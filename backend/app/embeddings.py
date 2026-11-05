"""Embeddings: ES-native inference first (`_inference/text_embedding`), OmniRoute fallback.

Default endpoint .multilingual-e5-small-elasticsearch (384-dim, cluster-side, no external
gateway). AMES_EMBED_BACKEND=es|omniroute|local pins one. `local` runs intfloat/multilingual-e5-small
in-process (onnxruntime + tokenizers, no `_inference`, so it works off-license) and produces the
same vectors as the ES-hosted `.multilingual-e5-small-elasticsearch` endpoint. Dims differ per backend — indices must
be rebuilt when switching backends. With AMES_EMBED_BACKEND unset, a valid baked local model is preferred
over `_inference` (which needs an ES license) and over OmniRoute.
"""
import json
import logging
import os
import threading
import time
import urllib.request
from typing import Optional

from . import http_retry
from .store import _es_url

log = logging.getLogger("ames.embed")
ES_ENDPOINT = os.environ.get("AMES_ES_INFERENCE", ".multilingual-e5-small-elasticsearch")
OR_BASE = os.environ.get("AMES_EMBED_BASE", "http://localhost:20128/v1")
OR_MODEL = os.environ.get("AMES_EMBED_MODEL", "openrouter/google/gemini-embedding-2")

_cache_dim = None
_backend = None


_probe_ctx = threading.local()   # set only inside probe(): single attempt, short timeout
PROBE_TIMEOUT_S = 5


def _post(url, body, headers=None, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    if getattr(_probe_ctx, "active", False):
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as response:
            return json.load(response)
    return http_retry.request_json(req, timeout=timeout)


def _es_embed(texts):
    out = _post(f"{_es_url()}/_inference/text_embedding/{ES_ENDPOINT}", {"input": texts})
    return [d["embedding"] for d in out["text_embedding"]]


def _or_embed(texts):
    key = os.environ.get("AMES_EMBED_KEY", os.environ.get("OMNIROUTE_API_KEY", ""))
    out = _post(f"{OR_BASE}/embeddings", {"model": OR_MODEL, "input": texts},
                headers={"Authorization": f"Bearer {key}"})
    return [d["embedding"] for d in out["data"]]


# ---- local backend: multilingual-e5-small, ONNX, loaded once per process ----------------------
# Recipe verified against ES `_inference` output (cosine 1.0000): NO "query:"/"passage:" prefix,
# mean pooling over the attention mask, L2 normalise, 384 dims, max 512 tokens.
LOCAL_MODEL_DIR = os.environ.get("AMES_LOCAL_MODEL_DIR", "/opt/models/multilingual-e5-small")
LOCAL_MODEL_REPO = os.environ.get("AMES_LOCAL_MODEL_REPO", "intfloat/multilingual-e5-small")
# Pinned HF revision of intfloat/multilingual-e5-small: the weights are the model identity, so a
# floating `main` could silently change every stored vector's geometry.
LOCAL_MODEL_REVISION = os.environ.get("AMES_LOCAL_MODEL_REVISION", "614241f622f53c4eeff9890bdc4f31cfecc418b3")
MODEL_FILES = ("onnx/model.onnx", "tokenizer.json")
REVISION_FILE = "REVISION"       # written next to the model by the image build (scripts/bake_e5_model.py)
HF_BASE = "https://huggingface.co"
LOCAL_MAX_TOKENS = 512
LOCAL_BATCH = 16
_local = None
_local_lock = threading.Lock()


class ModelUnavailable(RuntimeError):
    """The local embedding model is missing or is not the pinned revision. Never auto-fetched."""


class ModelLoading(RuntimeError):
    """The local model is still being loaded by another thread (startup warm-up)."""


def fetch_allowed() -> bool:
    """Runtime downloads are opt-in (AMES_LOCAL_MODEL_ALLOW_FETCH=1), for tests and dev checkouts
    only. Production bakes the model into the image and never reaches for the network."""
    return os.environ.get("AMES_LOCAL_MODEL_ALLOW_FETCH", "") == "1"


def model_problem(dest: str, revision: str = None) -> Optional[str]:
    """Why `dest` is not a usable model dir (names the offending path), or None when it is."""
    revision = revision or LOCAL_MODEL_REVISION
    for rel in MODEL_FILES:
        path = os.path.join(dest, rel)
        if not os.path.isfile(path):
            return f"model file missing: {path}"
    marker = os.path.join(dest, REVISION_FILE)
    try:
        with open(marker) as fh:
            have = fh.read().strip()
    except OSError:
        return f"model REVISION marker missing: {marker}"
    if have != revision:
        return f"model revision {have[:12]} != pinned {revision[:12]}: {marker}"
    return None


def fetch_model(dest: str, repo: str = None, revision: str = None) -> tuple:
    """Download the model files from the HF hub at a pinned revision (stdlib only) into `dest` and
    write the REVISION marker. Only reachable through load_local with
    AMES_LOCAL_MODEL_ALLOW_FETCH=1. Files already under a different or missing marker are
    re-downloaded: unmarked bytes are never trusted."""
    repo, revision = repo or LOCAL_MODEL_REPO, revision or LOCAL_MODEL_REVISION
    marker = os.path.join(dest, REVISION_FILE)
    trusted = model_problem(dest, revision) is None
    paths = []
    for rel in MODEL_FILES:
        path = os.path.join(dest, rel)
        if not trusted or not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = f"{HF_BASE}/{repo}/resolve/{revision}/{rel}"
            tmp = path + ".part"
            with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            os.replace(tmp, path)
        paths.append(path)
    with open(marker, "w") as fh:
        fh.write(revision + "\n")
    return tuple(paths)


def load_local(model_dir=None, repo=None, revision=None) -> tuple:
    """Build an (onnxruntime session, tokenizer) pair. Not cached: see _local_model().
    Raises ModelUnavailable (naming the path) when the files are absent or the REVISION marker
    does not match; downloads only when AMES_LOCAL_MODEL_ALLOW_FETCH=1."""
    dest, revision = model_dir or LOCAL_MODEL_DIR, revision or LOCAL_MODEL_REVISION
    problem = model_problem(dest, revision)
    if problem and fetch_allowed():
        fetch_model(dest, repo, revision)
        problem = model_problem(dest, revision)
    if problem:
        raise ModelUnavailable(problem)
    import onnxruntime as ort
    from tokenizers import Tokenizer
    onnx, tok_path = (os.path.join(dest, rel) for rel in MODEL_FILES)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(onnx, sess_options=opts, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(tok_path)
    tok.enable_truncation(max_length=LOCAL_MAX_TOKENS)
    tok.enable_padding()   # pad to the longest in the batch (pad id/token come from tokenizer.json)
    return session, tok


def _local_model():
    global _local
    if _local is None:
        with _local_lock:
            if _local is None:
                _local = load_local()
    return _local


def local_embed_with(session, tok, texts: list) -> list:
    """Embed `texts` with an already-loaded (session, tokenizer). Model-agnostic pooling so the
    parity test can run a wrong model through the same code path as a negative control."""
    import numpy as np
    wanted = {i.name for i in session.get_inputs()}
    out = []
    for start in range(0, len(texts), LOCAL_BATCH):
        enc = tok.encode_batch(texts[start:start + LOCAL_BATCH])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in wanted:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = session.run(None, {k: v for k, v in feeds.items() if k in wanted})[0]
        m = mask[:, :, None].astype(hidden.dtype)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        out.extend(pooled.astype(float).tolist())
    return out


def _local_embed(texts):
    session, tok = _local_model()
    return local_embed_with(session, tok, texts)


def _local_ready() -> bool:
    """Auto-detect: the baked model dir is valid and the runtime deps import. No I/O beyond stat."""
    import importlib.util
    return (importlib.util.find_spec("onnxruntime") is not None
            and importlib.util.find_spec("tokenizers") is not None
            and model_problem(LOCAL_MODEL_DIR) is None)


def backend() -> str:
    global _backend
    if _backend:
        return _backend
    pinned = os.environ.get("AMES_EMBED_BACKEND", "")
    if pinned in ("es", "omniroute", "local"):
        _backend = pinned
    elif _local_ready():
        _backend = "local"      # a valid baked model beats a license-gated or dims-mismatched backend
    else:
        try:
            _es_embed(["probe"])
            _backend = "es"
        except Exception:
            _backend = "omniroute"
    return _backend


def backend_name() -> str:
    """The configured backend without triggering auto-detection I/O."""
    return _backend or os.environ.get("AMES_EMBED_BACKEND", "") or "auto"


def embed(texts: list) -> list:
    b = backend()
    if b == "local":
        return _local_embed(texts)
    return _es_embed(texts) if b == "es" else _or_embed(texts)


LOAD_RETRY_S = 30                 # after a failed background load, report the error this long before retrying
_loader = None                    # the probe-started loader thread, if any
_load_failure = None              # (exception, monotonic time) of the last failed background load
_loader_lock = threading.Lock()


def _background_load() -> None:
    global _load_failure
    try:
        _local_model()
        _load_failure = None
    except Exception as exc:
        logging.getLogger(__name__).warning("background local model load failed", exc_info=True)
        _load_failure = (exc, time.monotonic())


def _ensure_background_load() -> None:
    """Start a load unless one is already running (this loader, or another thread holding the
    lock, e.g. the startup warm-up). Raises the last load error while it is recent."""
    global _loader, _load_failure
    with _loader_lock:
        if _local_lock.locked() or (_loader is not None and _loader.is_alive()):
            return
        if _load_failure is not None:
            exc, at = _load_failure
            if time.monotonic() - at < LOAD_RETRY_S:
                raise exc
            _load_failure = None
        _loader = threading.Thread(target=_background_load, daemon=True, name="embed-probe-load")
        _loader.start()


def probe() -> None:
    """One embed for /health: a single attempt with a PROBE_TIMEOUT_S cap, never http_retry's
    multi-minute budget (a dead backend must not hang the health endpoint). Raises on failure."""
    if backend() == "local" and _local is None:
        # The probe never loads the model itself and never queues on _local_lock: a watchdog's
        # curl -m 10 would give up first. Report "loading" and make sure a load is running.
        _ensure_background_load()
        raise ModelLoading("local embedding model is loading")
    _probe_ctx.active = True
    try:
        embed(["probe"])
    finally:
        _probe_ctx.active = False


# Process-wide embed failure counter. embed() failures on the retain/recall hot path used to be
# swallowed silently (a license expiry left retain writing vector-less docs for 5 days); every
# swallowed failure now lands here and is surfaced by /health and /memory/stats.
FAILURES = {"count": 0, "last_error": None, "last_at": None}
_failures_lock = threading.Lock()


def describe_error(exc: BaseException) -> str:
    """Short, single-line error string safe to return in an API response."""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}"


def embed_safe(texts: list) -> tuple:
    """embed() that never raises: (vectors, None) on success, (None, error_string) on failure.
    Failures are counted in FAILURES and logged. Looks up `embed` at call time so tests can
    monkeypatch it."""
    try:
        return embed(texts), None
    except Exception as exc:
        err = describe_error(exc)
        with _failures_lock:
            FAILURES["count"] += 1
            FAILURES["last_error"] = err
            FAILURES["last_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log.warning("ames.embed failed (writes land vector-less, recall loses its vector arm): %s", err)
        return None, err


def dim() -> int:
    global _cache_dim
    if _cache_dim:
        return _cache_dim
    _cache_dim = int(os.environ.get("AMES_EMBED_DIM", "0")) or len(embed(["probe"])[0])
    return _cache_dim
