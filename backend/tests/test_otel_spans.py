"""OTEL span-emission tests (app/telemetry.py + FastAPI/ES instrumentation).

The pipeline is built with an in-memory exporter: a TracerProvider whose
SpanProcessor is the SDK's `SimpleSpanProcessor(InMemorySpanExporter)` — the
TestSpanProcessor role. Nothing touches the network, and every assertion reads
real spans produced by a real recall against live ES.

Run:
  AMES_ES_URL=http://localhost:19301 pytest backend/tests/test_otel_spans.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from fastapi.testclient import TestClient

from app import auth, telemetry as _otel  # noqa: E402
from app.main import app  # noqa: E402

H = lambda k: {"X-API-Key": k}  # noqa: E731

# Text that must never surface in a span attribute. The privacy rule in
# telemetry.py is "kinds, visibility, counts, lengths, timings — never content".
SECRET_MEMORY = "zebra-callsign-quantum-muffin deploy passphrase is hunter2-espresso"
SECRET_QUERY = "zebra-callsign-quantum-muffin"


@pytest.fixture(scope="module")
def exporter():
    """In-memory span pipeline, attached to the app for the whole module."""
    exp = InMemorySpanExporter()
    # build_provider() is the production builder: scrubber first, then the
    # exporter's processor. The tests therefore run the real privacy pipeline.
    provider = _otel.build_provider(SimpleSpanProcessor(exp), {
        "service.name": "agent-memory-es-test",
        "deployment.environment.name": "test",
    })

    # configure(tracer_provider=...) is the documented test seam: it pins
    # app/telemetry.py to this provider and bypasses the AMES_OTEL_ENABLED gate.
    _otel.reset()
    _otel.configure(tracer_provider=provider)
    assert _otel.instrument_fastapi(app, tracer_provider=provider, force=True), \
        "FastAPI instrumentation did not attach"

    yield exp
    exp.clear()
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.uninstrument_app(app)
        app._ames_otel_instrumented = False
    except Exception:
        pass
    _otel.reset()


@pytest.fixture()
def spans(exporter):
    exporter.clear()
    return exporter


@pytest.fixture(scope="module")
def owner():
    key = auth.create_key("otel-alice", person="otel-alice")
    yield key
    # promoted/retained docs are visible to other suites sharing this ES: remove them
    from app.store import es, idx
    for kind in ("semantic", "episodic", "procedural"):
        es("POST", f"/{idx(kind)}/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": "otel-alice"}}})


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _by_name(spans, name):
    return [s for s in spans.get_finished_spans() if s.name == name]


def _attrs(span):
    # The SDK stores sequence attributes as tuples; normalise for comparison.
    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in (dict(span.attributes or {}).items())}


def _http_attr(attrs, *names):
    """HTTP attrs differ between semconv generations (0.64b emits http.method /
    http.status_code; newer emit http.request.method / http.response.status_code).
    Assert on the value, accept either spelling."""
    for n in names:
        if n in attrs:
            return attrs[n]
    raise KeyError(f"none of {names} in {sorted(attrs)}")


# --- the required case: a recall call emits the span tree --------------------

def test_recall_emits_fastapi_domain_and_es_spans(spans, client, owner):
    r = client.post("/memory/retain", headers=H(owner),
                    json={"kind": "semantic", "text": SECRET_MEMORY,
                          "visibility": "private"})
    assert r.status_code == 200, r.text
    spans.clear()

    r = client.post("/memory/recall", headers=H(owner),
                    json={"query": SECRET_QUERY, "size": 5})
    assert r.status_code == 200, r.text
    assert r.json()["fused"], "expected a hit so hit-count attrs are non-trivial"

    server = _by_name(spans, "POST /memory/recall")
    domain = _by_name(spans, "memory.recall")
    es = _by_name(spans, "es.request")
    assert server, f"no FastAPI server span; got {sorted({s.name for s in spans.get_finished_spans()})}"
    assert domain, "no memory.recall domain span"
    assert es, "no es.request client span"

    # one domain span per API call, parented by the HTTP server span
    assert len(domain) == 1
    assert domain[0].parent is not None
    assert domain[0].parent.span_id == server[0].context.span_id
    # ES client spans hang off the domain span and share its trace
    for s in es:
        assert s.context.trace_id == domain[0].context.trace_id

    # FastAPI server span carries the standard HTTP attributes
    sa = _attrs(server[0])
    assert _http_attr(sa, "http.request.method", "http.method") == "POST"
    assert sa.get("http.route") == "/memory/recall"
    assert _http_attr(sa, "http.response.status_code", "http.status_code") == 200


def test_recall_span_attributes_counts_and_visibility(spans, client, owner):
    client.post("/memory/recall", headers=H(owner), json={"query": "quantum muffin"})
    dom = _by_name(spans, "memory.recall")[0]
    a = _attrs(dom)

    assert a["memory.hits.total"] >= 1
    assert a["memory.size"] == 8
    assert a["memory.query.length"] == len("quantum muffin")
    # kinds default to all three; a single-kind request collapses to memory.kind
    assert a["memory.kinds"] == ["episodic", "semantic", "procedural"]
    # visibility mix + per-kind counts are recorded as counts, not documents
    assert a["memory.visibility.private"] >= 1
    assert "memory.visibility.distinct" in a
    assert a.get("memory.hits.kind.semantic", 0) >= 1
    assert a["memory.abstained"] is False
    # per-stage timings (memory.py already computed them; the span carries them)
    assert isinstance(a.get("memory.ms.total"), float)
    assert isinstance(a.get("memory.ms.search"), float)


def test_recall_single_kind_and_abstain(spans, client, owner):
    r = client.post("/memory/recall", headers=H(owner),
                    json={"query": "xylophone-unicorns-marshmallow", "size": 3,
                          "kinds": ["episodic"], "min_score": 9999})
    assert r.status_code == 200, r.text
    a = _attrs(_by_name(spans, "memory.recall")[0])
    assert a["memory.kind"] == "episodic"
    assert a["memory.kinds"] == ["episodic"]
    assert a["memory.hits.total"] == 0
    assert a["memory.abstained"] is True


def test_es_span_carries_index_pattern_and_latency(spans, client, owner):
    client.post("/memory/recall", headers=H(owner), json={"query": "quantum muffin"})
    es = _by_name(spans, "es.request")
    assert es
    search = [s for s in es if _attrs(s).get("es.index", "").endswith("_semantic")]
    assert search, "no ES span for the semantic index"
    a = _attrs(search[0])
    assert a["db.system"] == "elasticsearch"
    assert a["db.operation"] == "POST"
    # amtest_am_semantic -> am_semantic_* : the test prefix is noise, the pattern
    # keeps the collection+kind so operators can group by it
    assert a["es.index_pattern"] == "am_semantic_*", a
    assert "es.hits.total" in a
    # latency: round-trip is on the span even when ES reports took=0
    dur = (search[0].end_time - search[0].start_time) / 1e6
    assert dur > 0


def test_retain_span(spans, client, owner):
    client.post("/memory/retain", headers=H(owner),
                json={"kind": "procedural", "text": "run pytest before every merge",
                      "visibility": "team"})
    dom = _by_name(spans, "memory.retain")
    assert dom
    a = _attrs(dom[0])
    assert a["memory.kind"] == "procedural"
    assert a["memory.visibility"] == "team"
    assert a["memory.text_chars"] == len("run pytest before every merge")
    assert "ames.owner_hash" in a and len(a["ames.owner_hash"]) == 12


def test_promote_and_reflect_spans(spans, client, owner, monkeypatch):
    # reflect calls the LLM gateway; stub it so the test needs no live model
    from app import llm as _llm
    monkeypatch.setattr(_llm, "rewrite_queries", lambda q, n=2: [q + " rewrite"])
    monkeypatch.setattr(_llm, "chat", lambda q, sources: "stub answer")
    doc = client.post("/memory/retain", headers=H(owner),
                      json={"kind": "semantic",
                            "text": "convention: otel span names use dotted lowercase",
                            "visibility": "private"}).json()
    spans.clear()
    r = client.post("/memory/promote", headers=H(owner),
                    json={"kind": "semantic", "doc_id": doc["_id"],
                          "to_visibility": "common"})
    assert r.status_code == 200, r.text
    a = _attrs(_by_name(spans, "memory.promote")[0])
    assert a["memory.kind"] == "semantic"
    assert a["memory.visibility"] == "common"

    spans.clear()
    r = client.post("/memory/reflect", headers=H(owner),
                    json={"question": "otel span naming convention"})
    assert r.status_code == 200, r.text
    refl = _by_name(spans, "memory.reflect")
    assert refl, "no memory.reflect span"
    ra = _attrs(refl[0])
    assert ra["memory.query.length"] == len("otel span naming convention")
    assert ra["memory.reflect.queries"] >= 1
    assert "memory.hits.total" in ra


def _admin_headers(monkeypatch):
    monkeypatch.setenv("AMES_ADMIN_TOKEN", "dev-admin")
    return {"X-Admin-Token": "dev-admin"}


def test_admin_endpoint_is_traced(spans, client, monkeypatch):
    r = client.post("/admin/keys", headers=_admin_headers(monkeypatch),
                    params={"owner_id": "otel-admin-probe", "person": "otel-admin-probe"})
    assert r.status_code == 200, r.text
    server = _by_name(spans, "POST /admin/keys")
    assert server, "admin endpoints must be traced like the rest"
    assert _http_attr(_attrs(server[0]),
                      "http.response.status_code", "http.status_code") == 200


def test_health_is_excluded_from_tracing(spans, client):
    assert client.get("/health").status_code == 200
    assert not _by_name(spans, "GET /health"), "/health must stay out of the span stream"


def test_error_recall_records_error_status(spans, client, owner):
    r = client.post("/memory/recall", headers=H(owner),
                    json={"query": "x", "filter": {"nonsense": True}})
    assert r.status_code == 422, r.text
    server = _by_name(spans, "POST /memory/recall")
    assert server
    assert _http_attr(_attrs(server[0]), "http.response.status_code",
                      "http.status_code") == 422


# --- the privacy rule -------------------------------------------------------

def test_no_memory_text_query_or_key_in_any_attribute(spans, client, owner):
    """No raw secrets in attrs: not the memory text, not the query, not the key."""
    client.post("/memory/retain", headers=H(owner),
                json={"kind": "episodic", "text": SECRET_MEMORY,
                      "visibility": "private"})
    r = client.post("/memory/recall", headers=H(owner),
                    json={"query": SECRET_QUERY})
    assert r.status_code == 200

    blob = []
    for s in spans.get_finished_spans():
        for k, v in (s.attributes or {}).items():
            blob.append(f"{k}={v}")
        for ev in (s.events or []):
            blob.append(str(ev.name))
            for k, v in (ev.attributes or {}).items():
                blob.append(f"{k}={v}")
    joined = " | ".join(blob).lower()

    for forbidden in ("zebra-callsign", "hunter2", "espresso", "quantum-muffin",
                      "quantum muffin", owner.lower(), "ame_"):
        assert forbidden not in joined, f"secret leaked into span attrs: {forbidden!r}"

    # the owner is present only as the truncated digest the log line already uses
    dom = _attrs(_by_name(spans, "memory.recall")[0])
    assert dom["ames.owner_hash"] != "otel-alice"


def test_attribute_values_are_otlp_scalars(spans, client, owner):
    client.post("/memory/recall", headers=H(owner), json={"query": "quantum muffin"})
    for s in spans.get_finished_spans():
        for k, v in (s.attributes or {}).items():
            assert isinstance(v, (bool, int, float, str)) or (
                isinstance(v, (list, tuple)) and
                all(isinstance(x, (bool, int, float, str)) for x in v)), \
                f"{s.name}.{k} has non-scalar attribute {type(v).__name__}"


# --- the no-op contract -----------------------------------------------------

def test_telemetry_helpers_are_safe_without_a_provider(monkeypatch):
    """With telemetry unconfigured every helper is a silent no-op: no span is
    created even when someone else installed a recording global provider, and a
    failing body's error still propagates."""
    saved = dict(_otel._state)
    _otel.reset()
    try:
        with _otel.span("nope", {"a": 1}) as sp:
            assert sp is None, "tracing off must yield no span at all"
        assert _otel.current_span() is None or \
            not _otel.current_span().get_span_context().is_valid
        _otel.set_attrs({"a": 1})
        _otel.add_event("e")
        with pytest.raises(ZeroDivisionError):
            with _otel.span("boom"):
                raise ZeroDivisionError("body error must propagate")
    finally:
        _otel._state.update(saved)


def test_recording_global_provider_does_not_revive_disabled_tracing():
    """Negative control for the no-op test: a recording global tracer exists, and
    our spans must still not use it while the module provider is unset."""
    exp = InMemorySpanExporter()
    other = _otel.build_provider(SimpleSpanProcessor(exp))
    saved = dict(_otel._state)
    _otel.reset()
    try:
        with other.get_tracer("x").start_as_current_span("outer"):
            with _otel.span("must-not-record"):
                pass
        assert [s.name for s in exp.get_finished_spans()] == ["outer"]
    finally:
        _otel._state.update(saved)


# --- P1-1: off by default, no implicit endpoint ------------------------------

def test_default_is_off_without_endpoint(monkeypatch):
    for var in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AMES_OTEL_ENABLED", raising=False)
    saved = dict(_otel._state)
    try:
        _otel.reset()
        assert _otel.enabled() is False
        assert _otel.configure() is None
        from fastapi import FastAPI
        assert _otel.instrument_fastapi(FastAPI()) is False
        # enabled alone is still not enough: an endpoint has to be named
        monkeypatch.setenv("AMES_OTEL_ENABLED", "1")
        _otel.reset()
        assert _otel.configure() is None
        assert _otel.instrument_fastapi(FastAPI()) is False
    finally:
        _otel._state.update(saved)


def test_endpoint_without_enable_flag_is_still_off(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.invalid:4318")
    monkeypatch.setenv("AMES_OTEL_ENABLED", "0")
    saved = dict(_otel._state)
    try:
        _otel.reset()
        assert _otel.configure() is None
    finally:
        _otel._state.update(saved)


# --- P1-2: exception content never leaves the process -------------------------

CANARY = "CANARY-es-body-7f3a9c"


def _everything(spans_):
    """Every string a span could export: attrs, event names + attrs, status."""
    blob = []
    for s in spans_.get_finished_spans():
        blob.append(s.name)
        for k, v in (s.attributes or {}).items():
            blob.append(f"{k}={v}")
        for ev in (s.events or []):
            blob.append(str(ev.name))
            for k, v in (ev.attributes or {}).items():
                blob.append(f"{k}={v}")
        blob.append(str(getattr(s.status, "description", "") or ""))
    return " | ".join(blob)


def test_es_error_body_canary_never_reaches_spans_or_logs(spans, monkeypatch, caplog):
    """Force a real ES 400 whose body echoes a planted canary, as ES does with
    query text. The canary must be on no span attribute, event or status, and in
    no log record the telemetry layer could export."""
    import io
    import urllib.error
    import urllib.request
    from app.store import PREFIX, es

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(('{"error":{"reason":"parse failure near %s"}}' % CANARY).encode()))
    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with _otel.span("memory.recall"):
        with pytest.raises(RuntimeError) as ei:
            es("POST", "/%sam_semantic/_search" % PREFIX, {"query": {"match_all": {}}})
    assert CANARY in str(ei.value), "the error itself must stay informative to the caller"

    es_spans = _by_name(spans, "es.request")
    assert es_spans, "failed ES call must still produce a span"
    a = _attrs(es_spans[0])
    assert a["error.type"] == "HTTPError"
    assert es_spans[0].status.status_code.name == "ERROR"
    # status carries the raised class (RuntimeError wraps the HTTPError), never text
    assert es_spans[0].status.description == "RuntimeError"
    assert not [e for e in es_spans[0].events if e.name == "exception"], \
        "record_exception must not be used"
    assert CANARY not in _everything(spans)
    assert CANARY not in caplog.text
    import logging
    assert not [h for h in logging.getLogger().handlers
                if type(h).__name__ == "LoggingHandler"], \
        "no OTLP log bridge may be attached: it would export exception text"


def test_third_party_exception_events_are_scrubbed(spans):
    """The ASGI middleware calls record_exception on its own spans; the
    SpanProcessor must remove message + stacktrace from those too."""
    tr = _otel.tracer()
    with pytest.raises(ValueError):
        with tr.start_as_current_span("third.party") as sp:  # record_exception default True
            raise ValueError(CANARY)
    s = _by_name(spans, "third.party")[0]
    assert s.events, "precondition: the SDK did record an exception event"
    assert CANARY not in _everything(spans)
    for ev in s.events:
        assert "exception.message" not in (ev.attributes or {})
        assert "exception.stacktrace" not in (ev.attributes or {})
    assert s.status.description in (None, "", "ValueError")


def test_unhandled_route_error_keeps_class_only(spans, client, owner, monkeypatch):
    from app import memory

    def blow(*a, **k):
        raise RuntimeError("ES POST /x -> 400: " + CANARY)
    monkeypatch.setattr(memory, "recall", blow)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/memory/recall", headers=H(owner), json={"query": "q"})
    assert r.status_code == 500
    assert CANARY not in _everything(spans)


# --- P1-3: no parentless es.request noise -------------------------------------

def test_idle_es_call_without_parent_emits_nothing(spans):
    from app.store import es
    es("GET", "/_cluster/health")          # what an idle worker poll does
    assert not _by_name(spans, "es.request")


def test_unparented_es_call_is_emitted_when_slow(spans, monkeypatch):
    monkeypatch.setenv("AMES_OTEL_ES_SLOW_MS", "0")
    from app.store import es
    es("GET", "/_cluster/health")
    out = _by_name(spans, "es.request")
    assert len(out) == 1
    a = _attrs(out[0])
    assert a["ames.unparented"] is True and a["ames.slow"] is True
    assert a["db.system"] == "elasticsearch"


def test_unparented_es_call_is_emitted_when_failed(spans, monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from app.store import es

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "x", {}, io.BytesIO(b"{}"))
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError):
        es("GET", "/_cluster/health")
    out = _by_name(spans, "es.request")
    assert len(out) == 1 and _attrs(out[0])["error.type"] == "HTTPError"
    assert _attrs(out[0])["ames.unparented"] is True


def test_worker_job_span_parents_its_es_calls(spans, monkeypatch):
    """A claimed fact job opens worker.job; its es.request children join it and
    the job carries counts + ES write latency. An empty poll emits nothing."""
    from app import facts

    # empty poll: no job claimed -> no spans at all
    monkeypatch.setattr(facts, "claim", lambda *a, **k: None)
    assert facts.run_once() is False
    assert not spans.get_finished_spans()

    from app.store import es
    finished = {}
    monkeypatch.setattr(facts, "claim", lambda *a, **k: (
        "job-1", {"source_id": "s", "owner_id": "o", "attempts": 0,
                  "transient_retries": 0, "state": "running"}, "tok"))
    monkeypatch.setattr(facts, "_finish", lambda *a, **k: finished.update(k, args=a))

    def fake_process(source_id, owner_id, abort=None):
        es("GET", "/_cluster/health")      # child call joins worker.job
        _otel.set_attrs({"worker.job.passages": 2, "worker.job.facts": 3,
                         "worker.job.facts_written": 3, "worker.job.es_write_ms": 4.5})
    monkeypatch.setattr(facts, "process", fake_process)
    assert facts.run_once() is True
    job = _by_name(spans, "worker.job")
    assert len(job) == 1
    a = _attrs(job[0])
    assert a["worker.job.kind"] == "fact_extraction"
    assert a["worker.job.outcome"] == "completed"
    assert a["worker.job.facts"] == 3 and a["worker.job.es_write_ms"] == 4.5
    child = _by_name(spans, "es.request")
    assert child and child[0].parent.span_id == job[0].context.span_id
    assert "ames.unparented" not in _attrs(child[0])


def test_worker_job_failure_records_class_only(spans, monkeypatch):
    from app import facts
    monkeypatch.setattr(facts, "claim", lambda *a, **k: (
        "job-2", {"source_id": "s", "owner_id": "o", "attempts": 0,
                  "transient_retries": 0, "state": "running"}, "tok"))
    monkeypatch.setattr(facts, "_finish", lambda *a, **k: None)
    monkeypatch.setattr(facts, "process",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError(CANARY)))
    facts.run_once()
    a = _attrs(_by_name(spans, "worker.job")[0])
    assert a["error.type"] == "ValueError" and a["worker.job.outcome"] == "pending"
    assert CANARY not in _everything(spans)


def test_backfill_span_wraps_the_run(spans, monkeypatch):
    from app import facts
    monkeypatch.setattr(facts, "claim_backfill",
                        lambda: ("k", {"visibility": "private", "owner_id": "o"}, "tok"))
    monkeypatch.setattr(facts, "_heartbeat", lambda *a, **k: None)

    def fake_run(key, doc, token, **kw):
        _otel.set_attrs({"worker.backfill.queued": 7})
    monkeypatch.setattr(facts, "_run_backfill", fake_run)
    assert facts.run_backfill_once() is True
    a = _attrs(_by_name(spans, "worker.backfill")[0])
    assert a["worker.job.kind"] == "backfill"
    assert a["worker.backfill.queued"] == 7 and a["worker.job.outcome"] == "completed"
    assert "o" != a.get("worker.backfill.visibility")  # visibility only, no owner

    monkeypatch.setattr(facts, "_run_backfill",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(CANARY)))
    monkeypatch.setattr(facts, "_update_backfill", lambda *a, **k: None)
    spans.clear()
    facts.run_backfill_once()
    a = _attrs(_by_name(spans, "worker.backfill")[0])
    assert a["worker.job.outcome"] == "failed" and a["error.type"] == "RuntimeError"
    assert CANARY not in _everything(spans)


def test_consolidate_pass_span(spans, monkeypatch):
    from app import worker
    monkeypatch.setattr(worker, "known_owners", lambda: [])
    assert worker.run_once() == {}
    assert not spans.get_finished_spans(), "a pass over zero owners emits nothing"

    monkeypatch.setattr(worker, "known_owners", lambda: ["o1", "o2"])
    monkeypatch.setattr(worker, "_consolidate_owners", lambda owners: {
        o: {"proposals": 1, "superseded": 2, "model_drafts": 0} for o in owners})
    worker.run_once()
    a = _attrs(_by_name(spans, "worker.consolidate")[0])
    assert a["worker.owners"] == 2 and a["worker.proposals"] == 2
    assert a["worker.superseded"] == 4


# --- P2-1: query strings never reach URL attributes ---------------------------

def test_query_string_canaries_stripped_from_url_attributes(spans, client, monkeypatch):
    r = client.post("/admin/keys", headers=_admin_headers(monkeypatch),
                    params={"owner_id": "CANARYOWNER-91c", "person": "CANARYPERSON-42d"})
    assert r.status_code == 200, r.text
    server = _by_name(spans, "POST /admin/keys")[0]
    assert "CANARYOWNER" not in _everything(spans)
    assert "CANARYPERSON" not in _everything(spans)
    a = _attrs(server)
    assert "url.query" not in a
    for key in ("http.target", "http.url", "url.full"):
        if key in a:
            assert "?" not in a[key], (key, a[key])
    assert a.get("http.target", "/admin/keys") == "/admin/keys"


def test_no_header_capture_attributes(spans, client, owner):
    client.post("/memory/recall", headers=H(owner), json={"query": "q"})
    for s in spans.get_finished_spans():
        for k in (s.attributes or {}):
            assert not k.startswith(("http.request.header.", "http.response.header."))


# --- P2-3: the scan itself must be able to fail ------------------------------

def test_leak_scan_negative_control(spans):
    """Plant a canary in an attribute that bypasses the scrubber's rules (a plain
    attribute) and prove the scan finds it, so a green no-leak test means something."""
    with _otel.span("planted", {"planted.text": "PLANTED-CANARY-d00d"}):
        pass
    assert "PLANTED-CANARY-d00d" in _everything(spans)


def test_missing_otel_packages_do_not_break_the_app():
    """Minimal image: opentelemetry not installed. app.main must import and
    /health must answer."""
    import subprocess
    code = (
        "import sys\n"
        "for m in ('opentelemetry','opentelemetry.trace','opentelemetry.sdk',\n"
        "          'opentelemetry.sdk.trace','opentelemetry.sdk.resources',\n"
        "          'opentelemetry.instrumentation.fastapi'):\n"
        "    sys.modules[m] = None\n"
        "from fastapi.testclient import TestClient\n"
        "from app import telemetry\n"
        "import app.main as m\n"
        "assert telemetry._HAVE_API is False\n"
        "with telemetry.span('x') as sp: assert sp is None\n"
        "r = TestClient(m.app).get('/health')\n"
        "assert r.status_code == 200, r.text\n"
        # /health starts a daemon thread that loads the local model; exiting while it is inside
        # native code aborts the interpreter at shutdown, after the assertions already passed
        "from app import embeddings\n"
        "loader = embeddings._loader\n"
        "if loader is not None:\n"
        "    loader.join(timeout=90)\n"
        "    assert not loader.is_alive(), 'model loader still running'\n"
        "print('OK')\n")
    env = dict(os.environ, AMES_OTEL_ENABLED="1",
               OTEL_EXPORTER_OTLP_ENDPOINT="http://collector.invalid:4318")
    out = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(
        os.path.dirname(__file__), ".."), env=env, capture_output=True, text=True,
        timeout=120)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]


# --- P2-4: flush on shutdown --------------------------------------------------

def test_shutdown_hook_flushes_provider(monkeypatch):
    from app import main
    called = []
    monkeypatch.setattr(_otel, "shutdown", lambda: called.append("shutdown"))
    main._shutdown()
    assert called == ["shutdown"]


def test_worker_sigterm_handler_flushes_then_exits(monkeypatch):
    import signal
    from app import worker
    called = []
    monkeypatch.setattr(worker.telemetry, "shutdown", lambda: called.append(1))
    prev = signal.getsignal(signal.SIGTERM)
    try:
        worker.install_sigterm_flush()
        handler = signal.getsignal(signal.SIGTERM)
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN, prev)
        with pytest.raises(SystemExit) as ei:
            handler(signal.SIGTERM, None)
        assert ei.value.code == 0 and called == [1]
    finally:
        signal.signal(signal.SIGTERM, prev)


# --- B2: recall quality signal --------------------------------------------------

def test_recall_quality_attributes_on_recall_span(spans, client, owner):
    client.post("/memory/retain", headers=H(owner),
                json={"kind": "semantic", "text": SECRET_MEMORY, "visibility": "private"})
    spans.clear()
    r = client.post("/memory/recall", headers=H(owner), json={"query": SECRET_QUERY})
    assert r.status_code == 200 and r.json()["fused"]
    a = _attrs(_by_name(spans, "memory.recall")[0])
    assert a["memory.recall.quality.hit_count"] >= 1
    assert a["memory.recall.quality.empty"] is False
    top, lo = a["memory.recall.quality.top_score"], a["memory.recall.quality.min_score"]
    assert top >= lo and a["memory.recall.quality.score_spread"] == pytest.approx(top - lo, abs=1e-5)
    assert all(isinstance(v, (bool, int, float)) for k, v in a.items()
               if k.startswith("memory.recall.quality."))


def test_recall_quality_empty_result_flag():
    q = _otel.recall_quality_attrs([])
    assert q == {"memory.recall.quality.hit_count": 0, "memory.recall.quality.empty": True}
    q = _otel.recall_quality_attrs([{"score": 0.5, "text": "T"}, {"score": 0.2}])
    assert q["memory.recall.quality.top_score"] == 0.5
    assert q["memory.recall.quality.score_spread"] == pytest.approx(0.3)
    assert not any(isinstance(v, str) for v in q.values())


# --- B2: MCP tool-call spans + trace propagation --------------------------------

def _load_stdio_mcp():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "ames_mcp_server.py")
    spec = importlib.util.spec_from_file_location("ames_mcp_server_otel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stdio_mcp_tool_call_span_and_traceparent(spans, monkeypatch):
    mod = _load_stdio_mcp()
    assert mod._otel is _otel or mod._otel.__name__.endswith("telemetry")
    seen = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return b'{"fused":[{"score":0.9},{"score":0.4}],"abstained":false}'

    import urllib.request

    def fake_urlopen(req, timeout=None):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return R()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.json, "load", lambda r: {"fused": [{"score": 0.9}, {"score": 0.4}],
                                                     "abstained": False})
    res = mod.dispatch("http://backend.invalid", "ame_KEY", "ames_recall",
                       {"query": "CANARY-mcp-query"})
    assert res["fused"]
    sp = _by_name(spans, "mcp.tool_call")
    assert len(sp) == 1
    a = _attrs(sp[0])
    assert a["mcp.tool.name"] == "ames_recall"
    assert a["mcp.result.fused"] == 2
    assert a["memory.recall.quality.hit_count"] == 2
    assert a["mcp.latency_ms"] >= 0
    # traceparent carries the MCP span's own context
    tp = seen["headers"]["traceparent"]
    ver, trace_id, span_id, flags = tp.split("-")
    assert int(trace_id, 16) == sp[0].context.trace_id
    assert int(span_id, 16) == sp[0].context.span_id
    assert "CANARY-mcp-query" not in _everything(spans)
    assert "ame_KEY" not in _everything(spans)


def test_stdio_mcp_unknown_tool_name_is_not_recorded(spans):
    mod = _load_stdio_mcp()
    with pytest.raises(ValueError):
        mod.dispatch("http://backend.invalid", "k", "CANARY-tool-name", {})
    sp = _by_name(spans, "mcp.tool_call")[0]
    assert _attrs(sp)["mcp.tool.name"] == "unknown"
    assert sp.status.status_code.name == "ERROR"
    assert "CANARY-tool-name" not in _everything(spans)


def test_mcp_trace_connects_to_backend_server_span(spans, client, owner):
    """End to end in-process: the traceparent the MCP server injects makes the
    backend HTTP server span a child, memory.recall and es.request follow."""
    with _otel.span("mcp.tool_call", {"mcp.tool.name": "ames_recall"}, otel_kind="server") as mcp_sp:
        headers = _otel.inject_headers({**H(owner)})
        assert "traceparent" in headers
        r = client.post("/memory/recall", headers=headers, json={"query": SECRET_QUERY})
    assert r.status_code == 200
    mcp_span = _by_name(spans, "mcp.tool_call")[0]
    server = _by_name(spans, "POST /memory/recall")[0]
    dom = _by_name(spans, "memory.recall")[0]
    es_spans = _by_name(spans, "es.request")
    assert server.context.trace_id == mcp_span.context.trace_id
    assert server.parent.span_id == mcp_span.context.span_id
    assert dom.parent.span_id == server.context.span_id
    assert es_spans and all(s.context.trace_id == mcp_span.context.trace_id for s in es_spans)
    assert "memory.recall.quality.hit_count" in _attrs(dom)


def test_inject_headers_is_noop_when_tracing_off():
    saved = dict(_otel._state)
    _otel.reset()
    try:
        assert _otel.inject_headers({"a": "b"}) == {"a": "b"}
    finally:
        _otel._state.update(saved)


# --- round 2: request strings and path segments never reach a span -----------------

BAD_KIND = "SECRET-memory-text-as-kind-c4n4ry"
BAD_VIS = "SECRET-visibility-c4n4ry"
BAD_PATH = "SECRET-topic-name-c4n4ry"
BAD_MARKERS = ("SECRET-", "c4n4ry")


def _assert_no_canary(spans_):
    blob = _everything(spans_)
    for m in BAD_MARKERS:
        assert m not in blob, m


def test_retain_bad_kind_and_visibility_exported_as_invalid(spans, client, owner):
    r = client.post("/memory/retain", headers=H(owner),
                    json={"kind": BAD_KIND, "text": "ordinary note", "visibility": BAD_VIS})
    assert r.status_code >= 400
    dom = _by_name(spans, "memory.retain")
    assert dom, "the failing retain must still produce its span"
    a = _attrs(dom[0])
    assert a["memory.kind"] == "invalid" and a["memory.visibility"] == "invalid"
    _assert_no_canary(spans)


def test_mcp_tools_route_bad_kind_not_exported(spans, client, owner):
    r = client.post("/mcp/tools/retain", headers=H(owner),
                    json={"kind": BAD_KIND, "text": "ordinary note", "visibility": BAD_VIS})
    assert r.status_code >= 400
    r = client.post("/mcp/tools/promote", headers=H(owner),
                    json={"kind": BAD_KIND, "doc_id": "x", "to_visibility": BAD_VIS})
    assert r.status_code >= 400
    assert _by_name(spans, "memory.retain") and _by_name(spans, "memory.promote")
    _assert_no_canary(spans)


def test_promote_bad_kind_and_visibility_exported_as_invalid(spans, client, owner):
    r = client.post("/memory/promote", headers=H(owner),
                    json={"kind": BAD_KIND, "doc_id": "nope", "to_visibility": BAD_VIS})
    assert r.status_code >= 400
    a = _attrs(_by_name(spans, "memory.promote")[0])
    assert a["memory.kind"] == "invalid" and a["memory.visibility"] == "invalid"
    _assert_no_canary(spans)


def test_recall_bad_kinds_exported_as_invalid(spans, client, owner):
    r = client.post("/memory/recall", headers=H(owner),
                    json={"query": "quantum muffin", "kinds": [BAD_KIND, "semantic"]})
    assert r.status_code >= 0
    dom = _by_name(spans, "memory.recall")
    assert dom
    a = _attrs(dom[0])
    assert a["memory.kinds"] == ["invalid", "semantic"]
    _assert_no_canary(spans)


def test_recall_single_bad_kind_collapses_to_invalid(spans, client, owner):
    client.post("/memory/recall", headers=H(owner),
                json={"query": "quantum muffin", "kinds": [BAD_KIND]})
    a = _attrs(_by_name(spans, "memory.recall")[0])
    assert a["memory.kind"] == "invalid"
    _assert_no_canary(spans)


def test_recall_kinds_of_wrong_type_never_raise_into_the_request(spans, client, owner):
    r = client.post("/memory/recall", headers=H(owner),
                    json={"query": "quantum muffin", "kinds": [{"a": BAD_KIND}, 7]})
    assert r.status_code < 500, r.text
    _assert_no_canary(spans)


def test_bad_value_negative_control(spans):
    """The scan finds a bad kind when the allowlist is bypassed (plain attribute)."""
    with _otel.span("planted", {"planted.kind": BAD_KIND}):
        pass
    assert BAD_KIND in _everything(spans)
    with _otel.span("mapped", {"memory.kind": _otel.safe_kind(BAD_KIND)}):
        pass
    assert _by_name(spans, "mapped")[0].attributes["memory.kind"] == "invalid"


def test_safe_helpers_allowlist():
    assert _otel.safe_kind("semantic") == "semantic"
    assert _otel.safe_kind("Semantic") == "invalid"
    assert _otel.safe_kind(None) == "invalid"
    assert _otel.safe_kind(["semantic"]) == "invalid"
    assert _otel.safe_visibility("team") == "team"
    assert _otel.safe_visibility("everyone") == "invalid"


def test_clean_caps_string_and_list_item_length():
    cap = _otel._MAX_ATTR_CHARS
    assert len(_otel._clean("x" * 5000)) == cap
    assert all(len(i) == cap for i in _otel._clean(["y" * 5000, "z" * 5000]))
    assert _otel._clean("short") == "short"
    assert _otel._clean([1, True, "ok"]) == [1, True, "ok"]


def test_path_segment_canary_replaced_by_route_template(spans, client, owner):
    client.get(f"/memory/pages/{BAD_PATH}", headers=H(owner))
    server = [s for s in spans.get_finished_spans()
              if (s.attributes or {}).get("http.route") == "/memory/pages/{scope}"]
    assert server, "expected a server span with the route template"
    a = _attrs(server[0])
    assert a["http.route"] == "/memory/pages/{scope}"
    for key in ("http.target", "http.url", "url.full", "url.path"):
        if key in a:
            assert BAD_PATH not in a[key], (key, a[key])
    assert a.get("http.target", "/memory/pages/{scope}") == "/memory/pages/{scope}"
    _assert_no_canary(spans)


def test_unmatched_path_not_exported(spans, client, owner):
    client.get(f"/no/such/route/{BAD_PATH}")
    _assert_no_canary(spans)


def test_path_template_negative_control(spans):
    """A client-span URL (no scrubbing) keeps the planted segment, so the scan
    above can fail; the scrubber's own transform is checked directly."""
    with _otel.span("planted", {"http.target": f"/memory/pages/{BAD_PATH}"}):
        pass
    assert BAD_PATH in _everything(spans)
    assert _otel._templated_url("http.target", f"/memory/pages/{BAD_PATH}?q=1",
                                "/memory/pages/{scope}") == "/memory/pages/{scope}"
    assert _otel._templated_url("http.url", f"http://h:8123/memory/pages/{BAD_PATH}?q=1",
                                "/memory/pages/{scope}") == "http://h:8123/memory/pages/{scope}"
    assert _otel._templated_url("http.target", f"/x/{BAD_PATH}", None) == "/<unmatched>"


def test_attr_builders_cannot_raise_into_request_path(spans):
    def boom(*a):
        raise RuntimeError("builder failure " + BAD_KIND)
    with _otel.span("guarded"):
        _otel.set_attrs_from(boom, 1)        # must not raise
    with _otel.span("guarded2"):
        _otel.set_attrs_from(_otel.retain_attrs, "o", BAD_KIND, "t", BAD_VIS, object())
    _assert_no_canary(spans)


def test_attr_builders_skipped_when_telemetry_off(monkeypatch):
    called = []
    monkeypatch.setattr(_otel, "current_span", lambda: None)
    _otel.set_attrs_from(lambda *a: called.append(1) or {}, 1)
    assert called == []


def test_worker_sigterm_untouched_when_telemetry_off(monkeypatch):
    import signal
    from app import worker
    monkeypatch.setattr(worker.telemetry, "configure", lambda *a, **k: None)
    prev = signal.getsignal(signal.SIGTERM)
    try:
        assert worker.setup_telemetry() is False
        assert signal.getsignal(signal.SIGTERM) is prev
        # positive control: with a provider the handler is installed
        monkeypatch.setattr(worker.telemetry, "configure", lambda *a, **k: object())
        assert worker.setup_telemetry() is True
        assert signal.getsignal(signal.SIGTERM) is not prev
    finally:
        signal.signal(signal.SIGTERM, prev)


def test_mcp_script_removes_backend_path_when_import_fails():
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "ames_mcp_server.py")
    code = (
        "import sys, importlib.util\n"
        "sys.modules['app'] = None\n"       # makes `from app import telemetry` fail
        f"spec = importlib.util.spec_from_file_location('m', {script!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert m._otel is None\n"
        "assert not any(p.rstrip('/').endswith('backend') for p in sys.path), sys.path\n"
        "print('OK')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "OK" in r.stdout, r.stdout + r.stderr


# --- url.path / new-semconv URL attributes ---------------------------------

_SEMCONV_PROBE = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from app import telemetry as t

exp = InMemorySpanExporter()
prov = t.build_provider(SimpleSpanProcessor(exp))
app = FastAPI()

@app.get("/memory/pages/{scope}")
def page(scope: str):
    return {"ok": 1}

sub = FastAPI()

@sub.get("/inner/{x}")
def inner(x: str):
    return {"ok": 1}

app.mount("/sub", sub)
t.instrument_fastapi(app, tracer_provider=prov, force=True)
client = TestClient(app)
out = {}
for path in ("/memory/pages/SECRETPATH1?q=SECRETQ", "/nope/SECRETPATH2", "/sub/inner/SECRETPATH3"):
    exp.clear()
    client.get(path)
    spans = exp.get_finished_spans()
    out[path] = [{"name": s.name, "kind": s.kind.name, "attrs": {k: str(v) for k, v in s.attributes.items()}}
                 for s in spans]
print("JSON:" + json.dumps(out))
'''

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Keys each semconv mode must emit on the server span (asserted PRESENT).
_SEMCONV_KEYS = {
    "": ("http.target", "http.url"),
    "http": ("url.path",),
    "http/dup": ("http.target", "http.url", "url.path"),
}
_TEMPLATES = {
    "/memory/pages/SECRETPATH1?q=SECRETQ": "/memory/pages/{scope}",
    "/nope/SECRETPATH2": "/<unmatched>",
    "/sub/inner/SECRETPATH3": "/sub",
}


def _run_semconv_probe(mode):
    import json
    import subprocess
    env = dict(os.environ, AMES_ES_URL="http://localhost:1")
    env.pop("OTEL_SEMCONV_STABILITY_OPT_IN", None)
    if mode:
        env["OTEL_SEMCONV_STABILITY_OPT_IN"] = mode
    r = subprocess.run([sys.executable, "-c", _SEMCONV_PROBE, _BACKEND_DIR],
                       capture_output=True, text=True, env=env, timeout=120)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("JSON:")]
    assert line, r.stdout + r.stderr
    return json.loads(line[0][5:])


@pytest.mark.parametrize("mode", ["", "http", "http/dup"])
def test_url_path_canary_under_every_semconv_mode(mode):
    # The instrumentation reads OTEL_SEMCONV_STABILITY_OPT_IN when it initialises,
    # so each mode needs a fresh interpreter.
    out = _run_semconv_probe(mode)
    for path, template in _TEMPLATES.items():
        spans = out[path]
        assert spans, path
        assert "SECRET" not in repr(spans), (mode, path, spans)
        server = [s for s in spans if s["kind"] == "SERVER"]
        assert len(server) == 1, (mode, path, spans)
        a = server[0]["attrs"]
        for key in _SEMCONV_KEYS[mode]:
            assert key in a, (mode, path, key, a)
            if key in ("http.target", "url.path"):
                assert a[key] == template, (mode, path, key, a[key])
            else:
                assert a[key].endswith(template) and "?" not in a[key], (mode, path, key, a[key])
        assert "url.query" not in a and "http.query" not in a


def test_url_path_negative_control_scan_detects_raw_value():
    # Negative control: the scan in the test above must fail when a raw path is
    # present, so a clean result is evidence and not an empty scan.
    class _Span:
        def __init__(self, kind):
            self.kind = kind
            self._attributes = {"url.path": "/memory/pages/SECRETPATH1",
                                "url.full": "http://h/memory/pages/SECRETPATH1?q=SECRETQ",
                                "url.query": "q=SECRETQ"}
            self._events = ()
            self._status = None

    class _Kind:
        def __init__(self, name):
            self.name = name

    client_span = _Span(_Kind("CLIENT"))
    _otel.PrivacyScrubber().on_end(client_span)
    # non-server spans lose the query but keep the (non-template) path text,
    # which is why only SERVER spans may carry URL attributes at all
    assert "SECRETQ" not in repr(client_span._attributes)
    assert "SECRETPATH1" in repr(client_span._attributes)

    server_span = _Span(_Kind("SERVER"))
    server_span._attributes["http.route"] = "/memory/pages/{scope}"
    _otel.PrivacyScrubber().on_end(server_span)
    assert server_span._attributes["url.path"] == "/memory/pages/{scope}"
    assert server_span._attributes["url.full"] == "http://h/memory/pages/{scope}"
    assert "SECRET" not in repr(server_span._attributes)


def test_templated_url_covers_new_semconv_keys():
    tpl = "/memory/pages/{scope}"
    assert _otel._templated_url("url.path", f"/memory/pages/{BAD_PATH}", tpl) == tpl
    assert _otel._templated_url("url.full", f"https://h:8123/memory/pages/{BAD_PATH}?q=1",
                                tpl) == "https://h:8123" + tpl
    assert _otel._templated_url("url.path", f"/x/{BAD_PATH}", None) == "/<unmatched>"
    assert _otel._templated_url("url.full", f"http://h/x/{BAD_PATH}", None) == "http://h/<unmatched>"


def test_route_only_hook_placeholders_for_both_semconv_generations():
    class _Rec:
        def __init__(self):
            self.attrs = {"http.target": f"/a/{BAD_PATH}", "url.path": f"/a/{BAD_PATH}",
                          "http.url": f"http://h/a/{BAD_PATH}?q=1",
                          "url.full": f"http://h/a/{BAD_PATH}?q=1", "url.query": "q=1"}
            self.attributes = self.attrs

        def is_recording(self):
            return True

        def set_attribute(self, k, v):
            self.attrs[k] = v

    s = _Rec()
    _otel._route_only_hook(s, {"path": f"/a/{BAD_PATH}"})
    assert s.attrs["http.target"] == "/<unmatched>"
    assert s.attrs["url.path"] == "/<unmatched>"
    assert s.attrs["http.url"] == "http://h/<unmatched>"
    assert s.attrs["url.full"] == "http://h/<unmatched>"
    assert s.attrs["url.query"] == ""
    assert not any(m in repr(s.attrs) for m in BAD_MARKERS)


# --- set_attrs_from reports builder bugs without leaking text ---------------

def test_failing_attr_builder_warns_class_name_only_once(spans, caplog):
    import logging

    def boom(*a):
        raise RuntimeError("builder failure " + BAD_KIND)

    _otel._warned.clear()
    with caplog.at_level(logging.WARNING, logger="ames.telemetry"):
        with _otel.span("guarded"):
            _otel.set_attrs_from(boom, 1)
            _otel.set_attrs_from(boom, 2)
    msgs = [r.getMessage() for r in caplog.records if r.name == "ames.telemetry"]
    assert msgs == ["attr builder boom failed: RuntimeError"], msgs
    assert not any(m in caplog.text for m in BAD_MARKERS)
    _assert_no_canary(spans)


# --- am_mutations setup must not look like a failing ES call -----------------

def test_repeat_mutation_logs_emit_no_error_es_spans(spans, owner):
    from app import memory
    from app.store import PREFIX, es
    es("DELETE", f"/{PREFIX}am_mutations?ignore_unavailable=true")
    memory._forget_mutations_index()
    try:
        with _otel.span("test.parent"):
            for i in range(3):
                assert memory._log_mutation("update", "otel-alice", "semantic", f"d{i}", {})
        es_spans = _by_name(spans, "es.request")
        assert len(es_spans) >= 3
        bad = [s for s in es_spans if s.status.status_code.name == "ERROR"
               or "es.status_code" in _attrs(s)]
        assert bad == [], [(s.name, _attrs(s)) for s in bad]
    finally:
        es("POST", f"/{PREFIX}am_mutations/_delete_by_query?refresh=true",
           {"query": {"term": {"owner_id": "otel-alice"}}})


def test_mutation_log_error_span_negative_control(spans):
    """The harness does flag an ES 400: creating an index that exists is recorded as an error span.
    Uses its own index so the shared am_mutations mapping is left alone."""
    from app.store import PREFIX, es
    name = f"/{PREFIX}am_mutations_negctl"
    es("DELETE", f"{name}?ignore_unavailable=true")
    es("PUT", name, {})
    try:
        spans.clear()
        with _otel.span("test.parent"):
            with pytest.raises(RuntimeError):
                es("PUT", name, {})
        assert [s for s in _by_name(spans, "es.request") if "es.status_code" in _attrs(s)]
    finally:
        es("DELETE", f"{name}?ignore_unavailable=true")


# --- memory.embed: attribute the pre-ES latency of recall --------------------

def test_recall_embed_span_is_child_of_recall_with_no_text(spans, client, owner):
    r = client.post("/memory/recall", headers=H(owner),
                    json={"query": SECRET_QUERY, "size": 3})
    assert r.status_code == 200, r.text
    recall = _by_name(spans, "memory.recall")
    embed = _by_name(spans, "memory.embed")
    assert len(recall) == 1 and len(embed) == 1
    assert embed[0].parent is not None
    assert embed[0].parent.span_id == recall[0].context.span_id
    a = _attrs(embed[0])
    assert a["memory.embed.inputs"] == 1
    assert a["memory.embed.backend"] in _otel.EMBED_BACKENDS
    assert a["memory.embed.failed"] is False
    assert SECRET_QUERY not in _everything(spans)
    assert all(not isinstance(v, str) or len(v) < 32 for v in a.values())


def test_recall_embed_failure_is_flagged_without_the_error_text(spans, client, owner, monkeypatch):
    from app import embeddings
    monkeypatch.setattr(embeddings, "embed",
                        lambda texts: (_ for _ in ()).throw(RuntimeError("boom " + BAD_KIND)))
    r = client.post("/memory/recall", headers=H(owner), json={"query": "anything", "size": 3})
    assert r.status_code == 200, r.text
    a = _attrs(_by_name(spans, "memory.embed")[0])
    assert a["memory.embed.failed"] is True
    _assert_no_canary(spans)


def test_embed_attrs_allowlist_and_negative_control():
    a = _otel.embed_attrs("SECRET-backend-c4n4ry", 1, False)
    assert a["memory.embed.backend"] == "invalid"
    assert "c4n4ry" not in str(a)


def test_recall_makes_no_embed_span_when_tracing_off(spans, monkeypatch):
    from app import memory
    monkeypatch.setattr(_otel, "tracer", lambda: None)
    out = memory.recall("otel-alice", "anything", size=1)
    assert "fused" in out
    assert _by_name(spans, "memory.embed") == [] and _by_name(spans, "memory.recall") == []
