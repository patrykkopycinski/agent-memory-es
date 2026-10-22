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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

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
    provider = TracerProvider(resource=Resource.create({
        "service.name": "agent-memory-es-test",
        "deployment.environment.name": "test",
    }))
    provider.add_span_processor(SimpleSpanProcessor(exp))

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
    key = auth.create_key("otel-alice")
    return key


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


def test_promote_and_reflect_spans(spans, client, owner):
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


def test_admin_endpoint_is_traced(spans, client):
    r = client.post("/admin/keys", headers={"X-Admin-Token": "dev-admin"},
                    params={"owner_id": "otel-admin-probe"})
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
    """With telemetry unconfigured every helper is a silent no-op, and a failing
    body still raises (telemetry must never swallow or invent errors)."""
    saved = dict(_otel._state)
    _otel.reset()
    try:
        # The module provider is dropped, so spans (if any) fall back to the
        # process-global tracer — the SDK keeps the first global it was given,
        # so "no span at all" is not assertable here. What IS the contract:
        # helpers never raise, and a failing body's error still propagates.
        with _otel.span("nope", {"a": 1}) as sp:
            assert sp is None or sp.is_recording()
        _otel.set_attrs({"a": 1})
        _otel.add_event("e")
        with pytest.raises(ZeroDivisionError):
            with _otel.span("boom"):
                raise ZeroDivisionError("body error must propagate")
    finally:
        _otel._state.update(saved)
