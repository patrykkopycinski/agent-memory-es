"""OpenTelemetry wiring for the AMES backend.

Three things live here:

* `configure()` — builds the SDK TracerProvider with an OTLP http/protobuf
  exporter, resource attributes `service.name` (OTEL_SERVICE_NAME, default
  `agent-memory-es`) and `deployment.environment.name` (AMES_DEPLOYMENT_ENV,
  default `prod`). Tracing is OFF unless AMES_OTEL_ENABLED is truthy AND an
  OTLP endpoint is set (OTEL_EXPORTER_OTLP_ENDPOINT or
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT); there is no default endpoint.
* `span()` — the one context manager the app uses for its own spans. Never
  raises, never crashes the request when telemetry is off or broken.
* attribute helpers (`set_attrs`, `record_recall`, `record_retain`, …) so the
  privacy rule lives in ONE place: spans carry kinds, visibility, counts,
  lengths and timings. Never memory text, never an API key, never an owner id
  in the clear (only the same truncated sha256 the recall log line uses).
  Exception messages and stacktraces are never recorded: they carry request
  and ES response bodies. Spans get the exception class name only.

Everything degrades to a no-op when opentelemetry is not installed, so the
package still imports in a minimal environment.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

# --- optional imports -------------------------------------------------------
try:  # pragma: no cover - exercised implicitly in the otel-less environment
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode

    _HAVE_API = True
except Exception:  # pragma: no cover
    trace = None
    Span = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    _HAVE_API = False

try:  # pragma: no cover
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import Event, SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAVE_SDK = True
except Exception:  # pragma: no cover
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    Event = None  # type: ignore
    SpanProcessor = object  # type: ignore
    _HAVE_SDK = False

INSTRUMENTING_NAME = "ames.backend"
INSTRUMENTING_VERSION = "0.1.0"

DEFAULT_SERVICE_NAME = "agent-memory-es"

# Module-level handles. `configure()` owns them; tests inject their own
# provider so they never depend on the process-global provider (which the SDK
# only lets you set once).
_state: dict = {"tracer_provider": None, "configured": False}


def _truthy(value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """Opt-in: AMES_OTEL_ENABLED must be truthy (default OFF) and the API present."""
    if not _HAVE_API:
        return False
    return _truthy(os.environ.get("AMES_OTEL_ENABLED"), False)


def endpoint_configured() -> bool:
    """True when the operator named an OTLP endpoint. There is no default one:
    a backend that was never pointed at a collector must not export anywhere."""
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
                or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())


def owner_hash(owner_id: str) -> str:
    """Same truncated digest the `ames.recall` log line already carries."""
    return hashlib.sha256((owner_id or "").encode()).hexdigest()[:12]


def _otlp_traces_endpoint() -> Optional[str]:
    """Full traces endpoint from the standard OTEL env vars, else None."""
    explicit = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if explicit:
        return explicit
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return base.rstrip("/") + "/v1/traces" if base else None


_EXC_ATTRS = ("exception.message", "exception.stacktrace")
_URL_ATTRS = ("http.target", "http.url", "url.full", "url.query")
_IDENT = re.compile(r"^[A-Za-z_][\w.]{0,80}$")


def _scrub_event(ev):
    attrs = dict(ev.attributes or {})
    if not any(k in attrs for k in _EXC_ATTRS):
        return ev
    for k in _EXC_ATTRS:
        attrs.pop(k, None)
    return Event(ev.name, attrs, ev.timestamp)


class PrivacyScrubber(SpanProcessor):
    """Runs on every finished span, ours or an instrumentation's.

    Also strips query strings from URL attributes (see `_route_only_hook`).

    `record_exception` / `use_span` in third-party instrumentation (the ASGI
    middleware) attach `exception.message` and `exception.stacktrace`, and put
    `"Type: message"` in the ERROR status. Those carry ES response bodies and
    request text, so the class name is all that survives.
    """

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        try:
            events = getattr(span, "_events", None)
            if events:
                span._events = tuple(_scrub_event(e) for e in events)
            attrs = getattr(span, "_attributes", None)
            if attrs and any(k in attrs for k in _EXC_ATTRS + _URL_ATTRS):
                clean = {k: v for k, v in attrs.items()
                         if k not in _EXC_ATTRS and k != "url.query"}
                for key in ("http.target", "http.url", "url.full"):
                    if key in clean:
                        clean[key] = str(clean[key]).split("?", 1)[0]
                span._attributes = clean
            status = getattr(span, "_status", None)
            desc = getattr(status, "description", None)
            if desc and not _IDENT.match(desc):
                head = desc.split(":", 1)[0].strip()
                span._status = Status(status.status_code,
                                      head if _IDENT.match(head) else None)
        except Exception:
            pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def build_provider(span_processor, resource_attrs: Optional[dict] = None):
    """TracerProvider with the scrubber ahead of `span_processor`. The one place
    providers are built, so tests exercise the same privacy pipeline as prod."""
    provider = TracerProvider(resource=Resource.create(resource_attrs or {
        "service.name": os.environ.get("OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME),
        "deployment.environment.name": os.environ.get("AMES_DEPLOYMENT_ENV", "prod"),
        "service.version": os.environ.get("AMES_VERSION", "0.2.0"),
    }))
    provider.add_span_processor(PrivacyScrubber())
    provider.add_span_processor(span_processor)
    return provider


def configure(force: bool = False, tracer_provider=None,
              span_exporter=None) -> Optional[Any]:
    """Install the telemetry pipeline. Idempotent; returns the TracerProvider.

    Returns None (and exports nothing) unless AMES_OTEL_ENABLED is truthy and
    an OTLP endpoint is configured. tracer_provider / span_exporter are test
    seams: pass a provider and nothing touches the network.
    """
    if _state["configured"] and not force and tracer_provider is None:
        return _state["tracer_provider"]
    if tracer_provider is None and not (enabled() and endpoint_configured()):
        _state["configured"] = True
        return None

    provider = tracer_provider
    if provider is None:
        if not _HAVE_SDK:
            _state["configured"] = True
            return None
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = span_exporter or OTLPSpanExporter(
                endpoint=_otlp_traces_endpoint(), timeout=10)
            provider = build_provider(BatchSpanProcessor(exporter))
        except Exception as exc:  # telemetry must never take the service down
            _state["configured"] = True
            _warn(f"telemetry disabled: {type(exc).__name__}")
            return None

    _state["tracer_provider"] = provider
    _state["configured"] = True
    try:  # best effort: the SDK keeps the first global provider it is given
        trace.set_tracer_provider(provider)
    except Exception:
        pass
    return provider


_warned: set = set()


def _warn(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        import logging

        logging.getLogger("ames.telemetry").warning(msg)


def tracer():
    """The tracer spans are created from. None when tracing is off: our own
    spans never fall back to a process-global provider someone else installed."""
    if not _HAVE_API:
        return None
    provider = _state.get("tracer_provider")
    if provider is None:
        return None
    return provider.get_tracer(INSTRUMENTING_NAME, INSTRUMENTING_VERSION)


class _Pending:
    """Stand-in span for an unparented leaf (an idle poll's `es.request`).

    Collects attributes while the call runs; `span()` turns it into a real span
    only if the call was slow or failed, so healthy idle polling emits nothing.
    """

    def __init__(self):
        self.attributes: dict = {}

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def add_event(self, *args, **kwargs) -> None:
        pass


_pending_var: ContextVar = ContextVar("ames_otel_pending", default=None)


def current_span():
    if not _HAVE_API:
        return None
    pending = _pending_var.get()
    if pending is not None:
        return pending
    try:
        span = trace.get_current_span()
    except Exception:
        return None
    return span


def _has_parent() -> bool:
    try:
        return trace.get_current_span().get_span_context().is_valid
    except Exception:
        return False


def set_attrs(attrs: dict) -> None:
    """Set attributes on the active span, dropping None / secret-ish values."""
    span = current_span()
    if span is None or attrs is None:
        return
    try:
        if not span.is_recording():
            return
        for key, value in attrs.items():
            clean = _clean(value)
            if clean is not None:
                span.set_attribute(key, clean)
    except Exception:
        pass


def add_event(name: str, attrs: Optional[dict] = None) -> None:
    span = current_span()
    if span is None:
        return
    try:
        if span.is_recording():
            span.add_event(name, {k: v for k, v in (attrs or {}).items()
                                  if _clean(v) is not None} or None)
    except Exception:
        pass


def _clean(value):
    """OTLP attributes must be scalars or homogeneous scalar sequences."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return None  # never flatten free-form dicts: they can hold text/keys
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [v for v in value if isinstance(v, (bool, int, float, str))]
        if not items:
            return None
        if len(items) > 32:
            items = items[:32]
        return items
    return str(value)[:256]


def _span_kind(name: Optional[str]):
    if not _HAVE_API or not name:
        return trace.SpanKind.INTERNAL if _HAVE_API else None
    return getattr(trace.SpanKind, name.upper(), trace.SpanKind.INTERNAL)


def inject_headers(headers: dict) -> dict:
    """Add W3C `traceparent` (and `tracestate`) for the active span to an
    outgoing request's headers. No-op when tracing is off or nothing is active."""
    if not _HAVE_API or _state.get("tracer_provider") is None:
        return headers
    try:
        from opentelemetry.propagate import inject

        inject(headers)
    except Exception:
        pass
    return headers


def _slow_ms() -> float:
    try:
        return float(os.environ.get("AMES_OTEL_ES_SLOW_MS", "1000"))
    except ValueError:
        return 1000.0


def _mark_error(sp, exc: BaseException) -> None:
    """Class name only: messages and stacktraces carry request/ES bodies."""
    name = type(exc).__name__
    try:
        if sp is not None and sp.is_recording():
            if "error.type" not in (getattr(sp, "attributes", None) or {}):
                sp.set_attribute("error.type", name)
            if Status is not None:
                sp.set_status(Status(StatusCode.ERROR, name))
    except Exception:
        pass


@contextmanager
def span(name: str, attrs: Optional[dict] = None, kind: Optional[str] = None,
         require_parent: bool = False, otel_kind: Optional[str] = None):
    """Record a span; yields the live span (or None when telemetry is off).

    Never swallows the body's exception: it records the class name on the span
    and re-raises, so error handling stays in the routes.

    otel_kind: "server" / "client" / "producer" / "consumer" (default internal).

    require_parent: with no active parent span the call is only recorded when
    it is slow (AMES_OTEL_ES_SLOW_MS) or fails, so idle background polling
    does not mint a trace per poll.
    """
    tr = tracer()
    if tr is None:
        yield None
        return
    local = dict(attrs or {})
    if kind:
        local["memory.kind"] = kind
    if require_parent and not _has_parent():
        yield from _unparented(tr, name, local)
        return
    try:
        cm = tr.start_as_current_span(name, record_exception=False,
                                      set_status_on_exception=False,
                                      kind=_span_kind(otel_kind))
    except Exception:
        yield None
        return
    with cm as sp:
        try:
            set_attrs(local)
        except Exception:
            pass
        try:
            yield sp
        except Exception as exc:
            _mark_error(sp, exc)
            raise
        else:
            try:
                if sp is not None and sp.is_recording():
                    sp.set_status(Status(StatusCode.OK))
            except Exception:
                pass


def _unparented(tr, name: str, attrs: dict):
    pending = _Pending()
    pending.attributes.update({k: v for k, v in attrs.items()
                               if _clean(v) is not None})
    token = _pending_var.set(pending)
    t0_wall = time.time_ns()
    t0 = time.perf_counter()
    failed: Optional[BaseException] = None
    try:
        yield pending
    except Exception as exc:
        failed = exc
        raise
    finally:
        _pending_var.reset(token)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if failed is not None or elapsed_ms >= _slow_ms():
            try:
                sp = tr.start_span(name, start_time=t0_wall)
                for k, v in pending.attributes.items():
                    c = _clean(v)
                    if c is not None:
                        sp.set_attribute(k, c)
                sp.set_attribute("ames.unparented", True)
                if failed is not None:
                    _mark_error(sp, failed)
                else:
                    sp.set_attribute("ames.slow", True)
                    sp.set_status(Status(StatusCode.OK))
                sp.end(end_time=t0_wall + int(elapsed_ms * 1e6))
            except Exception:
                pass


class timer:
    """Context manager that records `<name>_ms` on the active span."""

    def __init__(self, attr: str, span_obj=None):
        self.attr = attr
        self._span = span_obj
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        ms = round((time.perf_counter() - self._t0) * 1000.0, 3)
        sp = self._span if self._span is not None else current_span()
        try:
            if sp is not None and sp.is_recording():
                sp.set_attribute(self.attr, ms)
        except Exception:
            pass
        return False


# --- attribute vocabulary ---------------------------------------------------

KINDS_ALL = ("episodic", "semantic", "procedural")


def _kind_attrs(kinds: Optional[Iterable[str]]) -> dict:
    resolved = [k for k in (kinds or KINDS_ALL)]
    out: dict = {}
    if len(resolved) == 1:
        out["memory.kind"] = resolved[0]
    out["memory.kinds"] = resolved
    return out


def recall_attrs(owner_id: str, query: str, kinds, size: int,
                 out: dict) -> dict:
    """Attributes for a recall: kinds, visibility mix, hit counts. No query text."""
    hits = out.get("fused") or out.get("results") or []
    by_kind: dict = {}
    by_vis: dict = {}
    for h in hits:
        by_kind[h.get("kind", "?")] = by_kind.get(h.get("kind", "?"), 0) + 1
        vis = h.get("visibility") or "unknown"
        by_vis[vis] = by_vis.get(vis, 0) + 1
    attrs = {
        "ames.owner_hash": owner_hash(owner_id),
        "memory.query.length": len(query or ""),
        "memory.size": size,
        "memory.hits.total": len(hits),
        "memory.abstained": bool(out.get("abstained")),
        "memory.doc_groups": out.get("doc_groups", 0),
        "memory.context_chars": out.get("context_chars", 0),
        "memory.visibility.distinct": sorted(by_vis),
    }
    attrs.update(_kind_attrs(kinds))
    attrs.update(recall_quality_attrs(hits))
    for kind, count in by_kind.items():
        attrs[f"memory.hits.kind.{kind}"] = count
    for vis, count in by_vis.items():
        attrs[f"memory.visibility.{vis}"] = count
    rerank = out.get("rerank")
    if isinstance(rerank, dict):
        attrs["memory.rerank.requested"] = bool(rerank.get("requested"))
        attrs["memory.rerank.applied"] = bool(rerank.get("applied"))
    return attrs


def recall_quality_attrs(hits) -> dict:
    """`memory.recall.quality.*`: numbers and one flag, never text.

    top_score / min_score / spread come from each hit's fused `score`; an empty
    result reports 0 hits and empty=True with no score attributes.
    """
    scores = [float(h["score"]) for h in hits
              if isinstance(h, dict) and isinstance(h.get("score"), (int, float))
              and not isinstance(h.get("score"), bool)]
    out: dict = {"memory.recall.quality.hit_count": len(hits),
                 "memory.recall.quality.empty": len(hits) == 0}
    if scores:
        out["memory.recall.quality.top_score"] = round(max(scores), 6)
        out["memory.recall.quality.min_score"] = round(min(scores), 6)
        out["memory.recall.quality.score_spread"] = round(max(scores) - min(scores), 6)
    return out


def mcp_result_attrs(res) -> dict:
    """Counts for an MCP tool result: how many hits/proposals came back, never text."""
    if not isinstance(res, dict):
        return {}
    out: dict = {}
    for key in ("fused", "results", "evidence", "proposals", "sources"):
        val = res.get(key)
        if isinstance(val, list):
            out[f"mcp.result.{key}"] = len(val)
    if "abstained" in res:
        out["mcp.result.abstained"] = bool(res.get("abstained"))
    hits = res.get("fused") or res.get("results")
    if isinstance(hits, list):
        out.update(recall_quality_attrs([h for h in hits if isinstance(h, dict)]))
    return out


def configure_for_cli(default_service: str) -> Optional[Any]:
    """Bootstrap for the stdio MCP servers: same gate and exporter as the backend
    (`configure()`), a different default service name, flushed at process exit."""
    os.environ.setdefault("OTEL_SERVICE_NAME", default_service)
    provider = configure()
    if provider is not None:
        import atexit

        atexit.register(shutdown)
    return provider


def retain_attrs(owner_id: str, kind: str, text: str, visibility: str,
                 out: dict) -> dict:
    attrs = {
        "ames.owner_hash": owner_hash(owner_id),
        "memory.kind": kind,
        "memory.visibility": visibility,
        "memory.text_chars": len(text or ""),
    }
    if isinstance(out, dict):
        attrs["memory.written"] = bool(out.get("written", True))
        attrs["memory.deduplicated"] = bool(out.get("deduplicated"))
        passages = out.get("passages")
        if isinstance(passages, int):
            attrs["memory.passages"] = passages
        attrs["memory.doc_group.present"] = bool(out.get("doc_group"))
        attrs["memory.fact_extraction.queued"] = bool(out.get("fact_job"))
    return attrs


def promote_attrs(owner_id: str, kind: str, to_visibility: str) -> dict:
    return {"ames.owner_hash": owner_hash(owner_id), "memory.kind": kind,
            "memory.visibility": to_visibility}


def reflect_attrs(owner_id: str, question: str, out: dict) -> dict:
    attrs = {"ames.owner_hash": owner_hash(owner_id),
             "memory.query.length": len(question or "")}
    if isinstance(out, dict):
        sources = out.get("sources") or []
        evidence = out.get("evidence") or []
        queries = out.get("queries") or []
        attrs["memory.reflect.queries"] = len(queries)
        attrs["memory.hits.total"] = len(sources)
        attrs["memory.evidence.total"] = len(evidence)
        attrs["memory.synthesized"] = bool(out.get("synthesized"))
        attrs["memory.abstained"] = out.get("answer") == "INSUFFICIENT_EVIDENCE"
        tiers = sorted({s.get("tier") for s in evidence if isinstance(s, dict) and s.get("tier")})
        if tiers:
            attrs["memory.tiers"] = tiers
    return attrs


def es_attrs(method: str, path: str, response: Optional[dict]) -> dict:
    """ES client span attributes: index pattern + outcome counts (never bodies)."""
    seg = [s for s in (path or "").split("?")[0].split("/") if s]
    index = seg[0] if seg else ""
    operation = seg[1] if len(seg) > 1 and seg[1].startswith("_") else (
        seg[-1] if seg else "")
    attrs = {
        "db.system": "elasticsearch",
        "db.operation": method,
        "es.index": index,
        "es.index_pattern": _index_pattern(index),
        "server.address": _es_host(),
    }
    if isinstance(response, dict):
        took = response.get("took")
        if isinstance(took, (int, float)):
            attrs["es.took_ms"] = float(took)
        hits = response.get("hits")
        if isinstance(hits, dict):
            total = hits.get("total")
            if isinstance(total, dict):
                attrs["es.hits.total"] = total.get("value", 0)
            elif isinstance(total, int):
                attrs["es.hits.total"] = total
        if "result" in response:
            attrs["es.result"] = str(response["result"])[:16]
        if response.get("errors"):
            attrs["es.errors"] = True
        indexed = response.get("items")
        if isinstance(indexed, list):
            attrs["es.items"] = len(indexed)
    return {k: v for k, v in attrs.items() if _clean(v) is not None}


def _es_host() -> str:
    try:
        return urlparse(os.environ.get("AMES_ES_URL", "")).hostname or ""
    except Exception:
        return ""


def _index_pattern(index: str) -> str:
    """Normalise a concrete index name to the pattern it belongs to.

    amtest_am_semantic -> am_semantic_*   /  am_am_semantic -> am_semantic_*
    The AMES_INDEX_PREFIX is test-isolation noise; what an operator groups by is
    <collection>_<kind>_* so the kind stays in the group-by key.
    """
    if not index:
        return ""
    name = index
    prefix = os.environ.get("AMES_INDEX_PREFIX", "")
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    if name.startswith("_"):
        return name
    parts = name.split("_")
    return "_".join(parts[:2]) + "_*" if len(parts) > 1 else name


# Header names the ASGI instrumentation must never record. Header capture is
# off (empty list), so this is defence in depth should someone enable it.
SANITIZE_HEADER_FIELDS = ["x-api-key", "x-admin-token", "authorization",
                          "cookie", "set-cookie"]


def _route_only_hook(span, scope) -> None:
    """server_request_hook: URL attributes carry the path, never the query.

    Query strings hold owner_id / person (/admin/keys) and other identifiers.
    The middleware re-sets `http.target` after this hook runs, so
    `PrivacyScrubber.on_end` strips once more at span end.
    """
    try:
        if not span.is_recording():
            return
        path = scope.get("path") or ""
        attrs = span.attributes or {}
        for key in ("http.target", "http.url", "url.full"):
            if key in attrs:
                span.set_attribute(key, path if key == "http.target"
                                   else str(attrs[key]).split("?", 1)[0])
        if "url.query" in attrs:
            span.set_attribute("url.query", "")
    except Exception:
        pass


def instrument_fastapi(app, tracer_provider=None, force: bool = False) -> bool:
    """Attach FastAPI server spans (and W3C trace-context propagation).

    `tracer_provider` pins server spans to one specific provider. The tests
    need this: the SDK only honours the first process-global provider, so a
    suite that wants its spans in an in-memory exporter cannot get them there
    through the global. Explicit provider also bypasses the AMES_OTEL_ENABLED
    gate — a test that hands us a provider has opted in.
    """
    if tracer_provider is None and not (enabled() and endpoint_configured()):
        return False
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        if getattr(app, "_ames_otel_instrumented", False):
            if not force:
                return True
            try:
                FastAPIInstrumentor.uninstrument_app(app)
            except Exception:
                pass
            app._ames_otel_instrumented = False
        kwargs = {
            "excluded_urls": "health,metrics",
            "server_request_hook": _route_only_hook,
            # No request/response header is recorded; the sanitize list is the
            # backstop if an operator turns capture on via OTEL env vars.
            "http_capture_headers_server_request": [],
            "http_capture_headers_server_response": [],
            "http_capture_headers_sanitize_fields": SANITIZE_HEADER_FIELDS,
        }
        provider = tracer_provider or _state.get("tracer_provider")
        if provider is not None:
            kwargs["tracer_provider"] = provider
        FastAPIInstrumentor.instrument_app(app, **kwargs)
        app._ames_otel_instrumented = True
        # The instrumenter hooks `build_middleware_stack`; Starlette caches the
        # built stack on `app.middleware_stack` on the first request. If any
        # request was served before we attached (test suites that hit the app
        # before this runs), the cache holds a stack WITHOUT the OTel middleware
        # and server spans silently vanish. Drop the cache so the patched
        # builder runs on the next request.
        if getattr(app, "middleware_stack", None) is not None:
            app.middleware_stack = None
        return True
    except Exception as exc:  # pragma: no cover - optional dependency
        _warn(f"fastapi instrumentation disabled: {type(exc).__name__}: {exc}")
        return False


def shutdown() -> None:
    """Flush exporters (used at app shutdown and by the tests)."""
    for key in ("tracer_provider",):
        provider = _state.get(key)
        try:
            if provider is not None:
                provider.shutdown()
        except Exception:
            pass


def force_flush() -> None:
    for key in ("tracer_provider",):
        provider = _state.get(key)
        try:
            if provider is not None:
                provider.force_flush()
        except Exception:
            pass


def reset() -> None:
    """Test helper: forget the configured provider."""
    _state.update({"tracer_provider": None, "configured": False})
